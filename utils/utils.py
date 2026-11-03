import re
import json
import os
from datetime import datetime

_local_model = None
_local_tokenizer = None
_local_model_path = None


def _resolve_model_path(model_path):
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    fp16_path = os.path.join(model_path, "FP16")
    if os.path.isfile(os.path.join(fp16_path, "config.json")):
        return fp16_path

    raise FileNotFoundError(
        f"No config.json found in {model_path} or its FP16 subdirectory. "
        "Set LLM_MODEL_PATH to the directory containing the model config."
    )


def _load_local_model(model_path):
    global _local_model, _local_tokenizer, _local_model_path

    model_path = _resolve_model_path(model_path)
    if _local_model is None or _local_model_path != model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        if _local_tokenizer.pad_token_id is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        _local_model_path = model_path

    return _local_model, _local_tokenizer


def generate_local_text(messages, model_path=None, max_new_tokens=512):
    import torch

    model_path = model_path or os.environ.get("LLM_MODEL_PATH")
    if not model_path:
        raise ValueError("Set LLM_MODEL_PATH to a local Hugging Face causal language model.")
    model_path = _resolve_model_path(model_path)

    model, tokenizer = _load_local_model(model_path)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load local model from {model_path}")

    if tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        prompt = "\n\n".join(
            f"{message['role'].capitalize()}:\n{message['content']}" for message in messages
        )
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()

def call_gpt(cur_prompt, stop=None, model="gpt-4o-mini"):
    local_model_path = os.environ.get("LLM_MODEL_PATH")
    if local_model_path:
        return generate_local_text(
            [{"role": "user", "content": cur_prompt}],
            model_path=local_model_path,
        )

    from openai import OpenAI

    reasoner_messages = [
        {
            "role": "user",
            "content": cur_prompt
        },
    ]
    completion = OpenAI().chat.completions.create(
        model=model,
        messages=reasoner_messages,
    )
    returned = completion.choices[0].message.content
    return returned


def extract_ans(input_string):
    """
    Extract all contents inside angle brackets <> from the given string.

    Args:
        input_string (str): The input string containing angle brackets.

    Returns:
        list: A list of strings extracted from inside the angle brackets.
    """
    # Use regex to find all matches between angle brackets and trim extra whitespace
    matches = re.findall(r'<\s*(.*?)\s*>', input_string)
    return matches[0]


class DataHandler:
    def __init__(self, task_description, result_folder):
        # Get the current time and format it as the folder name
        self.folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Check if the result folder exists, if not, create it
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        # Build the full path of the log folder
        self.folder_path = os.path.join(result_folder, self.folder_name)
        # Create the log folder, if it already exists, do not raise an error
        os.makedirs(self.folder_path, exist_ok=True)
        # Build the full path of the log file
        self.log_file_path = os.path.join(self.folder_path, "log.jsonl")
        # Open the log file in write mode
        self.log_file = open(self.log_file_path, "w")
        # Create the README file and write the task description
        self.create_readme(task_description)

    def create_readme(self, task_description):
        """
        Create a README file in the log folder and write the task description into it.
        :param task_description: The description of the task to be written into the README file.
        """
        readme_path = os.path.join(self.folder_path, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(f"# Task Description\n{task_description}")

    def log_iteration(self, data):
        """
        Write data in JSON format to the log file.
        :param data: The data to be written, which should be a JSON serializable object.
        """
        json.dump(data, self.log_file)
        self.log_file.write("\n")

    def save_file(self, file_name, content):
        """
        Save content to a file in the log folder.
        :param file_name: The name of the file to save.
        :param content: The content to be saved, should be in bytes.
        """
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, "w") as f:
            json.dump(content, f, indent=4)

    def close(self):
        """
        Close the log file.
        """
        self.log_file.close()


def completion_finetune(model, messages):
    local_model_path = os.environ.get("LLM_MODEL_PATH")
    if local_model_path:
        return generate_local_text(messages, model_path=local_model_path)

    from openai import OpenAI

    # Call the OpenAI API to generate completion
    completion = OpenAI().chat.completions.create(
        model=model,
        messages=messages
    )
    # Get the content of the first choice from the completion result
    return completion.choices[0].message.content