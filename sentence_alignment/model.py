import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig, PretrainedConfig


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"

    def __init__(self, dropout=0.2, encoder_name="FacebookAI/roberta-large", encoder_config=None, **kwargs):
        self.dropout = dropout
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config
        super().__init__(**kwargs)


class Aligner(PreTrainedModel):
    config_class = AlignerConfig

    def __init__(self, config, tokenizer=None):
        super(Aligner, self).__init__(config)

        if config.encoder_config:
            encoder_config = dict(config.encoder_config)
            model_type = encoder_config.pop("model_type")
            self.sentence_encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **encoder_config))
        else:
            self.sentence_encoder = AutoModel.from_pretrained(config.encoder_name)
        if tokenizer is not None:
            self.sentence_encoder.resize_token_embeddings(len(tokenizer))
        config.encoder_config = self.sentence_encoder.config.to_dict()
        self.hidden_size = self.sentence_encoder.config.hidden_size

        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.sentence_encoder.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, special_token_positions, labels=None, labels_mask=None):
        outputs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        batch_size = hidden_states.size(0)

        special_token_embeddings = hidden_states[
            torch.arange(batch_size, device=hidden_states.device).unsqueeze(1),
            special_token_positions
        ]  # (batch_size, num_sentences, hidden_size)

        logits = self.classification_head(special_token_embeddings).squeeze(-1)  # (batch_size, num_sentences)

        output = {"logits": logits}
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fn(logits, labels)
            masked_loss = loss * labels_mask  # (batch_size, num_sentences)
            loss = masked_loss.sum() / labels_mask.sum().clamp_min(1)
            output["loss"] = loss

        return output


