import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from method.reassessment import (
    _decision_guard_check,
    _parse_decision_guard_analysis,
    decision_consistency_guard,
    main,
)


def analysis(central_status, components):
    return {"central_claim_status": central_status, "components": components}


def component(name, importance, statuses, supported=(), contradicted=(), insufficient=()):
    return {
        "component": name,
        "importance": importance,
        "status": list(statuses),
        "evidence_ids": {
            "SUPPORTED": list(supported),
            "CONTRADICTED": list(contradicted),
            "INSUFFICIENT": list(insufficient),
        },
    }


class DecisionConsistencyGuardTests(unittest.TestCase):
    def test_false_with_supported_central_claim_reconsiders(self):
        state = analysis("SUPPORTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"])
        ])
        self.assertEqual(
            _decision_guard_check("false", state)[0],
            "RECONSIDER",
        )

    def test_insufficient_central_evidence_is_not_contradiction(self):
        state = analysis("INSUFFICIENT", [
            component("central", "CENTRAL", ["INSUFFICIENT"])
        ])
        action, reason = _decision_guard_check("false", state)
        self.assertEqual(action, "RECONSIDER")
        self.assertIn("insufficient evidence alone", reason)

    def test_supported_and_contradicted_material_components_keep_half_true(self):
        state = analysis("SUPPORTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"]),
            component(
                "material context",
                "MATERIAL_SECONDARY",
                ["CONTRADICTED"],
                contradicted=["Evidence_2"],
            ),
        ])
        self.assertEqual(_decision_guard_check("half-true", state)[0], "KEEP")

    def test_true_with_contradicted_material_component_reconsiders(self):
        state = analysis("SUPPORTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"]),
            component(
                "material detail",
                "MATERIAL_SECONDARY",
                ["CONTRADICTED"],
                contradicted=["Evidence_2"],
            ),
        ])
        self.assertEqual(_decision_guard_check("true", state)[0], "RECONSIDER")

    def test_half_true_with_only_insufficient_material_evidence_escalates(self):
        state = analysis("SUPPORTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"]),
            component("material detail", "MATERIAL_SECONDARY", ["INSUFFICIENT"]),
        ])
        self.assertEqual(_decision_guard_check("half-true", state)[0], "ESCALATE")

    def test_false_with_conflicting_central_status_escalates(self):
        state = analysis("MIXED", [
            component(
                "central",
                "CENTRAL",
                ["SUPPORTED", "CONTRADICTED"],
                supported=["Evidence_1"],
                contradicted=["Evidence_2"],
            )
        ])
        self.assertEqual(_decision_guard_check("false", state)[0], "ESCALATE")

    def test_analysis_rejects_unsupplied_evidence_ids(self):
        response = json.dumps(analysis("SUPPORTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_9"])
        ]))
        with self.assertRaisesRegex(ValueError, "ID not supplied"):
            _parse_decision_guard_analysis(response, {"Evidence_1": "text"})

    def test_analysis_preserves_conflicting_evidence_statuses(self):
        response = json.dumps(analysis("MIXED", [
            component(
                "central",
                "CENTRAL",
                ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
                supported=["Evidence_1"],
                contradicted=["Evidence_2"],
            )
        ]))
        parsed = _parse_decision_guard_analysis(
            response,
            {"Evidence_1": "supports", "Evidence_2": "contradicts"},
        )
        self.assertEqual(parsed["central_claim_status"], "MIXED")
        self.assertEqual(
            parsed["components"][0]["status"],
            ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
        )
        self.assertEqual(
            parsed["components"][0]["evidence_ids"]["CONTRADICTED"],
            ["Evidence_2"],
        )

    def test_central_status_is_derived_from_central_components(self):
        response = json.dumps(analysis("CONTRADICTED", [
            component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"])
        ]))
        parsed = _parse_decision_guard_analysis(
            response, {"Evidence_1": "supports"}
        )
        self.assertEqual(parsed["central_claim_status"], "SUPPORTED")

    def test_pipe_delimited_analysis_preserves_conflicting_evidence(self):
        response = (
            "CENTRAL | central claim | SUPPORTED=Evidence_1 | "
            "CONTRADICTED=Evidence_2 | INSUFFICIENT=none\n"
            "MATERIAL_SECONDARY | detail | SUPPORTED=none | "
            "CONTRADICTED=none | INSUFFICIENT=no evidence IDs"
        )
        parsed = _parse_decision_guard_analysis(
            response,
            {"Evidence_1": "supports", "Evidence_2": "contradicts"},
        )
        self.assertEqual(parsed["central_claim_status"], "MIXED")
        self.assertEqual(
            parsed["components"][0]["status"],
            ["SUPPORTED", "CONTRADICTED"],
        )
        self.assertEqual(
            parsed["components"][1]["status"],
            ["INSUFFICIENT"],
        )

    @patch("method.reassessment.call_gpt")
    def test_malformed_analysis_escalates_and_retains_prediction(self, call_gpt):
        call_gpt.return_value = "not json"
        prediction, details = decision_consistency_guard(
            "false", "rationale", "claim", {"Evidence_1": "text"},
            return_details=True,
        )
        self.assertEqual(prediction, "false")
        self.assertEqual(details["guard_action"], "ESCALATE")
        self.assertFalse(details["analysis_parse_valid"])
        call_gpt.assert_called_once()

    @patch("method.reassessment.call_gpt")
    def test_malformed_reconsideration_escalates_and_retains_prediction(self, call_gpt):
        call_gpt.side_effect = [
            json.dumps(analysis("SUPPORTED", [
                component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"])
            ])),
            "A different answer without the required label line",
        ]
        prediction, details = decision_consistency_guard(
            "false", "rationale", "claim", {"Evidence_1": "text"},
            return_details=True,
        )
        self.assertEqual(prediction, "false")
        self.assertEqual(details["guard_action"], "ESCALATE")
        self.assertIsNone(details["consistency_check"])
        self.assertIn("malformed", details["consistency_reason"])
        self.assertEqual(call_gpt.call_count, 2)

    @patch("method.reassessment.call_gpt")
    def test_reconsideration_uses_valid_model_label(self, call_gpt):
        call_gpt.side_effect = [
            json.dumps(analysis("SUPPORTED", [
                component("central", "CENTRAL", ["SUPPORTED"], supported=["Evidence_1"])
            ])),
            "Label: HALF-TRUE\nExplanation: The evidence is mixed.",
        ]
        prediction, details = decision_consistency_guard(
            "false", "rationale", "claim", {"Evidence_1": "text"},
            return_details=True,
        )
        self.assertEqual(prediction, "half-true")
        self.assertEqual(details["guard_action"], "RECONSIDER")
        self.assertTrue(details["consistency_check"] is False)

    def test_guard_cannot_be_combined_with_extension(self):
        args = SimpleNamespace(
            decision_consistency_guard=True,
            classifier_prompt="evidence-consistency",
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            main(args)


if __name__ == "__main__":
    unittest.main()
