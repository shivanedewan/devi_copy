from __future__ import annotations

import unittest

from retrieval_planner import RetrievalPlanner


class RetrievalPlannerDeterministicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RetrievalPlanner(
            client=None,
            model_name="test-model",
            known_branches=[],
            known_report_types=[],
            temperature=0.0,
            max_tokens=1024,
            max_query_variants=4,
            max_exact_terms=4,
            max_turns=6,
            max_context_chars=2600,
            default_time_field="ingestion_date",
        )

    def test_format_followup_reuses_previous_topic(self) -> None:
        chat_context = {
            "recent_conversations": [
                {
                    "user_message": "tell me about donald trump<|endoftext|>",
                    "assistant_reply": "Donald Trump served as President of the United States.<|endoftext|>",
                }
            ]
        }

        payload = self.planner._build_deterministic_fallback_payload(
            query="respond in 2 page report",
            chat_context=chat_context,
            allow_history_expansion=True,
            expanded_history_used=False,
        )

        self.assertTrue(payload["context_dependent"])
        self.assertEqual(payload["answer_intent"], "format")
        self.assertEqual(payload["retrieval_action"], "reuse_previous_topic")
        self.assertEqual(payload["standalone_query"], "donald trump")
        self.assertIn("donald trump", payload["query_variants"])
        self.assertIn("donald trump", payload["exact_terms"])

    def test_reference_followup_ignores_prior_formatting_turn(self) -> None:
        chat_context = {
            "recent_conversations": [
                {
                    "user_message": "tell me about donald trump<|endoftext|>",
                    "assistant_reply": "Donald Trump served as President of the United States.<|endoftext|>",
                },
                {
                    "user_message": "respond in 2 page report<|endoftext|>",
                    "assistant_reply": "Please clarify the topic.<|endoftext|>",
                },
            ]
        }

        payload = self.planner._build_deterministic_fallback_payload(
            query="consider my first question and then respond",
            chat_context=chat_context,
            allow_history_expansion=True,
            expanded_history_used=False,
        )

        self.assertTrue(payload["context_dependent"])
        self.assertEqual(payload["retrieval_action"], "reuse_previous_topic")
        self.assertEqual(payload["standalone_query"], "donald trump")
        self.assertIn("donald trump", payload["query_variants"])


if __name__ == "__main__":
    unittest.main()
