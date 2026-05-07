from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import prompt_store


class PromptStoreTests(unittest.TestCase):
    def test_bootstrap_write_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rag_prompt.txt"
            created = prompt_store.ensure_prompt_file(prompt_file=path)
            self.assertEqual(created, path)
            self.assertTrue(path.exists())
            self.assertEqual(
                prompt_store.read_active_prompt(prompt_file=path),
                prompt_store.get_default_prompt_text(),
            )

            prompt_store.write_active_prompt("custom rag prompt", prompt_file=path)
            self.assertEqual(prompt_store.read_active_prompt(prompt_file=path), "custom rag prompt\n")

            reset_text = prompt_store.reset_active_prompt(prompt_file=path)
            self.assertEqual(reset_text, prompt_store.get_default_prompt_text())
            self.assertEqual(path.read_text(encoding="utf-8"), reset_text)

    def test_empty_prompt_write_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rag_prompt.txt"
            with self.assertRaises(ValueError):
                prompt_store.write_active_prompt("   ", prompt_file=path)


if __name__ == "__main__":
    unittest.main()
