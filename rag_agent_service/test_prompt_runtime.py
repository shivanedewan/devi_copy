from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prompt_store
from prompt import build_rag_system_prompt


class PromptRuntimeTests(unittest.TestCase):
    def test_build_rag_system_prompt_reads_active_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rag_prompt.txt"

            with patch("prompt_store.get_prompt_file_path", return_value=path):
                prompt_store.write_active_prompt("runtime rag prompt")
                self.assertEqual(build_rag_system_prompt(), "runtime rag prompt")


if __name__ == "__main__":
    unittest.main()
