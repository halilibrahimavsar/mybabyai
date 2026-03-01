import unittest
from pathlib import Path

from src.core.prompting import (
    CORRUPTED_PROMPT_LITERALS,
    LEGACY_PROMPT_MARKERS,
    TOKENS,
    build_chat_prompt,
    build_instruction_prompt,
    extract_assistant_response,
    validate_prompt_template,
)


class PromptContractTests(unittest.TestCase):
    def test_chat_prompt_uses_canonical_tokens(self):
        prompt = build_chat_prompt(
            user_input="Merhaba",
            system_prompt="Test system",
            history=[{"user": "u1", "assistant": "a1"}],
            context="ctx",
        )
        self.assertIn(TOKENS.assistant, prompt)
        self.assertNotIn("<|assistant|:", prompt)
        self.assertNotIn("<|assistant|/>", prompt)

    def test_instruction_prompt_extract(self):
        full = build_instruction_prompt(
            user="Write a function",
            assistant="def x():\n    return 1",
            include_eos=True,
        )
        extracted = extract_assistant_response(full)
        self.assertIn("def x()", extracted)

    def test_validate_rejects_legacy_marker(self):
        with self.assertRaises(ValueError):
            validate_prompt_template("<|user|>\nX\n<|assistant|/>\nY")

    def test_repository_has_no_corrupted_literals(self):
        root = Path(__file__).resolve().parents[1]
        skip_files = {
            root / "src" / "core" / "prompting" / "__init__.py",
            root / "scripts" / "lint_prompt_templates.py",
            root / "tests" / "test_prompt_contract.py",
        }
        banned = list(LEGACY_PROMPT_MARKERS) + list(CORRUPTED_PROMPT_LITERALS)

        for py_file in root.rglob("*.py"):
            if py_file in skip_files:
                continue
            py_path = str(py_file)
            if any(
                part in py_path
                for part in ("__pycache__", "venv/", ".git/", "checkpoints/")
            ):
                continue
            text = py_file.read_text(encoding="utf-8")
            for marker in banned:
                self.assertNotIn(marker, text, f"{py_file} contains {marker!r}")


if __name__ == "__main__":
    unittest.main()
