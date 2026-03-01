import unittest

from src.core.model_manager import ModelManager
from src.utils.config import Config


class ModelAliasTests(unittest.TestCase):
    def test_aliases_resolve_to_canonical_codemind(self):
        manager = ModelManager(Config())
        self.assertEqual(manager.resolve_model_name("CodeMind"), "CodeMind-125M")
        self.assertEqual(manager.resolve_model_name("codemind"), "CodeMind-125M")
        self.assertEqual(manager.resolve_model_name("CodeMind-125M"), "CodeMind-125M")
        self.assertEqual(
            manager.resolve_model_name("CodeMind-125M (Local)"), "CodeMind-125M"
        )


if __name__ == "__main__":
    unittest.main()
