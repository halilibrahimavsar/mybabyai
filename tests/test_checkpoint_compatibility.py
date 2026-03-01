import tempfile
import unittest
from pathlib import Path

import torch

from src.core.tokenizer.code_tokenizer import CodeTokenizer
from src.core.codemind_adapter import CodeMindAdapter
from src.utils.config import Config


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_incompatible_checkpoint_loads_partial_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            tokenizer_dir = checkpoint_dir / "tokenizer"
            tokenizer = CodeTokenizer(vocab_size=128)
            tokenizer.save(str(tokenizer_dir))

            checkpoint = {
                "model_state_dict": {
                    "bad.weight": torch.randn(2, 2),
                },
                "config": {
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 64,
                },
            }
            torch.save(checkpoint, checkpoint_dir / "model.pt")

            adapter = CodeMindAdapter(Config())
            adapter._checkpoint_dirs = [checkpoint_dir]
            adapter._compatibility_threshold = 0.80

            # It should load successfully, logging a warning, but NOT raising RuntimeError
            model, tok, report = adapter.load_model()
            
            self.assertFalse(report.is_compatible)
            self.assertEqual(report.compatibility_ratio, 0.0)

    def test_deprecated_keys_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            tokenizer_dir = checkpoint_dir / "tokenizer"
            tokenizer = CodeTokenizer(vocab_size=128)
            tokenizer.save(str(tokenizer_dir))

            checkpoint = {
                "model_state_dict": {
                    "model.position_embeddings.weight": torch.randn(2, 2),
                    "input_layernorm.bias": torch.randn(2),
                    "valid_key": torch.randn(2)
                },
                "config": {
                    "vocab_size": 128,
                },
            }
            torch.save(checkpoint, checkpoint_dir / "model.pt")

            adapter = CodeMindAdapter(Config())
            adapter._checkpoint_dirs = [checkpoint_dir]
            
            model, tok, report = adapter.load_model()
            
            self.assertEqual(report.deprecated_keys_found, 2)


if __name__ == "__main__":
    unittest.main()
