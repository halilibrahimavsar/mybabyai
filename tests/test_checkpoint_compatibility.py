import tempfile
import unittest
from pathlib import Path

import torch

from src.core.tokenizer.code_tokenizer import CodeTokenizer
from src.core.codemind_adapter import CodeMindAdapter
from src.utils.config import Config


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_incompatible_checkpoint_fails_fast(self):
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

            with self.assertRaises(RuntimeError) as ctx:
                adapter.load_model()

            self.assertIn("compatibility below threshold", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
