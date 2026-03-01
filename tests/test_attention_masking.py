import unittest

import torch

from src.core.model.codemind import CodeMindModel


class AttentionMaskingTests(unittest.TestCase):
    def test_causal_mask_blocks_future_tokens(self):
        mask = CodeMindModel._build_combined_attention_mask(
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            batch_size=1,
            seq_len=4,
            past_length=0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.shape, (1, 1, 4, 4))
        self.assertLess(mask[0, 0, 0, 1].item(), -1e20)
        self.assertEqual(mask[0, 0, 3, 0].item(), 0.0)

    def test_padding_mask_blocks_padded_positions(self):
        mask = CodeMindModel._build_combined_attention_mask(
            attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            batch_size=1,
            seq_len=4,
            past_length=0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertLess(mask[0, 0, 3, 3].item(), -1e20)

    def test_past_length_respected(self):
        mask = CodeMindModel._build_combined_attention_mask(
            attention_mask=torch.ones((1, 6), dtype=torch.long),
            batch_size=1,
            seq_len=2,
            past_length=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(mask.shape, (1, 1, 2, 6))
        self.assertLess(mask[0, 0, 0, 5].item(), -1e20)
        self.assertEqual(mask[0, 0, 1, 5].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
