import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import torch
from src.core.inference import InferenceEngine
from src.core.model_manager import ModelManager
from src.core.cognitive.modes import CognitiveMode
from src.utils.config import Config

class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.config = None
    
    def generate(self, *args, **kwargs):
        # This is never actually called because we use _lm_generate_multiple wrapper in InferenceEngine
        return torch.tensor([[1, 2, 3]])

    def train(self):
        pass
        
    def parameters(self):
        return []

    def named_modules(self):
        return []

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
    def decode(self, ids, **kwargs):
        return "Step: Bu bir test düşüncesidir."
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
    def convert_tokens_to_ids(self, token):
        return 0

def test_flow():
    print("=== Nihai Akıl: Bilişsel Akış Testi ===")
    
    # Setup mock manager
    config = Config()
    model_manager = ModelManager(config)
    model_manager.model = MockModel()
    model_manager.tokenizer = MockTokenizer()
    model_manager.device = "cpu"
    model_manager.is_codemind = False # Use standard flow for this test
    
    engine = InferenceEngine(model_manager, config=config)
    
    # 1. Test System 1 (Simple prompt)
    print("\n[Test 1] Basit Soru (System 1 beklenen):")
    prompt_fast = "Merhaba, nasılsın?"
    response_fast = engine.generate(prompt_fast, use_memory=False)
    print(f"Girdi: {prompt_fast}")
    print(f"Yanıt: {response_fast}")

    # 2. Test System 2 (Reasoning prompt)
    print("\n[Test 2] Karmaşık Soru (System 2 beklenen):")
    prompt_slow = "Bir web sunucusu tasarlamak için adım adım bir plan oluştur."
    
    # We need to monkeypatch _fast_generate to avoid actual model call errors if any
    engine._fast_generate = lambda p, **kw: "Step: Mantıklı bir düşünce adımı."
    
    response_slow = engine.generate(prompt_slow, use_memory=False)
    print(f"Girdi: {prompt_slow}")
    print(f"Yanıt (MCTS Sonucu):\n{response_slow}")

    print("\n=== Test Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    test_flow()
