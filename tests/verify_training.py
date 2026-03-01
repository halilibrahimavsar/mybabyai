import os
import torch
from pathlib import Path
from src.core.trainer import LoRATrainer
from src.utils.config import Config

def test_training_codemind():
    print("\n--- Testing CodeMind Training ---")
    config = Config()
    # Force codemind name
    config.data["model"] = config.data.get("model", {})
    config.data["model"]["name"] = "CodeMind-125M"
    
    trainer = LoRATrainer(config=config)
    try:
        # Just check initialization and one step setup
        model, tokenizer = trainer.model_manager.load_model()
        print(f"CodeMind Model Loaded: {type(model)}")
        print("Initialization Success.")
    except Exception as e:
        print(f"CodeMind Training Init Failed: {e}")
        import traceback
        traceback.print_exc()

def test_training_hf():
    print("\n--- Testing Generic HF Training (GPT-2) ---")
    config = Config()
    config.data["model"] = config.data.get("model", {})
    config.data["model"]["name"] = "gpt2"
    
    trainer = LoRATrainer(config=config)
    try:
        model, tokenizer = trainer.model_manager.load_model()
        print(f"HF Model Loaded: {type(model)}")
        print("Initialization Success.")
    except Exception as e:
        print(f"HF Training Init Failed: {e}")

if __name__ == "__main__":
    test_training_codemind()
    test_training_hf()
