import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.model_manager import ModelManager
from src.core.trainer import LoRATrainer
from src.utils.config import Config

def test_hf_integration():
    print("HF Integration Testi Başlatılıyor...")
    config = Config()
    model_manager = ModelManager(config)
    
    print("Model yükleniyor (fallback-enabled)...")
    # Use codemind with fallback to ensure it works even if no checkpoint exists
    model_manager.load_model("codemind", allow_fresh_fallback=True)
    
    trainer = LoRATrainer(model_manager, config)
    
    print("HF Dataset (tiny_shakespeare) ile eğitim deneniyor (max 2 samples)...")
    try:
        # Using a very small sample size for quick verification
        metrics = trainer.train_from_hf(
            "tiny_shakespeare", 
            max_samples=2,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1
        )
        print(f"Eğitim Başarılı! Metrikler: {metrics}")
        return True
    except Exception as e:
        print(f"Eğitim HATASI: {e}")
        return False

if __name__ == "__main__":
    success = test_hf_integration()
    sys.exit(0 if success else 1)
