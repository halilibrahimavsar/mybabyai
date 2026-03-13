#!/usr/bin/env python3
"""
🚀 MyBabyAI - Turbo Training Script
Optimized for high-end hardware (90GB VRAM+, A100/H100/RTX4090).
"""

import os
import sys
import torch
import gc
from pathlib import Path

# --- PRE-IMPORT OPTIMIZATIONS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_turbo_env():
    print("🚀 Initializing Turbo Training Environment...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires a high-end GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU Detected: {gpu_name} | VRAM: {total_mem:.2f} GB")
    
    if total_mem < 20:
        print("⚠️ Warning: VRAM is below 20GB. Turbo settings might cause OOM.")
    
    # Enable TF32 for speedup on Ampere+ GPUs
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ TF32 Enabled")

def main():
    setup_turbo_env()
    
    # Reset path to current dir
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.config import Config
    from src.core.model_manager import ModelManager
    from src.core.trainer import LoRATrainer
    
    # Load Turbo Config
    turbo_config_path = project_root / "configs" / "turbo_config.yaml"
    if not turbo_config_path.exists():
        print(f"❌ Turbo config file not found at {turbo_config_path}")
        sys.exit(1)
        
    config = Config(config_path=str(turbo_config_path))
    print(f"📝 Loaded Turbo Configuration: {turbo_config_path.name}")
    
    # Initialize Model Manager
    model_manager = ModelManager(config)
    
    model_size = "650M"
    training_mode = "full" # Default to full training for maximum quality since we have 90GB
    
    print(f"\n🌱 Initializing CodeMind-{model_size} (Mode: {training_mode})...")
    model_manager.load_fresh_model(size=model_size)
    
    # Initialize Trainer
    trainer = LoRATrainer(model_manager, config)
    
    # Dataset Pool (Turkish Focused)
    # Using the same pool as in cloud_train.ipynb but with higher samples
    dataset_pool = [
        {
            "name": "🇹🇷 Turkish Instructions (Merve)",
            "type": "huggingface",
            "dataset_key": "turkish_instructions_merve",
            "max_samples": 5000
        },
        {
            "name": "🇹🇷 Turkish Alpaca",
            "type": "huggingface",
            "dataset_key": "turkish_alpaca",
            "max_samples": 5000
        },
        {
            "name": "🇹🇷 GPT-4 Alpaca TR",
            "type": "huggingface",
            "dataset_key": "alpaca_gpt4_tr",
            "max_samples": 5000
        },
        {
            "name": "🇹🇷 InstrucTurca",
            "type": "huggingface",
            "dataset_key": "instructurca",
            "lazy_load": True,
            "max_samples": 50000 
        }
    ]
    
    print(f"📊 Dataset pool size: {len(dataset_pool)} sources.")
    
    # VRAM Cleanup before start
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🚀 Starting Turbo Training...")
    try:
        trainer.train_from_pool(
            dataset_pool,
            training_type=training_mode,
            max_length=config.get("training.max_length", 4096),
            use_notebook_callback=False
        )
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
