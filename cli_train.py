import argparse
import sys
import logging
from pathlib import Path
from typing import List

from src.utils.config import Config
from src.core.model_manager import ModelManager
from src.core.trainer import LoRATrainer
from src.data.dataset_downloader import DatasetDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cli_train")

def main():
    parser = argparse.ArgumentParser(description="MyBabyAI Headless CLI Training")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset key (e.g., tiny_shakespeare) or path to a local text/json file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--fresh-model", action="store_true", help="Train a completely new model from scratch")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    config = Config()
    config.set("training.num_epochs", args.epochs)
    config.set("training.batch_size", args.batch_size)
    config.set("training.learning_rate", args.lr)
    config.set("model.lora.r", args.lora_r)
    config.set("model.lora.lora_alpha", args.lora_alpha)
    
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager(config)
    
    if args.fresh_model:
        model_manager.load_fresh_model()
    else:
        try:
            model_manager.load_model("CodeMind-125M")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Hint: pass --fresh-model to initialize a new model from scratch.")
            sys.exit(1)
            
    logger.info("Preparing dataset...")
    dataset_path = Path(args.dataset)
    texts: List[str] = []
    
    if dataset_path.exists() and dataset_path.is_file():
        logger.info(f"Loading local file: {dataset_path}")
        if dataset_path.suffix == ".json":
            import json
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in data]
                else:
                    texts = [str(data)]
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        downloader = DatasetDownloader()
        available = downloader.list_available_datasets()
        if args.dataset in available:
            logger.info(f"Downloading pre-defined dataset: {args.dataset}...")
            ds_dict = downloader.download_dataset(args.dataset)
            if ds_dict and len(ds_dict) > 0:
                # Use the first split directly
                first_split = list(ds_dict.values())[0]
                texts = [item.get("text", "") for item in first_split]
        else:
            logger.error(f"Dataset '{args.dataset}' not found locally or in predefined keys.")
            logger.info(f"Available predefined datasets: {', '.join(available.keys())}")
            sys.exit(1)
            
    # Filter empty texts
    texts = [t for t in texts if t]
            
    if not texts:
        logger.error("No valid training text found in dataset.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(texts)} training items.")
    
    logger.info("Initializing Trainer...")
    trainer = LoRATrainer(model_manager, config)
    
    logger.info(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    try:
        metrics = trainer.train_from_texts(texts)
        logger.info(f"Training completed successfully!")
        logger.info(f"Metrics: {metrics}")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Stopping and attempting to save...")
        trainer.stop_training()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
