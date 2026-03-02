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
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--fresh-model", action="store_true", help="Train a completely new model from scratch")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific model checkpoint to load (e.g. codemind/checkpoints/model_best.pt)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    config = Config()
    config.set("training.num_train_epochs", args.epochs)
    config.set("training.per_device_train_batch_size", args.batch_size)
    config.set("training.learning_rate", args.lr)
    config.set("model.lora.r", args.lora_r)
    config.set("model.lora.lora_alpha", args.lora_alpha)
    
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager(config)
    
    if args.fresh_model:
        model_manager.load_fresh_model()
    else:
        try:
            model_name = args.checkpoint if args.checkpoint else "CodeMind-125M"
            logger.info(f"Loading CodeMind model: {model_name}...")
            model_manager.load_model(model_name)
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
            from src.core.prompting import build_instruction_prompt
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "user" in item and "assistant" in item:
                                texts.append(build_instruction_prompt(user=item["user"], assistant=item["assistant"]))
                            else:
                                texts.append(item.get("text", str(item)))
                        else:
                            texts.append(str(item))
                else:
                    texts = [str(data)]
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        downloader = DatasetDownloader()
        available = downloader.list_available_datasets()
        ds_list = []
        if args.dataset in available:
            logger.info(f"Downloading pre-defined dataset: {args.dataset}...")
            ds_list = downloader.download_dataset(args.dataset)
        else:
            logger.info(f"Attempting to download custom HuggingFace dataset: {args.dataset}...")
            try:
                ds_list = downloader.download_custom(args.dataset)
            except Exception as e:
                logger.error(f"Dataset '{args.dataset}' not found locally, in predefined keys, or on HuggingFace: {e}")
                logger.info(f"Available predefined datasets: {', '.join(available.keys())}")
                sys.exit(1)
                
        if ds_list and len(ds_list) > 0:
            from src.core.prompting import build_instruction_prompt
            texts = [
                build_instruction_prompt(user=item.get("user", ""), assistant=item.get("assistant", ""))
                for item in ds_list if isinstance(item, dict)
            ]
            
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
