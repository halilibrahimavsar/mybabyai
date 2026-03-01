import os
import json
from pathlib import Path
import sys

# Add project root to path


from src.data.dataset_downloader import DatasetDownloader
from src.utils.logger import get_logger

def setup_data():
    logger = get_logger("setup_codemind_data")
    downloader = DatasetDownloader()
    
    # We want a good mix of code instructions
    # CodeAlpaca-20k is a great starting point for multi-language and Python
    target_file = "code_training_data.json"
    
    logger.info("CodeAlpaca-20k veriseti indiriliyor...")
    try:
        # Download 5000 samples for a solid start (can be increased later)
        # 5000 is enough for 125M model to start showing patterns without taking days to train
        data_path = downloader.download_and_save(
            "code_alpaca", 
            output_filename=target_file,
            max_samples=5000
        )
        
        # Also let's append some Turkish instructions if available to keep the Turkish capability
        logger.info("Türkçe Alpaca veriseti indiriliyor...")
        tr_data = downloader.download_dataset("turkish_instructions", max_samples=500)
        
        # Load the existing file and merge
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        
        all_data.extend(tr_data)
        
        # Overwrite with merged data
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Başarılı! Toplam {len(all_data)} örnek '{data_path}' konumuna kaydedildi.")
        
    except Exception as e:
        logger.error(f"Veri kurulumu sırasında hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_data()
