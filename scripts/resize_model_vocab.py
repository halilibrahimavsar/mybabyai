#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.core.model_manager import ModelManager
from src.core.checkpointing import extract_checkpoint_metadata, attach_checkpoint_metadata, build_checkpoint_metadata
import torch

def main():
    print("Pytorch, Model Manager yükleniyor...")
    config = Config()
    manager = ModelManager(config)
    
    checkpoint_dir = Path(config.get("model.codemind.checkpoint_dirs", ["codemind/checkpoints"])[0])
    checkpoint_path = checkpoint_dir / "model_final.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = Path("codemind/checkpoints_instruct/model_best.pt")
            
    print(f"Model yükleniyor: {checkpoint_path}")
    
    # Load model and tokenizer
    manager._load_codemind(str(checkpoint_path))
    
    model = manager.model
    tokenizer = manager.tokenizer
    
    old_vocab = model.config.vocab_size
    new_vocab = tokenizer.vocab_size_actual
    
    print(f"Mevcut Model Vocab: {old_vocab}")
    print(f"Yeni Tokenizer Vocab: {new_vocab}")
    
    if old_vocab == new_vocab:
        print("Boyutlar zaten eşleşiyor, işlem iptal ediliyor.")
        return
        
    print(f"Token embedding matrisi {old_vocab} -> {new_vocab} olarak yeniden boyutlandırılıyor...")
    
    # Ensure it's in a trainable state so embeddings can be modified
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    
    # Backup
    backup_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_backup_vocab{old_vocab}.pt")
    if not backup_path.exists():
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Eski model yedeği alındı: {backup_path}")
    else:
        print("Yedek zaten mevcut, üstüne yazılmıyor.")
    
    print("Yeni model kaydediliyor...")
    
    # Save manually mimicking save_checkpoint
    state_dict = model.state_dict()
    checkpoint = {"model_state_dict": state_dict}
    
    config_dict = getattr(model, "config", None)
    config_data = config_dict.to_dict() if config_dict else {}

    metadata = build_checkpoint_metadata(
        model_config=config_data,
        tokenizer=tokenizer,
        tokenizer_type="pretrained",
        architecture_version="codemind-v2"
    )
    checkpoint = attach_checkpoint_metadata(checkpoint, metadata)

    torch.save(checkpoint, checkpoint_path)
    
    print("İşlem başarıyla tamamlandı! Artık UI üzerinden Full Training başlatılabilir.")

if __name__ == "__main__":
    main()
