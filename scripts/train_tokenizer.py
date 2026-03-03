#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.core.tokenizer.code_tokenizer import CodeTokenizer

def create_training_corpus():
    print("Veri toplanıyor...")
    corpus_files = []
    temp_dir = Path("temp_tokenizer_data")
    temp_dir.mkdir(exist_ok=True)
    
    # 1. Gather Python/Dart Code from local project to preserve code syntax
    code_text = ""
    for ext in ["**/*.py", "**/*.dart"]:
        for file_path in glob.glob(f"src/{ext}", recursive=True):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_text += f.read() + "\n\n"
            except:
                pass
    
    code_file = temp_dir / "local_code.txt"
    with open(code_file, "w", encoding="utf-8") as f:
        f.write(code_text)
    corpus_files.append(str(code_file))
    print(f"Local code corpus oluşturuldu: {len(code_text)} karakter")

    # 2. Gather Turkish Text from Turkish-Alpaca
    print("Turkish-Alpaca veri seti indiriliyor (Tokenizer için)...")
    try:
        ds = load_dataset("TFLai/Turkish-Alpaca", split="train")
        turkish_text = ""
        # Let's take a good sample: 15,000 instructions
        sample_size = min(len(ds), 15000)
        for i in range(sample_size):
            item = ds[i]
            turkish_text += f"{item.get('instruction', '')}\n{item.get('input', '')}\n{item.get('output', '')}\n\n"
            
        tr_file = temp_dir / "turkish_corpus.txt"
        with open(tr_file, "w", encoding="utf-8") as f:
            f.write(turkish_text)
        corpus_files.append(str(tr_file))
        print(f"Türkçe corpus oluşturuldu: {len(turkish_text)} karakter ({sample_size} satır)")
    except Exception as e:
        print(f"Dataset indirme hatası, lokal corpus ile devam edilecek: {e}")

    return corpus_files, temp_dir

def main():
    target_vocab_size = 16384
    checkpoint_dir = Path("codemind/checkpoints/tokenizer")
    
    print(f"Token hazinesi {target_vocab_size} kelime kapasitesine genişletiliyor...")
    
    corpus_files, temp_dir = create_training_corpus()
    
    try:
        # Create fresh tokenizer instance but with larger capacity
        tokenizer = CodeTokenizer(vocab_size=target_vocab_size)
        
        print("BPE Tokenizer eğitimi başlatılıyor (Bu işlem 1-2 dakika sürebilir)...")
        tokenizer.train(files=corpus_files)
        
        print("Eğitim tamamlandı. Kaydediliyor...")
        tokenizer.save(str(checkpoint_dir))
        
        print(f"Başarılı! Yeni Tokenizer ({tokenizer.vocab_size_actual} kelime) şu konuma kaydedildi: {checkpoint_dir}")
        
    finally:
        # Cleanup temp files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Geçici dosyalar temizlendi.")

if __name__ == "__main__":
    main()
