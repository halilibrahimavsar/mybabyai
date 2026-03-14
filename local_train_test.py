"""
CodeMind — Yerel Eğitim Testi (10M / 20M / 125M boyut desteği)

Kullanım:
    python local_train_test.py             # varsayılan: 10M
    python local_train_test.py --size 20M
    python local_train_test.py --size 125M

Strateji:
  • Model   : CodeMind-{size} (sıfırdan / eğitilmemiş)
  • Tip      : LoRA (sadece birkaç M parametre eğitilir)
  • Batch    : 1 + gradient_accumulation=8 → efektif batch=8
  • Seq len  : 64 token (10M/20M) / 128 token (125M) — VRAM tasarrufu
  • Veri     : 200 örnek dummy conversation (HF indirme yok)
  • Epoch    : 1 (hızlı sanity-check)
  • Hedef    : Hata olmadan tamamlanırsa local testi geçti

Başarı kriterleri:
  ✅ Training loss azalıyor (overfit = öğreniyor demek)
  ✅ OOM hatası yok

Sonraki adım (başarılı olursa):
  → cloud_train_turbo.ipynb ile Kaggle / Colab'da daha büyük model tam eğitim
"""

import argparse
import sys
import os
import gc
import torch

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── VRAM koruma: PyTorch fragmantasyon çözücüsü ──────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.core.trainer import LoRATrainer

logger = get_logger("local_train_test")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Konfigürasyon — GTX 1650 / düşük VRAM için konservatif ayarlar
# ─────────────────────────────────────────────────────────────────────────────
def build_config(size: str) -> Config:
    cfg = Config()

    # Model boyutuna göre seq length belirle:
    # 10M/20M → 64 token (çok küçük model, kısa seq yeterli)
    # 125M → 128 token (biraz daha uzun context)
    if size in ("10M", "20M"):
        max_length = 64
    else:
        max_length = 128

    cfg.set("model.name", f"CodeMind-{size}")
    cfg.set("model.device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.set("model.load_in_4bit", False)   # GTX 1650'de bitsandbytes desteği kırık

    # ConversationDataset tokenizer() __call__ interface kullanır.
    # CodeTokenizer sadece .encode() destekler → OOB embedding crash.
    # Küçük modeller için pre-trained Türkçe tokenizer kullan.
    cfg.set("model.pretrained_tokenizer", "ytu-ce-cosmos/Turkish-GPT2-large")

    # LoRA — sadece q_proj + v_proj (minimum VRAM)
    cfg.set("model.lora.r", 8)
    cfg.set("model.lora.lora_alpha", 16)
    cfg.set("model.lora.lora_dropout", 0.05)
    cfg.set("model.lora.target_modules", ["q_proj", "v_proj"])

    # Eğitim
    cfg.set("training.output_dir", f"models/local_test_{size.lower()}")
    cfg.set("training.per_device_train_batch_size", 1)
    cfg.set("training.gradient_accumulation_steps", 8)
    cfg.set("training.gradient_checkpointing", True)
    cfg.set("training.max_length", max_length)
    cfg.set("training.num_train_epochs", 1)
    cfg.set("training.learning_rate", 3e-4)
    cfg.set("training.warmup_steps", 20)
    cfg.set("training.logging_steps", 5)
    cfg.set("training.save_steps", 9999)   # sadece sonda kaydet
    cfg.set("training.optim", "adamw_torch")  # paged_adamw bitsandbytes gerektirir
    cfg.set("training.pack_sequences", False)
    cfg.set("training.torch_compile", False)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dummy veri — HF indirme yok, bağlantı gerektirmez
# ─────────────────────────────────────────────────────────────────────────────
def generate_dummy_conversations(n: int = 200):
    """Gerçekçi biçimde küçük ama çeşitli conversation örnekleri."""
    templates = [
        (
            "Python'da for döngüsü nasıl kullanılır?",
            "Python'da for döngüsü şu şekilde kullanılır:\n\nfor i in range(10):\n    print(i)\n\nBu döngü 0'dan 9'a kadar sayıları yazdırır.",
        ),
        (
            "Merhaba, nasılsın?",
            "Merhaba! İyiyim, teşekkür ederim. Sana nasıl yardımcı olabilirim?",
        ),
        (
            "Flutter'da StatefulWidget ne zaman kullanılır?",
            "StatefulWidget, zaman içinde değişen (dinamik) veriler içerdiğinde kullanılır. Örneğin bir sayaç, form alanı veya animasyon için uygundur.",
        ),
        (
            "Türkiye'nin başkenti neresidir?",
            "Türkiye'nin başkenti Ankara'dır.",
        ),
        (
            "SQL'de JOIN nedir?",
            "SQL'de JOIN, iki veya daha fazla tabloyu ortak bir sütun üzerinden birleştirmek için kullanılır. En yaygın türler: INNER JOIN, LEFT JOIN, RIGHT JOIN ve FULL JOIN'dir.",
        ),
        (
            "Dart'ta null safety nasıl çalışır?",
            "Dart'ta null safety, değişkenlerin varsayılan olarak null olamayacağını garanti eder. Null olabilecek değişkenler için `String?` gibi `?` operatörü kullanılır.",
        ),
        (
            "Makine öğrenmesi nedir?",
            "Makine öğrenmesi, bilgisayarların verilerden otomatik olarak öğrenmesini sağlayan yapay zeka alt dalıdır. Açıkça programlanmadan deneyimle gelişir.",
        ),
        (
            "Git commit nasıl yapılır?",
            "Git commit şu şekilde yapılır:\n\ngit add .\ngit commit -m 'Commit mesajı'\n\nBu komutlar tüm değişiklikleri hazırlar ve kaydeder.",
        ),
    ]

    conversations = []
    for i in range(n):
        template = templates[i % len(templates)]
        conversations.append({"user": template[0], "assistant": template[1]})
    return conversations


# ─────────────────────────────────────────────────────────────────────────────
# 3. VRAM ön raporu
# ─────────────────────────────────────────────────────────────────────────────
def print_vram_status(stage: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0)  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(
            f"[{stage}] VRAM: {allocated:.2f}/{total:.2f} GB kullanımda "
            f"({reserved:.2f} GB rezerve)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ana test akışı
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CodeMind Yerel Eğitim Testi")
    parser.add_argument(
        "--size",
        default="10M",
        choices=["10M", "20M", "125M"],
        help="Test edilecek model boyutu (default: 10M)",
    )
    args = parser.parse_args()
    size = args.size

    logger.info("=" * 60)
    logger.info(f"  CodeMind-{size} Yerel Eğitim Testi")
    logger.info("=" * 60)

    # GPU varlık kontrolü
    if not torch.cuda.is_available():
        logger.warning("CUDA bulunamadı! CPU'da çalışacak — çok yavaş olabilir.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")

    # ── Adım 1: Config ──────────────────────────────────────────────────────
    logger.info("\n[1/5] Konfigürasyon yükleniyor...")
    config = build_config(size)
    logger.info("✅ Config hazır")

    # ── Adım 2: Model yükleme (sıfırdan=eğitilmemiş) ───────────────────────
    logger.info(f"\n[2/5] CodeMind-{size} sıfır model oluşturuluyor...")
    model_manager = ModelManager(config)
    model_manager.load_fresh_model(size=size)

    total_params = sum(p.numel() for p in model_manager.model.parameters())
    logger.info(f"✅ Model hazır — toplam parametre: {total_params:,}")
    print_vram_status("Model Yüklendi")

    # ── Adım 3: Veri hazırlama ───────────────────────────────────────────────
    logger.info("\n[3/5] Dummy veri hazırlanıyor (200 örnek)...")
    conversations = generate_dummy_conversations(200)
    logger.info(f"✅ {len(conversations)} konuşma örneği hazır")

    # ── Adım 4: Eğitim ──────────────────────────────────────────────────────
    logger.info("\n[4/5] LoRA eğitimi başlatılıyor...")
    max_length = config.get("training.max_length", 64)
    logger.info(f"  • Model: CodeMind-{size} | Params: {total_params:,}")
    logger.info(f"  • Batch: 1 (grad_acc=8 → efektif 8)")
    logger.info(f"  • Seq length: {max_length} token")
    logger.info("  • LoRA rank: 8")
    logger.info("  • Gradient checkpointing: ETKİN")
    logger.info("  • Epoch: 1")

    trainer = LoRATrainer(model_manager=model_manager, config=config)

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_vram_status("Eğitim Başlamadan Önce")

        metrics = trainer.train_from_conversations(
            conversations=conversations,
            training_type="lora",
        )

        print_vram_status("Eğitim Tamamlandı")

        # ── Adım 5: Sonuç raporu ─────────────────────────────────────────────
        logger.info("\n[5/5] Test Sonuçları:")
        logger.info("=" * 60)
        logger.info(f"✅ YEREL EĞİTİM TESTİ BAŞARILI! (CodeMind-{size})")
        logger.info(f"  Loss         : {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  Adım/sn      : {metrics.get('train_steps_per_second', 'N/A')}")
        logger.info(f"  Süre (sn)    : {metrics.get('train_runtime', 'N/A'):.1f}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Bir sonraki adım:")
        logger.info(f"  → cloud_train_turbo.ipynb ile Kaggle/Colab'da daha büyük model eğit")
        logger.info(f"  → model_size seçeneğini '{size}' yerine '350M' veya '650M' yap")

    except torch.cuda.OutOfMemoryError:
        logger.error("\n❌ CUDA Bellek Hatası (OOM)!")
        logger.error("Çözüm önerileri:")
        if size in ("10M", "20M"):
            logger.error(f"  1. training.max_length: {max_length//2}'e düşür")
            logger.error("  2. Başka uygulamaları kapat, GPU boşalt")
        else:
            logger.error("  1. --size 10M veya --size 20M ile küçük boyutta dene")
            logger.error(f"  2. training.max_length: {max_length//2}'e düşür")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
