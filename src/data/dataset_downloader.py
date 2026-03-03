import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datasets import load_dataset
import requests
from tqdm import tqdm
import shutil
import zipfile
import tarfile

import sys


from src.utils.logger import get_logger


class DatasetDownloader:
    READY_DATASETS = {
        # --- ENGLISH (INSTRUCTION & CHAT) ---
        "slim_orca": {
            "name": "Open-Orca/SlimOrca",
            "description": "GPT-4 tabanlı devasa talimat seti (EN)",
            "size": "Büyük (~1GB JSONL)",
            "languages": ["İngilizce"],
        },
        "ultrachat_200k": {
            "name": "HuggingFaceH4/ultrachat_200k",
            "description": "Yüksek kaliteli sohbet verisi (EN)",
            "size": "Büyük (~1GB)",
            "languages": ["İngilizce"],
        },
        "dolly_15k": {
            "name": "databricks/databricks-dolly-15k",
            "description": "İnsan yapımı temiz talimatlar (EN)",
            "size": "Orta (~50MB)",
            "languages": ["İngilizce"],
        },
        "alpaca_en": {
            "name": "tatsu-lab/alpaca",
            "description": "Klasik Alpaca talimat seti (EN)",
            "size": "Orta (~40MB)",
            "languages": ["İngilizce"],
        },
        "wizardlm_70k": {
            "name": "WizardLM/WizardLM_evol_instruct_V2_196k",
            "description": "Karmaşık mantık ve kod talimatları (EN)",
            "size": "Büyük (~200MB)",
            "languages": ["İngilizce"],
        },
        
        # --- TURKISH (INSTRUCTION & NEWS) ---
        "turkish_instructions_merve": {
            "name": "merve/turkish_instructions",
            "description": "Geniş kapsamlı Türkçe talimatlar",
            "size": "Orta (~30MB)",
            "languages": ["Türkçe"],
        },
        "turkish_alpaca": {
            "name": "TFLai/Turkish-Alpaca",
            "description": "Türkçe Alpaca çeviri seti",
            "size": "Küçük (~15MB)",
            "languages": ["Türkçe"],
        },
        "turkish_news_70k": {
            "name": "savasy/turkish-news-7x10k",
            "description": "Resmi dil için Türkçe haberler",
            "size": "Orta (~60MB)",
            "languages": ["Türkçe"],
        },
        "tr_instruction_dataset": {
            "name": "m3hrdadfi/tr-instruction-dataset",
            "description": "Çeşitli Türkçe görev talimatları",
            "size": "Orta (~20MB)",
            "languages": ["Türkçe"],
        },

        # --- MULTILINGUAL & KNOWLEDGE ---
        "oasst1_multilingual": {
            "name": "OpenAssistant/oasst1",
            "description": "Gerçek insan konuşmaları (TR/EN/DE...)",
            "size": "Büyük (~500MB)",
            "languages": ["Çok dilli"],
        },
        "wikineural_multilingual": {
            "name": "Babelscape/wikineural",
            "description": "Wikipedia tabanlı temiz metin (9 Dil)",
            "size": "Büyük (~2GB)",
            "languages": ["Çok dilli"],
        },
        "culturax_tr_sample": {
            "name": "uonlp/CulturaX",
            "config": "tr",
            "description": "Devasa Türkçe web külliyatı (Örneklem)",
            "size": "Çok Büyük",
            "languages": ["Türkçe"],
        },

        # --- CODE ---
        "code_alpaca_20k": {
            "name": "sahil2801/CodeAlpaca-20k",
            "description": "Kod yazma ve hata ayıklama (EN)",
            "size": "Orta (~25MB)",
            "languages": ["Kodlama", "İngilizce"],
        },
        "python_instructions": {
            "name": "iamtarun/python_code_instructions_18k_alpaca",
            "description": "Python özelinde talimatlar",
            "size": "Orta (~50MB)",
            "languages": ["Python"],
        },
        "flutter_dart_stack": {
            "name": "bigcode/the-stack",
            "config": "data/dart",
            "description": "Dart/Flutter kod bankası",
            "size": "Çok Büyük",
            "languages": ["Dart", "Flutter"],
        },

        # --- SMALL & TEST ---
        "tiny_shakespeare": {
            "name": "karpathy/tiny_shakespeare",
            "description": "Shakespeare metinleri (Hızlı test)",
            "size": "Çok Küçük",
            "languages": ["İngilizce"],
        },
        "gsm8k_math": {
            "name": "openai/gsm8k",
            "config": "main",
            "description": "Matematiksel akıl yürütme",
            "size": "Küçük",
            "languages": ["İngilizce"],
        },
        "en_tr_parallel": {
            "name": "Helsinki-NLP/opus-100",
            "config": "en-tr",
            "description": "EN-TR Çeviri ve Hizalama",
            "size": "Orta (~250MB)",
            "languages": ["İngilizce", "Türkçe"],
        },
        
        # --- ARCHIVES & DIRECT URLS ---
        "sample_archive_zip": {
            "name": "https://github.com/karpathy/char-rnn/archive/refs/heads/master.zip",
            "description": "Örnek GitHub Arşivi (Zip)",
            "size": "Çok Küçük",
            "languages": ["Kodlama"],
        },
        "turkish_sample_archive": {
            "name": "https://raw.githubusercontent.com/TSK-AI/datasets/main/sample_tr.tar.gz",
            "description": "Örnek Tar.GZ Arşivi (TR)",
            "size": "Küçük",
            "languages": ["Türkçe"],
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = get_logger("dataset_downloader")
        self.cache_dir = cache_dir or str(
            Path(__file__).parent.parent.parent / "data" / "datasets"
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def list_available_datasets(self) -> Dict[str, Dict]:
        return self.READY_DATASETS

    def download_dataset(
        self, dataset_key: str, max_samples: Optional[int] = None, split: str = "train"
    ) -> List[Dict[str, str]]:
        if dataset_key not in self.READY_DATASETS:
            raise ValueError(f"Bilinmeyen dataset: {dataset_key}")

        dataset_info = self.READY_DATASETS[dataset_key]
        dataset_name = dataset_info["name"]
        config = dataset_info.get("config")

        self.logger.info(f"Dataset indiriliyor: {dataset_name}")

        if dataset_name.startswith(("http://", "https://")):
            # Direct URL or Archive
            path = self.download_url_and_extract(dataset_name)
            
            # Load the data from the extracted path using DatasetLoader
            from src.data.dataset_loader import DatasetLoader
            loader = DatasetLoader()
            
            if os.path.isdir(path):
                return loader.load_from_directory(path)
            else:
                return loader.load_from_file(path)

        # Handle HuggingFace datasets
        splits_to_try = [split, "train", "train_sft", "train_gen", "test_sft"]
        if split not in splits_to_try:
            splits_to_try.insert(0, split)
        
        # Remove duplicates while preserving order
        splits_to_try = list(dict.fromkeys(splits_to_try))
        
        token = os.environ.get("HF_TOKEN")
        last_exception = None

        for s in splits_to_try:
            try:
                self.logger.info(f"Split deneniyor: {s}")
                if config:
                    dataset = load_dataset(
                        dataset_name, config, split=s, token=token
                    )
                else:
                    dataset = load_dataset(
                        dataset_name, split=s, token=token
                    )

                conversations = self._convert_to_conversations(dataset, max_samples)
                if len(conversations) > 0:
                    self.logger.info(f"Dataset indirildi: {len(conversations)} örnek (Split: {s})")
                    return conversations
                else:
                    self.logger.info(f"Split {s} boş veya uyumsuz formatta, sıradakine geçiliyor...")
                    continue
            except (ValueError, Exception) as e:
                last_exception = e
                err_str = str(e).lower()
                # Continue trying other splits for any split-related error
                if "unknown split" in err_str or "split" in err_str or "should be one of" in err_str:
                    continue
                else:
                    self.logger.error(f"Dataset indirme hatası: {e}")
                    raise

        err_str = str(last_exception).lower()
        if "unknown split" in err_str or "should be one of" in err_str:
            # All splits tried and none worked — return empty instead of crashing
            self.logger.warning(
                f"Dataset '{dataset_name}' için hiçbir uyumlu split veya format bulunamadı. "
                f"Son hata: {last_exception}"
            )
            return []
        self.logger.error(f"Dataset indirme hatası (hiçbir split bulunamadı): {last_exception}")
        raise last_exception

    def _convert_to_conversations(
        self, dataset, max_samples: Optional[int] = None
    ) -> List[Dict[str, str]]:
        conversations = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            conv = self._extract_conversation(item)
            if conv:
                conversations.append(conv)

        return conversations

    def _extract_conversation(self, item: Dict) -> Optional[Dict[str, str]]:
        # Case-insensitive key mapping
        k = {key.lower(): key for key in item.keys()}
        
        # 1. Alpaca Style (instruction, input, output)
        if "instruction" in k and ("output" in k or "response" in k or "answer" in k):
            inst = item.get(k["instruction"], "")
            inp = item.get(k.get("input", ""), "")
            out_key = k.get("output") or k.get("response") or k.get("answer")
            out = item.get(out_key, "")
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            if user_msg and out:
                return {"user": user_msg, "assistant": out}

        # 2. Prompt/Completion or Prompt/Response
        if "prompt" in k and ("completion" in k or "response" in k or "answer" in k):
            out_key = k.get("completion") or k.get("response") or k.get("answer")
            if item[k["prompt"]] and item[out_key]:
                return {"user": str(item[k["prompt"]]), "assistant": str(item[out_key])}

        # 3. Question/Answer Style
        if "question" in k and ("answer" in k or "response" in k):
            out_key = k.get("answer") or k.get("response")
            return {"user": str(item[k["question"]]), "assistant": str(item[out_key])}

        # 4. Messages Style (ChatML / ShareGPT)
        if "messages" in k:
            messages = item[k["messages"]]
            user_msg = ""
            assistant_msg = ""
            for msg in messages:
                role = str(msg.get("role", "")).lower()
                content = msg.get("content", "")
                if role == "user":
                    user_msg = content
                elif role in ["assistant", "bot", "model"]:
                    assistant_msg = content
            if user_msg and assistant_msg:
                return {"user": user_msg, "assistant": assistant_msg}

        # 5. Translation Style
        if "translation" in k:
            trans = item[k["translation"]]
            langs = list(trans.keys())
            if len(langs) >= 2:
                return {"user": str(trans[langs[0]]), "assistant": str(trans[langs[1]])}

        # 6. Plain Text Fallback
        if "text" in k:
            text = str(item[k["text"]])
            if "### Human:" in text and "### Assistant:" in text:
                parts = text.split("### Assistant:")
                user_part = parts[0].replace("### Human:", "").strip()
                assistant_part = parts[1].split("###")[0].strip() if len(parts) > 1 else ""
                return {"user": user_part, "assistant": assistant_part}
            elif len(text) > 20:
                half = len(text) // 2
                return {"user": text[:half].strip(), "assistant": text[half:].strip()}

        # Log unrecognized schema — helps debug new datasets
        self.logger.debug(f"Tanınmayan veri formatı! Anahtarlar: {list(item.keys())}")
        return None

    def download_and_save(
        self,
        dataset_key: str,
        output_filename: Optional[str] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
    ) -> str:
        conversations = self.download_dataset(dataset_key, max_samples, split)

        output_filename = output_filename or f"{dataset_key}.json"
        output_path = Path(self.cache_dir) / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Dataset kaydedildi: {output_path}")
        return str(output_path)

    def download_custom(
        self,
        huggingface_id: str,
        config: Optional[str] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
    ) -> List[Dict[str, str]]:
        self.logger.info(f"Özel dataset indiriliyor: {huggingface_id}")

        splits_to_try = [split, "train", "train_sft", "train_gen", "test_sft"]
        splits_to_try = list(dict.fromkeys(splits_to_try))
        
        token = os.environ.get("HF_TOKEN")
        last_exception = None

        for s in splits_to_try:
            try:
                self.logger.info(f"Split deneniyor: {s}")
                if config:
                    dataset = load_dataset(
                        huggingface_id, config, split=s, token=token, trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        huggingface_id, split=s, token=token, trust_remote_code=True
                    )

                return self._convert_to_conversations(dataset, max_samples)
            except Exception as e:
                last_exception = e
                if "Unknown split" in str(e) or "split" in str(e).lower():
                    continue
                else:
                    self.logger.error(f"Özel dataset indirme hatası: {e}")
                    raise

        self.logger.error(f"Özel dataset indirme hatası: {last_exception}")
        raise last_exception

    def merge_datasets(
        self,
        dataset_keys: List[str],
        max_samples_per_dataset: Optional[int] = None,
        output_filename: str = "merged_dataset.json",
    ) -> str:
        all_conversations = []

        for key in dataset_keys:
            try:
                convs = self.download_dataset(key, max_samples_per_dataset)
                all_conversations.extend(convs)
                self.logger.info(f"{key}: {len(convs)} örnek eklendi")
            except Exception as e:
                self.logger.error(f"{key} indirilemedi: {e}")

        output_path = Path(self.cache_dir) / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_conversations, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Toplam {len(all_conversations)} örnek kaydedildi: {output_path}"
        )
        return str(output_path)

    def download_file(self, url: str, destination: str) -> str:
        """Downloads a file from a URL with progress logging."""
        self.logger.info(f"Dosya indiriliyor: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                
        self.logger.info(f"İndirme tamamlandı: {destination}")
        return destination

    def extract_archive(self, archive_path: str, extract_to: Optional[str] = None) -> str:
        """Extracts zip, tar, tar.gz, etc. to a directory."""
        archive_path = Path(archive_path)
        if not extract_to:
            extract_to = str(archive_path.parent / archive_path.stem)
        
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Arşiv çıkarılıyor: {archive_path} -> {extract_to}")
        
        shutil.unpack_archive(str(archive_path), extract_to)
        self.logger.info(f"Çıkarma işlemi başarılı.")
        return extract_to

    def download_url_and_extract(self, url: str) -> str:
        """Downloads a URL and automatically extracts if it's an archive."""
        filename = url.split('/')[-1] or "downloaded_data"
        if "?" in filename:
            filename = filename.split("?")[0]
        
        destination = Path(self.cache_dir) / filename
        
        self.download_file(url, str(destination))
        
        # Check if it's an archive
        suffixes = ['.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz']
        if any(filename.lower().endswith(s) for s in suffixes):
            return self.extract_archive(str(destination))
        
        return str(destination)

    def get_dataset_info(self, dataset_key: str) -> Dict[str, Any]:
        if dataset_key not in self.READY_DATASETS:
            return {"error": "Dataset bulunamadı"}

        info = self.READY_DATASETS[dataset_key].copy()
        info["key"] = dataset_key
        return info


def list_all_datasets() -> None:
    downloader = DatasetDownloader()
    datasets = downloader.list_available_datasets()

    print("\n" + "=" * 60)
    print("KULLANILABİLİR DATASET'LER")
    print("=" * 60)

    for key, info in datasets.items():
        print(f"\n📦 {key}")
        print(f"   İsim: {info['name']}")
        print(f"   Açıklama: {info['description']}")
        print(f"   Boyut: {info['size']}")
        print(f"   Diller: {', '.join(info['languages'])}")

    print("\n" + "=" * 60)
    print("Kullanım örneği:")
    print("  downloader = DatasetDownloader()")
    print(
        "  data = downloader.download_dataset('python_instructions', max_samples=1000)"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    list_all_datasets()
