import time
import torch
import psutil
import logging
from typing import List, Dict
from src.utils.config import Config
from src.core.model_manager import ModelManager
from src.core.inference import InferenceEngine

def setup_logger():
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class BenchmarkRunner:
    def __init__(self, model_name: str = "CodeMind-350M-MoE"):
        self.logger = setup_logger()
        self.config = Config()
        
        self.logger.info(f"Yükleniyor: {model_name}...")
        self.model_manager = ModelManager(self.config)
        self.model_manager.load_model(model_name)
        
        self.inference_engine = InferenceEngine(self.model_manager, config=self.config)
        
        self.test_prompts = [
            "Merhaba, nasılsın? Bana kısaca yapay zekanın ne olduğunu açıkla.",
            "Write a Python function to calculate the Fibonacci sequence up to n terms.",
            "Explain quantum computing to a 5-year-old in simple terms.",
            "Türkiye'nin coğrafi bölgelerini ve özelliklerini maddeler halinde yaz."
        ]

    def measure_memory(self) -> Dict[str, str]:
        stats = {}
        if torch.cuda.is_available():
            stats['gpu_allocated'] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            stats['gpu_reserved'] = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
        
        process = psutil.Process()
        stats['ram_usage'] = f"{process.memory_info().rss / 1e9:.2f} GB"
        return stats

    def run_throughput_test(self, num_runs: int = 1):
        self.logger.info("=== Throughput Testi Başlıyor ===")
        
        total_tokens = 0
        total_time = 0.0
        
        for idx, prompt in enumerate(self.test_prompts):
            self.logger.info(f"\nPrompt {idx+1}/{len(self.test_prompts)}: {prompt[:50]}...")
            
            # TTFT (Time to First Token) and Generation Phase
            start_time = time.time()
            first_token_time = None
            
            output_tokens = ""
            for token_chunk in self.inference_engine.generate_stream(prompt, use_memory=False):
                if first_token_time is None:
                    first_token_time = time.time()
                output_tokens += token_chunk
            
            end_time = time.time()
            
            # Metrics
            ttft = first_token_time - start_time if first_token_time else 0.0
            gen_time = end_time - start_time
            
            # Approximate token count (for exact, we'd use tokenizer)
            tok_count = len(self.model_manager.tokenizer.encode(output_tokens)) if self.model_manager.tokenizer else len(output_tokens.split())
            
            tokens_per_sec = tok_count / gen_time if gen_time > 0 else 0
            
            total_tokens += tok_count
            total_time += gen_time
            
            self.logger.info(f"Sonuç: {tok_count} token üretildi.")
            self.logger.info(f"TTFT (İlk Token Süresi): {ttft:.3f} sn")
            self.logger.info(f"Toplam Jenerasyon Süresi: {gen_time:.3f} sn")
            self.logger.info(f"Hız: {tokens_per_sec:.2f} token/sn")
            
            mem_stats = self.measure_memory()
            self.logger.info(f"Hafıza: {mem_stats}")
            
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        self.logger.info(f"\n=== Test Tamamlandı ===")
        self.logger.info(f"Ortalama Hız: {avg_speed:.2f} token/sn")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CodeMind Benchmark Runner")
    parser.add_argument("--model", type=str, default="CodeMind-350M-MoE", help="Test edilecek model adı")
    args = parser.parse_args()
    
    runner = BenchmarkRunner(model_name=args.model)
    runner.run_throughput_test()
