import logging
from typing import Optional
from dataclasses import dataclass
import threading

from src.core.memory import MemoryManager
from src.core.trainer import LoRATrainer

logger = logging.getLogger(__name__)

@dataclass
class NightShiftConfig:
    min_experiences_to_train: int = 50
    top_k_experiences: int = 100
    learning_rate: float = 2e-5
    epochs: int = 2

class NightShift:
    """
    Handles the Continuous Learning (Night Shift) process.
    It takes high-reward experiences from the Experience Buffer (via MemoryManager)
    and uses them to fine-tune the base model (System 1) via LoRA, effectively
    transferring System 2 "slow thinking" successes into System 1 "fast intuition".
    """
    def __init__(self, memory_manager: MemoryManager, trainer: LoRATrainer, config: Optional[NightShiftConfig] = None):
        self.memory = memory_manager
        self.trainer = trainer
        self.config = config or NightShiftConfig()
        self._is_running = False

    def start_night_shift(self, background: bool = True):
        """
        Initiates the night shift process.
        """
        if self._is_running:
            logger.warning("Night Shift is already running.")
            return

        if background:
            thread = threading.Thread(target=self._run_night_shift_pipeline)
            thread.daemon = True
            thread.start()
        else:
            self._run_night_shift_pipeline()

    def _run_night_shift_pipeline(self):
        self._is_running = True
        logger.info("=== STARTING NIGHT SHIFT ===")
        
        try:
            # 1. Zikir: Memory Consolidation
            self.memory.zikir()
            
            # 2. Extract Experiences for Training
            conversations = self._extract_training_data()
            
            if len(conversations) < self.config.min_experiences_to_train:
                logger.info(f"Not enough high-quality experiences yet ({len(conversations)} < {self.config.min_experiences_to_train}). Skipping fine-tuning.")
                return
                
            # 3. Fine-Tune System 1 (LoRA)
            logger.info(f"Fine-tuning System 1 with {len(conversations)} System 2 experiences...")
            
            # Using LoRATrainer's capability
            metrics = self.trainer.train_from_conversations(
                conversations=conversations,
                training_type="lora",
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4
            )
            
            logger.info(f"Night Shift Fine-tuning completed successfully! Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Night Shift encountered an error: {e}")
            
        finally:
            self._is_running = False
            logger.info("=== NIGHT SHIFT ENDED ===")
            
    def _extract_training_data(self) -> list:
        # Query the ChromaDB specifically for experiences
        # Since we use metadata {"type": "system2_experience"}
        results = self.memory.search_documents(
            query="REASONING", # Dummy query, the where clause does the heavy lifting
            n_results=self.config.top_k_experiences,
            where={"type": "system2_experience"}
        )
        
        conversations = []
        if not results.get("documents"):
             return conversations
             
        for doc in results["documents"]:
            # Basic parsing of the experience format
            # "Query: {query}\nReasoning Steps:\n{thought_chain}\nFinal Reward Score: ..."
            try:
                query_part = doc.split("Query: ")[1].split("\nReasoning Steps:\n")[0]
                reasoning_part = doc.split("Reasoning Steps:\n")[1].split("\nFinal Reward Score:")[0]
                
                conversations.append({
                    "user": query_part.strip(),
                    "assistant": reasoning_part.strip()
                })
            except Exception as e:
                logger.debug(f"Failed to parse experience doc: {e}")
                continue
                
        return conversations
