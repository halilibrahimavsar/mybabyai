
import sys
import os
from pathlib import Path
import torch
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.trainer import LoRATrainer
from src.core.model_manager import ModelManager
from src.utils.config import Config

def test_progress():
    config = Config()
    model_manager = ModelManager(config)
    
    # Mocking a small training run is hard because it needs a model
    # Let's just test the UIProgressCallback logic directly if possible
    # or just manually verify the code logic.
    
    from src.core.trainer import UIProgressCallback
    
    def progress_fn(data):
        print(f"Progress Data: {data}")
        assert "grad_norm" in data
        assert "speed" in data
        assert "eta" in data
        assert "elapsed_time" in data
        print("Verification Successful!")

    callback = UIProgressCallback(progress_fn)
    
    class MockState:
        def __init__(self):
            self.global_step = 10
            self.max_steps = 100
            self.epoch = 0.5
            
    class MockArgs:
        pass
        
    state = MockState()
    args = MockArgs()
    logs = {"loss": 0.5, "learning_rate": 1e-4, "grad_norm": 0.1}
    
    print("Testing on_log callback...")
    callback.on_log(args, state, None, logs=logs)

if __name__ == "__main__":
    test_progress()
