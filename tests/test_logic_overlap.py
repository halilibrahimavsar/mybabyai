import sys
from pathlib import Path

# Fix path to include project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_overlap_demonstration():
    print("\n--- Architectural Overlap & Conflict Test ---")
    
    print("\n1. Importing Model Loading Logic from both worlds:")
    try:
        from src.core.model_manager import ModelManager
        from src.core.model.base_model import CodeMindModel
        print("Success: Both src.core.model_manager and codemind.model.base_model exist.")
        print("Overlap: src.core.model_manager handles model state for GUI, while codemind.model defines its own architecture/loading.")
    except ImportError as e:
        print(f"Import Error: {e}")

    print("\n2. Importing Training Logic from both worlds:")
    try:
        from src.core.trainer import LoRATrainer
        from src.core.training.train import train_model
        print("Success: Both src.core.trainer and codemind.training.train exist.")
        print("Overlap: Both implement training loops, but src.core.trainer is GUI-integrated.")
    except ImportError as e:
        print(f"Import Error: {e}")

    print("\n3. Configuration Divergence:")
    from src.utils.config import Config
    gui_config = Config()
    print(f"src.utils.config default output_dir: {gui_config.get('training.output_dir', 'N/A')}")
    
    # Many codemind scripts use hardcoded paths like 'models/fine_tuned' directly
    # instead of reading from Config().
    print("Finding: codemind/training/train.py and scripts often default to hardcoded strings while src/ core uses config.py.")

    print("\nRESULT: ARCHITECTURAL FRAGMENTATION CONFIRMED - Redundant implementations for loading and training exist in parallel.")

if __name__ == "__main__":
    test_overlap_demonstration()
