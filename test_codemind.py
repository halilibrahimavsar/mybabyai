import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from src.core.model_manager import ModelManager
from src.core.inference import InferenceEngine
from src.utils.config import Config

def test_generation():
    print("Loading CodeMind config...")
    config = Config()
    # Force mock memory manager to avoid chroma db download output length
    print("Initializing ModelManager...")
    manager = ModelManager(config)
    print("Loading CodeMind model explicitly...")
    manager.load_model("CodeMind-125M (Local)")
    
    print("Initializing InferenceEngine...")
    engine = InferenceEngine(manager)
    
    print("Testing generation (cached)...")
    try:
        response = engine.generate("def hello_world():", max_new_tokens=20, use_memory=False)
        print("Generation successful!")
        print(f"Output: {response}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_generation()
