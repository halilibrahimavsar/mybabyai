import sys
import os
from pathlib import Path

# Fix path to include project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.model_manager import ModelManager
from src.utils.config import Config

def test_model_resolution_bypass():
    print("\n--- Model Name Resolution Bypass Test ---")
    
    # 1. Create a config with a non-default model
    config = Config()
    # Manually override a config value if possible, or just check the code behavior
    # In src/core/model_manager.py, resolve_model_name is:
    # def resolve_model_name(self, model_name: Optional[str] = None) -> str:
    #     return "CodeMind-125M"
    
    mm = ModelManager(config)
    
    requested_model = "microsoft/phi-2"
    resolved = mm.resolve_model_name(requested_model)
    
    print(f"Requested Model: {requested_model}")
    print(f"Resolved Model:  {resolved}")
    
    if resolved == "CodeMind-125M" and requested_model != "CodeMind-125M":
        print("RESULT: ISSUE CONFIRMED - Model name is hardcoded to CodeMind-125M and ignores input.")
    else:
        print("RESULT: ISSUE NOT FOUND - Resolution seems to respect input (or input was default).")

    # 2. Check load_model behavior
    print("\nChecking load_model behavior...")
    # mm.load_model covers resolution internally
    # return self._base_load_model(self.resolve_model_name(model_name))
    
    # We won't actually call load_model because it loads weights, 
    # but the code inspection already confirmed it.
    
if __name__ == "__main__":
    test_model_resolution_bypass()
