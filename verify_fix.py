import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from core.model.codemind import CodeMindConfig, CodeMindForCausalLM

def test_forward_with_output_attentions():
    print("Testing CodeMindAttention.forward with output_attentions...")
    
    config = CodeMindConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128
    )
    
    model = CodeMindForCausalLM(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Test case 1: Standard forward
    print("Calling forward...")
    outputs = model(input_ids)
    print("Standard forward success!")
    
    # Test case 2: Forward with output_attentions=True
    print("Calling forward with output_attentions=True...")
    outputs = model(input_ids, output_attentions=True)
    if outputs.attentions is not None:
        print(f"Success! Attentions shape: {len(outputs.attentions)} layers, each {outputs.attentions[0].shape}")
    else:
        print("FAILED: Attentions are None even though output_attentions=True")
        sys.exit(1)
        
    # Test case 3: Forward with extra kwargs (to test **kwargs)
    print("Calling forward with extra kwargs...")
    outputs = model(input_ids, some_random_kwarg="test")
    print("Extra kwargs success!")
    
    print("All verification tests passed!")

if __name__ == "__main__":
    test_forward_with_output_attentions()
