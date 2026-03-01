import torch
from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM

def test_codemind_moe():
    config = CodeMindConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2, # GQA
        intermediate_size=512,
        num_experts=4, # MoE
        num_experts_per_tok=2,
        output_router_logits=True
    )
    
    model = CodeMindForCausalLM(config)
    print("Model instantiated successfully.")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 1. Test Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    print(f"Forward pass successful. Loss: {loss.item()}")
    assert loss is not None
    
    # 2. Test Backward pass
    loss.backward()
    print("Backward pass successful.")
    
    # 3. Test Generation
    gen_tokens = model.generate(input_ids, max_new_tokens=10)
    print(f"Generation successful. Output shape: {gen_tokens.shape}")

if __name__ == "__main__":
    test_codemind_moe()
