import torch
from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM
from transformers import AutoTokenizer

def test_initial_loss():
    config = CodeMindConfig(
        vocab_size=50257,
        hidden_size=448,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        tie_word_embeddings=True
    )
    
    model = CodeMindForCausalLM(config)
    print(f"lm_head type: {type(model.lm_head)}")
    print(f"Word embeddings weight matched to lm_head weight: {model.get_input_embeddings().weight is model.lm_head.weight}")
    
test_initial_loss()
