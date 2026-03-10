import torch
from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM
from transformers import AutoTokenizer

def test_loss():
    # Setup tokenizer exactly as in model_manager.py
    pretrained_tok = "ytu-ce-cosmos/Turkish-GPT2-large"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tok, trust_remote_code=True)
    special_tokens = ["<|pad|>", "<|eos|>", "<|unk|>", "<|python|>", "<|dart|>", "<|javascript|>", "▁"]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    if tokenizer.pad_token is None:
        if "<|pad|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|pad|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
            
    vocab_size = len(tokenizer)
    
    # Setup config
    config = CodeMindConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_experts=1,
        num_experts_per_tok=1,
        intermediate_size=768 * 4,
        max_position_embeddings=4096,
        tie_word_embeddings=True
    )
    
    model = CodeMindForCausalLM(config).half().cuda()
    model.eval()
    
    # Real text
    text = "Aşağıdaki Türkçe metni özetle: \nTürkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur. ▁ Ankara, iç anadolu bölgesinde bulunur."
    tokens = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    
    input_ids = tokens["input_ids"].cuda()
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        
    print(f"Loss on real text: {outputs.loss.item()}")
    
    # Let's check what tie_weights does
    model.tie_weights()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    print(f"Loss after manual tie_weights: {outputs.loss.item()}")

if __name__ == "__main__":
    test_loss()
