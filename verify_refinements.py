
import sys
import os
import torch
from typing import List

# Add src to path
sys.path.append(os.path.abspath("."))

from src.core.tokenizer.advanced_tokenizer import AdvancedCodeTokenizer, CODE_SPECIAL_TOKENS
from src.core.datasets import TextDataset, ConversationDataset
from src.core.prompting import build_instruction_prompt, TOKENS

def test_tokenizer_special_tokens():
    print("--- Testing Tokenizer Special Tokens ---")
    tokenizer = AdvancedCodeTokenizer()
    # We need to simulate a trained state or check defined special tokens
    # Note: AdvancedCodeTokenizer uses HuggingFace tokenizers under the hood
    
    special_tokens = tokenizer.special_tokens
    print(f"Special tokens count: {len(special_tokens)}")
    
    if "<|tr|>" in CODE_SPECIAL_TOKENS:
        print("SUCCESS: <|tr|> is in CODE_SPECIAL_TOKENS list.")
    else:
        print("FAILURE: <|tr|> is MISSING from CODE_SPECIAL_TOKENS list.")

    # Try encoding <|tr|>
    # Since it's a new token, it might not be in the VOCAB if not retrained, 
    # but it SHOULD be in the special tokens list for the next training run.
    print(f"Tokenizer special tokens: {special_tokens}")

def test_text_dataset_formatting():
    print("\n--- Testing TextDataset Formatting ---")
    tokenizer = AdvancedCodeTokenizer()
    texts = ["Merhaba dünya!"]
    
    # Test with Turkish
    ds_tr = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        language="tr",
        pack_sequences=False
    )
    
    # Test with General
    ds_gen = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        language="general",
        pack_sequences=False
    )
    
    # We can't easily check the raw string inside encoded samples, 
    # but we can check the build_instruction_prompt output directly
    prompt_tr = build_instruction_prompt(user=texts[0], language="tr", include_eos=False)
    prompt_gen = build_instruction_prompt(user=texts[0], language="general", include_eos=False)
    
    print(f"Turkish Prompt: {repr(prompt_tr)}")
    print(f"General Prompt: {repr(prompt_gen)}")
    
    if prompt_tr.startswith("<|tr|>"):
        print("SUCCESS: Turkish prompt starts with <|tr|>")
    else:
        print("FAILURE: Turkish prompt DOES NOT start with <|tr|>")
        
    if not prompt_gen.startswith("<|general|>") and TOKENS.user in prompt_gen:
        print("SUCCESS: General prompt does not have language tag (correct behavior)")

def test_conversation_dataset_formatting():
    print("\n--- Testing ConversationDataset Formatting ---")
    tokenizer = AdvancedCodeTokenizer()
    convs = [{"user": "Nasılsın?", "assistant": "İyiyim, teşekkürler."}]
    
    ds = ConversationDataset(
        conversations=convs,
        tokenizer=tokenizer,
        language="tr",
        pack_sequences=False
    )
    
    prompt = build_instruction_prompt(
        user=convs[0]["user"],
        assistant=convs[0]["assistant"],
        language="tr",
        include_eos=False
    )
    
    print(f"Conversation Prompt: {repr(prompt)}")
    if prompt.startswith("<|tr|>") and TOKENS.assistant in prompt:
        print("SUCCESS: Conversation prompt is correctly formatted.")
    else:
        print("FAILURE: Conversation prompt formatting issue.")

if __name__ == "__main__":
    try:
        test_tokenizer_special_tokens()
        test_text_dataset_formatting()
        test_conversation_dataset_formatting()
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        import traceback
        traceback.print_exc()
