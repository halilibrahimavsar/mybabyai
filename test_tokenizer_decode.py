import sys
sys.path.append("/home/garuda/Masaüstü/mybabyai")

from src.core.tokenizer.code_tokenizer import CodeTokenizer

# Load tokenizer from a model dir or instantiate
tokenizer = CodeTokenizer()

text = "def hello():\n    print('hello world')"
ids = tokenizer.encode(text)

decoded = tokenizer.decode(ids)
print("Decoded text:")
print(repr(decoded))
print(decoded)
