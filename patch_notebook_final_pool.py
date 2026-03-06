import json

with open("cloud_train.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        if "dataset_pool =" in source and "merve" in source:
            new_pool_code = """# @title 5. 🎯 DATASET HAVUZU (TÜRKÇE ODAKLI)

# 💡 T4 için max_samples 5000-10000 arasında tutulması önerilir.
#    Daha fazlası işleme süresi uzatır ama OOM'a neden olmaz (tokeniz. CPU'da yapılır).
dataset_pool = [
    {
        "name": "🇹🇷 Turkish Instructions (Merve)",
        "type": "huggingface",
        "dataset_key": "turkish_instructions_merve",
        "max_samples": 5000
    },
    {
        "name": "🇹🇷 Turkish Alpaca",
        "type": "huggingface",
        "dataset_key": "turkish_alpaca",
        "max_samples": 5000
    },
    {
        "name": "🇹🇷 GPT-4 Alpaca TR",
        "type": "huggingface",
        "dataset_key": "alpaca_gpt4_tr",
        "max_samples": 10000
    },
    {
        "name": "🇹🇷 OpenOrca TR",
        "type": "huggingface",
        "dataset_key": "open_orca_tr",
        "max_samples": 10000
    }
]

print(f"📊 Veri havuzu hazır: {len(dataset_pool)} kaynak.")
"""
            cell["source"] = [line + '\n' for line in new_pool_code.split('\n')]

with open("cloud_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
