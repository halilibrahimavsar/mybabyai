import json

with open("cloud_train.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        if "tr_wikipedia" in source:
            new_source = source.replace('"dataset_key": "tr_wikipedia"', '"dataset_key": "alpaca_gpt4_tr"')
            new_source = new_source.replace('"name": "🇹🇷 Turkish Wikipedia (Külliyat)"', '"name": "🇹🇷 GPT-4 Alpaca TR"')
            if new_source != source:
                cell["source"] = [line + '\n' for line in new_source.split('\n')[:-1]]

with open("cloud_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
