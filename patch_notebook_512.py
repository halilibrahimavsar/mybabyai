import json

with open("cloud_train.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        if "max_seq_length =" in source:
            new_source = source.replace('max_seq_length = 256', 'max_seq_length = 512')
            if new_source != source:
                cell["source"] = [line + '\n' for line in new_source.split('\n')[:-1]]

with open("cloud_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
