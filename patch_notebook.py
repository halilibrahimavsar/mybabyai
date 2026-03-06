import json
import os

with open("cloud_train.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        if "output_dir =" in source and "batch_size = 1" in source:
            new_source = source.replace('learning_rate = 5e-5', 'learning_rate = 1e-4')
            new_source = new_source.replace('max_seq_length = 128', 'max_seq_length = 256')
            
            old_output_dir_lines = '''output_dir = "/content/drive/MyDrive/mybabyai_checkpoints" if ENV == "colab" else "/kaggle/working/checkpoints"

if ENV == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
'''
            new_output_dir_lines = '''if ENV == "colab":
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    output_dir = "/content/drive/MyDrive/mybabyai_checkpoints"
else:
    output_dir = "/kaggle/working/checkpoints"
'''
            new_source = new_source.replace(old_output_dir_lines, new_output_dir_lines)
            
            if old_output_dir_lines not in source:
                # Fallback, just try to replace the mount only
                new_source = new_source.replace("drive.mount('/content/drive')", "if not os.path.exists('/content/drive'):\n        drive.mount('/content/drive')\n    output_dir = '/content/drive/MyDrive/mybabyai_checkpoints'")
                
            # Write back cell source
            cell["source"] = [line + '\n' for line in new_source.split('\n')[:-1]] # keep structure

with open("cloud_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
