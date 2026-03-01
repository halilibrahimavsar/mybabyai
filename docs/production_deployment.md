# MyBabyAI Production Deployment Guide

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended for training)
- Minimum 16GB System RAM (32GB recommended)

## Installation

1. Clone the repository and navigate to the project root:
   ```bash
   git clone <repository_url>
   cd mybabyai
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have a specific CUDA version, you may need to install PyTorch manually according to [pytorch.org](https://pytorch.org).*

## Configuration

The application uses `configs/config.yaml` for configuration. The default configuration is created if missing.
Edit it as needed to change resource allocations:

- `model.device`: "auto" (automatically detects CUDA/MPS), "cuda", or "cpu".
- `model.load_in_4bit`: Enable this (True) to load the base model in 4-bit precision, significantly reducing VRAM footprint.
- `training.per_device_train_batch_size` & `training.gradient_accumulation_steps`: Adjust block sizes according to your VRAM.
- `training.qlora.use_qlora`: Set to True for efficient QLoRA fine-tuning.

## Running the Application

To launch the GUI:

```bash
python main.py
```

The application checks for required dependencies upon startup and gracefully warns you if anything is missing.

## Memory & Checkpoint Management

- Checkpoints for CodeMind are stored at `<basedir>/codemind/checkpoints/`.
- Training models and adapters are stored in `<basedir>/models/fine_tuned/`.
- Vector databases (for RAG memory) are stored at `<basedir>/data/chroma/`.

Ensure that you have sufficient disk space. If you want to back up your trained models, compress the `models/fine_tuned/` and `codemind/checkpoints/` folders.

## Logging

Application logs are written to `<basedir>/logs/app.log`. 
If you encounter any unexpected behaviors or errors, please refer to this file as it captures error details down to the layer where they occurred.

## Optional Capabilities

- **RAG Memory Search**: Ensure `chromadb` is installed. The Agent will use previous chat logs or inserted knowledge chunks for context generation.
- **Custom Adapters**: You can train multiple LoRA adapters and load them as needed from the Trainer dashboard.
