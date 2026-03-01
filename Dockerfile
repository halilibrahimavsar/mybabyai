# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# We exclude GUI components if running in headless cloud mode usually, 
# but for simplicity we copy all and users can run training scripts.
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run a training script (to be created or used by user)
# Usage: docker run --gpus all mybabyai python src/core/trainer_cli.py --url https://en.wikipedia.org/wiki/Artificial_intelligence
CMD ["python", "main.py"] 
