import torch
import os

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    
    try:
        # map_location='cpu' to avoid CUDA issues
        checkpoint = torch.load(path, map_location='cpu')
        
        print(f"--- Metrics for {os.path.basename(path)} ---")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"Step: {checkpoint['step']}")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.4f}")
        if 'best_loss' in checkpoint:
            print(f"Best Loss: {checkpoint['best_loss']:.4f}")
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")
        print("\n")
    except Exception as e:
        print(f"Error loading {path}: {e}")

check_checkpoint("/home/garuda/Masaüstü/mybabyai/codemind/checkpoints_instruct/model_best.pt")
check_checkpoint("/home/garuda/Masaüstü/mybabyai/codemind/checkpoints_instruct/model_final.pt")
check_checkpoint("/home/garuda/Masaüstü/mybabyai/codemind/checkpoints/model.pt")
