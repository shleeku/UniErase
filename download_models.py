from huggingface_hub import snapshot_download
import os
import sys

# Destination folder
model_path = "./data/models/tofu_Llama-3.2-1B-Instruct_full"
os.makedirs(model_path, exist_ok=True)

# Download model from HF
snapshot_download(repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full", local_dir=model_path)
