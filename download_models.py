from huggingface_hub import snapshot_download
import os
import sys

# Destination folder
# model_path = "./data/models/tofu_Llama-3.2-1B-Instruct_full"
# model_path = "./data/models/tofu_Llama-2-7b-chat-hf_full"
# model_path = "./data/models/Llama-3.2-1B-Instruct"
model_path = "./data/models/Llama-2-7b-chat-hf"
os.makedirs(model_path, exist_ok=True)

# Download model from HF
# snapshot_download(repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full", local_dir=model_path)
# snapshot_download(repo_id="open-unlearning/tofu_Llama-2-7b-chat-hf_full", local_dir=model_path)
# snapshot_download(repo_id="meta-llama/Llama-3.2-1b-Instruct", local_dir=model_path)
snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf", local_dir=model_path)
