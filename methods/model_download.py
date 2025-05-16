import os
import modelscope
import huggingface_hub

proxy = "http://10.31.100.51:7890"
os.environ["proxy"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["ftp_proxy"] = proxy

url = "huggingface"

model_name = "tofu_Llama-3.1-8B-Instruct_full"
model_id = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
save_path = f"./models/{model_name}"

if url == "modelscope":
    modelscope.snapshot_download(repo_id=model_id, local_dir=save_path)

if url == "huggingface":
    huggingface_hub.snapshot_download(repo_id=model_id, local_dir=save_path)