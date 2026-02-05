from huggingface_hub import snapshot_download
import os

repo_id = "castorini/rank_zephyr_7b_v1_full"
local_folder = "./models_and_benchmarks/rankzephyr" 
os.makedirs(local_folder, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_folder)

repo_id = "nreimers/MiniLM-L6-H384-uncased"
local_folder = "./models_and_benchmarks/em_model" 
os.makedirs(local_folder, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_folder)

repo_id = "Qwen/Qwen3-8B"
local_folder = "./models_and_benchmarks/qwen" 
os.makedirs(local_folder, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_folder)

repo_id = "HuggingFaceH4/zephyr-7b-beta"
local_folder = "./models_and_benchmarks/zephyr7b" 
os.makedirs(local_folder, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_folder)