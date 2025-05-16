import os
import random
import socket
import subprocess
import sys


def find_available_port(start=10000, end=60000):
    while True:
        port = random.randint(start, end)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))  # 尝试绑定端口
            sock.close()
            return port
        except socket.error:
            continue  # 端口被占用，继续尝试


def main():
    # Configuration
    project_root = "/data/ym/Unlearning_Token/closer-look-LLM-unlearning"
    forget_losses = ["GA+GD", "GA+KL", "NPO+GD", "NPO+KL",
                     "ME+GD", "DPO+GD", "DPO+KL", "IDK+AP"][:4]
    task_list = [1]
    learning_rates = [1e-5]
    mask = True
    use_LoRA = False
    save_root = "results/tofu"
    forget_coeff = 1.0
    regularization_coeff = 1.0
    save_checkpoint = False
    num_epochs = 5
    batch_size = 32
    max_length = 96
    save_steps = "last"
    splits = ["forget10"]

    model_paths = [
        "/data/models/tofu_Llama-3.1-8B-Instruct_full",
    ]
    model_family = "llama3.1-8b"

    # Convert task list to comma-separated string for environment variable
    os.environ["TASK_LIST"] = ",".join(map(str, task_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    MASTER_PORT = find_available_port()
    print(MASTER_PORT)
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    # 获取当前 Python 解释器的路径
    python_executable = sys.executable

    for model_path in model_paths:
        print(f"---Model path: {model_path}---")
        for split in splits:
            for forget_loss in forget_losses:
                for lr in learning_rates:
                    # Build common arguments
                    common_args = [
                        f"use_LoRA={use_LoRA}",
                        f"forget_coeff={forget_coeff}",
                        f"regularization_coeff={regularization_coeff}",
                        f"lr={lr}",
                        f"split={split}",
                        f"forget_loss={forget_loss}",
                        f"num_epochs={num_epochs}",
                        f"batch_size={batch_size}",
                        f"+max_length={max_length}",
                        f"mask={mask}",
                        f"fix_ref_model={False}",
                        f"save_root={save_root}",
                        f"save_checkpoint={save_checkpoint}",
                        f"+working_dir={project_root}",
                        f"model_family={model_family}",
                        f"model_path={model_path}",
                    ]

                    # Run forget.py (单卡版本)
                    forget_cmd = [python_executable,  # 如 `sys.executable` 或 `"python3"`
                                  "-m",  # 使用 `-m` 来运行模块
                                  "torch.distributed.run",  # `torchrun` 的实际模块名
                                  f"--nproc_per_node=1",
                                  f"--master_port={MASTER_PORT}",
                                  "forget.py",
                                  "--config-name=tofu.yaml",
                                  f"save_steps={save_steps}",
                                  ] + common_args

                    subprocess.run(" ".join(forget_cmd), shell=True, check=True, cwd=project_root)


if __name__ == "__main__":
    main()