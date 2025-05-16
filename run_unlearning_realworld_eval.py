import os
import random
import subprocess
import sys

proxy = "http://10.31.100.51:7


def main():
    # Configuration
    project_root = "/data/ym/Unlearning_Token/closer-look-LLM-unlearning"
    # forget_losses = ["GA+GD", "GA+KL", "NPO+GD", "NPO+KL",
    #                  "ME+GD", "DPO+GD", "DPO+KL", "IDK+AP"]
    forget_losses = ["None"]
    task_list = [1]
    learning_rates = [1e-5]
    mask = True
    use_LoRA = False
    save_root = "results/real_world"
    forget_coeff = 1.0
    regularization_coeff = 1.0
    save_checkpoint = False
    num_epochs = 5
    batch_size = 32
    # eval_steps = ["last"]
    eval_steps = [0]
    eval_task_name = "UniErase-M"  # or "Baseline" or "UniErase" or "Base" or "UNL"

    model_paths = [
        # "/data/models/Llama-3.1-8B-Instruct",
        "/data/models/Llama-3.1-8B-Instruct-UL_real_world"
    ]
    model_family = "llama3.1-8b"

    load_model_path = "/data/ym/Unlearning_Token/edited_model/Llama-3.1-8B-Instruct-UL_real_world/AlphaEdit_400_batched_real_world_multi.pth"

    # Convert task list to comma-separated string for environment variable
    os.environ["TASK_LIST"] = ",".join(map(str, task_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define splits to process
    splits = ["forget"]

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
                        f"mask={mask}",
                        f"fix_ref_model={False}",
                        f"save_root={save_root}",
                        f"save_checkpoint={save_checkpoint}",
                        f"+working_dir={project_root}",
                        f"model_family={model_family}",
                        f"model_path={model_path}",
                        f"+eval_task_name={eval_task_name}",
                    ]

                    if "UniErase" in eval_task_name:
                        common_args.append(f"+load_model_path={load_model_path}")

                    # Run eval.py for each step (单卡版本)
                    for step in eval_steps:
                        eval_cmd = [python_executable,  # 使用当前 Python 解释器
                                    "real_world_eval.py",
                                    "--config-name=real_world.yaml",
                                    f"eval_unlearn_step={step}"
                                    ] + common_args

                        subprocess.run(" ".join(eval_cmd), shell=True, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
