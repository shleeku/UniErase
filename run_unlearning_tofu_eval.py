import os
import subprocess
import sys

proxy = "http://10.31.100.51:7890"
os.environ["proxy"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["ftp_proxy"] = proxy

no_proxy_list = [
    "localhost", "127.0.0.1",
    ".huggingface.co", ".hf.co",
    "huggingface.co", "hf.co",
    "cdn-lfs.hf.co", "cdn-lfs.huggingface.co"
]
os.environ["NO_PROXY"] = ",".join(no_proxy_list)
os.environ["no_proxy"] = os.environ["NO_PROXY"]

def abspath(*p):
    return os.path.abspath(os.path.join(*p))

def main():
    # Configuration
    # project_root = "./closer-look-LLM-unlearning"
    project_root = os.path.abspath("./")
    eval_workdir = abspath(project_root, "closer-look-LLM-unlearning")
    eval_script = abspath(eval_workdir, "eval.py")
    model_paths = [
        abspath(project_root, "data/models/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share"),
    ]
    load_model_path = abspath(project_root, "edited_model/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share/AlphaEdit_400_test.pth")
    data_path = abspath(project_root, "closer-look-LLM-unlearning/data/tofu/task_data/forget10")
    save_root = abspath(project_root, "results/tofu")
    # forget_losses = ["GA+GD", "GA+KL", "NPO+GD", "NPO+KL",
    #                  "ME+GD", "DPO+GD", "DPO+KL", "IDK+AP"]
    forget_losses = ["None"]
    task_list = [1]
    learning_rates = [1e-5]
    mask = True
    use_LoRA = False
    # save_root = "results/tofu"
    forget_coeff = 1.0
    regularization_coeff = 1.0
    save_checkpoint = False
    num_epochs = 5
    batch_size = 32
    max_length = 96
    # eval_steps = ["last"]
    eval_steps = [0]
    eval_task_name = "UniErase-M"  # or "Baseline" or "UniErase" or "UNL"

    model_family = "llama3.1-8b"

    # Convert task list to comma-separated string for environment variable
    os.environ["TASK_LIST"] = ",".join(map(str, task_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define splits to process
    splits = ["forget10"]

    # model_paths = [
    #     # "./data/models/tofu_Llama-3.1-8B-Instruct_full",
    #     # "./data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu",
    #     "./data/models/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share",
    # ]

    # load_model_path = "./edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu/AlphaEdit_400_batched_tofu_multi.pth"
    # data_path = "./closer-look-LLM-unlearning/data/tofu/task_data/forget10"

    # 获取当前 Python 解释器的路径
    python_executable = sys.executable

    for model_path in model_paths:
        for split in splits:
            for forget_loss in forget_losses:
                for lr in learning_rates:
                    for task_id in task_list:
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
                            f"+eval_task_name={eval_task_name}",
                        ]

                        if "UniErase" in eval_task_name:
                            common_args.append(f"+load_model_path={load_model_path}")

                        if "None" in forget_loss:
                            common_args.append(f"+custom_data_path={data_path}")

                        # Run eval.py for each step (单卡版本)
                        for step in eval_steps:
                            eval_cmd = [python_executable,  # 使用当前 Python 解释器
                                        "eval.py",
                                        "--config-name=tofu.yaml",
                                        f"task_id={task_id}",
                                        f"eval_unlearn_step={step}"
                                        ] + common_args

                            subprocess.run(" ".join(eval_cmd), shell=True, check=True, cwd=eval_workdir)


if __name__ == "__main__":
    main()
