import os
import shutil
import subprocess
import sys

from methods import methods
import train_unl
import run_edit_precise

for precise_id in range(1,2):
    unlearn_batch_size = 1
    batch_size = 1
    max_length = 128

    tofu_forget_ds = methods.load_jsonl("/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/tofu/forget01_subject.jsonl")
    forget_ds = [tofu_forget_ds[precise_id]]
    print(forget_ds)

    model_path = "/data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu"
    run_edit_precise.run(model_path, precise_id, forget_ds)

    # Configuration
    project_root = "/data/ym/Unlearning_Token/closer-look-LLM-unlearning"
    # forget_losses = ["GA+GD", "GA+KL", "NPO+GD", "NPO+KL",
    #                  "ME+GD", "DPO+GD", "DPO+KL", "IDK+AP"]
    forget_losses = ["None"]

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
    # eval_steps = ["last"]
    eval_steps = [0]
    eval_task_name = "UniErase"  # or "Baseline" or "UniErase" or "UNL"

    model_family = "llama3.1-8b"

    # Convert task list to comma-separated string for environment variable
    os.environ["TASK_LIST"] = ",".join(map(str, task_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define splits to process
    splits = ["forget01"]

    model_paths = [
        "/data/ym/models/tofu_Llama-3.1-8B-Instruct_full",
    ]

    load_model_path = "/data/ym/Unlearning_Token/edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu/AlphaEdit_precise_tofu_temp.pth"
    data_path = f"/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/tofu/task_data/forget01/precise/{precise_id}"

    os.makedirs(data_path, exist_ok=True)
    methods.save_jsonl(f"{data_path}/forget.json", forget_ds)
    retain_ds_0 = methods.load_jsonl("/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/tofu/task_data/forget01/retain.json")
    methods.save_jsonl(f"{data_path}/retain.json", retain_ds_0)
    forget_perturbed_0 = methods.load_jsonl(f"/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/tofu/task_data/forget01/forget_perturbed.json")
    methods.save_jsonl(f"{data_path}/forget_perturbed.json", [retain_ds_0[precise_id]])

    # 获取当前 Python 解释器的路径
    python_executable = sys.executable

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

                    if precise_id is not None:
                        common_args.append(f"+precise_id={precise_id}")
                        common_args.append(f"gradient_accumulation_steps=1")

                    # Run eval.py for each step (单卡版本)
                    for step in eval_steps:
                        eval_cmd = [python_executable,  # 使用当前 Python 解释器
                                    "eval.py",
                                    "--config-name=tofu.yaml",
                                    f"task_id={task_id}",
                                    f"eval_unlearn_step={step}"
                                    ] + common_args

                        subprocess.run(" ".join(eval_cmd), shell=True, check=True, cwd=project_root)

