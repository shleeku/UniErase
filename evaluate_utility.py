import os
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util
from methods import methods
from utility_evaluation import mmlu, trivia_qa, gsm8k, human_eval
import time

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

perplexity_model = "None"
perplexity_tokenizer = "None"

proxy = "http://10.31.100.51:7890"
os.environ["proxy"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["ftp_proxy"] = proxy


def query_batch(evaluator, model, tokenizer, prompts, use_chat_template):
    if use_chat_template:
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompt = [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evaluator.get_prompt(prompt)},
            ]
            formatted_prompts.append(formatted_prompt)
    else:
        formatted_prompts = [evaluator.get_hint_prompt(prompt) for prompt in prompts]
    responses = methods.local_generate_batch(model, tokenizer, formatted_prompts, use_chat_template,
                                             True, True, max_new_tokens=evaluator.max_new_tokens)
    return responses


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=False)
reject_list = []
with open("/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/idontknow.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = str(line.strip())
        reject_list.append(line)
reject_embeddings = embedding_model.encode(reject_list)


def equal_reject(model_answer):
    if model_answer in reject_list:
        return 1
    else:
        model_answer_embeddings = embedding_model.encode(model_answer)
        cos_scores = util.cos_sim(model_answer_embeddings, reject_embeddings)[0]
        max_score, max_idx = torch.max(cos_scores, dim=0)
        # print(max_score)
        if max_score < 0.5:
            max_score = 0
        if max_score > 0.75:
            max_score = 1
        return max_score


def calculate_perplexity(text):
    # 1. Tokenize 文本并转换为模型输入
    global perplexity_model, perplexity_tokenizer
    inputs = perplexity_tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(perplexity_model.device)

    # 2. 计算模型的负对数似然（Negative Log-Likelihood）
    with torch.no_grad():
        outputs = perplexity_model(input_ids, labels=input_ids)
        loss = outputs.loss  # 交叉熵损失（Cross-Entropy Loss）

    # 3. 计算困惑度（Perplexity = exp(loss)）
    perplexity = torch.exp(loss / len(input_ids)).item()

    return perplexity


def evaluate_model(evaluator, model, tokenizer, dataset, use_chat_template, batch_size):
    total = len(dataset)
    # Calculate the number of batches
    n_batches = (total + batch_size - 1) // batch_size  # Ceiling division

    dataset = sorted(dataset, key=lambda x: len(x[evaluator.sort_key]))

    # Process in batches with accurate progress bar
    for i in tqdm(range(0, total, batch_size), total=n_batches, desc="Evaluating"):
        batch_end = min(i + batch_size, total)
        batch_items = dataset[i:batch_end]

        # Prepare batch inputs
        prompts = []
        correct_answers = []
        for item in batch_items:
            prompt, correct_answer = evaluator.process(item)
            prompts.append(prompt)
            correct_answers.append(correct_answer)

        # Query model with batch
        model_answers = query_batch(evaluator, model, tokenizer, prompts, use_chat_template)
        for item, model_answer in zip(batch_items, model_answers):
            evaluator.append_record(item, model_answer)


def analyze(evaluator, model, tokenizer):
    results = []
    metrics = {"correct": 0, "rejection": 0, "token_nums": 0, "similarity": 0, "perplexity": 0, "follow": 0}
    with open(evaluator.output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            results.append(eval(line.strip()))

    for result in tqdm(results):
        item = result["data"]
        model_answer = result["model_answer"]
        _, correct_answer = evaluator.process(item)
        if evaluator.equal(model_answer, correct_answer, item):
            metrics["correct"] += 1
        if equal_reject(model_answer):
            metrics["rejection"] += 1
        if evaluator.follow_instruction(model_answer):
            metrics["follow"] += 1
        metrics["token_nums"] += len(tokenizer.tokenize(model_answer))
        metrics["similarity"] += \
            util.cos_sim(embedding_model.encode(model_answer), embedding_model.encode(correct_answer))[0].item()
        metrics["perplexity"] += calculate_perplexity(model_answer)

    for key in metrics.keys():
        metrics[key] /= len(results)

    for key, value in metrics.items():
        print("{}: {:.4f}".format(key, value))

    return metrics


def run(device, ds_name, n_sample, use_chat_template, batch_size, model_path, output_path, only_run, load_path=None):
    # 加载数据集
    evaluator = None
    if ds_name == "mmlu":
        ds = methods.load_jsonl("./dataset/mmlu/mmlu.jsonl")
        evaluator = mmlu.MMLUEvaluator()
    if ds_name == "trivia_qa":
        ds = methods.load_jsonl("./dataset/trivia_qa/trivia_qa.jsonl")
        evaluator = trivia_qa.TriviaQaEvaluator()
    if ds_name == "gsm8k":
        ds = methods.load_jsonl("./dataset/gsm8k/gsm8k.jsonl")
        evaluator = gsm8k.GSM8KEvaluator()
    if ds_name == "human_eval":
        ds = methods.load_jsonl("./dataset/human_eval/human_eval.jsonl")
        evaluator = human_eval.HumanEvalEvaluator()

    ds_list = list(ds)  # 转换为普通Python列表
    if n_sample is not None:
        random.seed(SEED)
        random.shuffle(ds_list)  # 打乱顺序
        if type(n_sample) == float:
            n_sample = int(n_sample * len(ds_list))
        ds_list = ds_list[:n_sample]

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    # 运行评估（可以设置max_samples限制测试样本数量）
    evaluator.set_output_path(output_path)
    if not os.path.exists(evaluator.output_path):
        evaluate_model(evaluator, model, tokenizer, ds_list, use_chat_template, batch_size)
        evaluator.save()
    else:
        print("Using cached...")

    if not only_run:
        metrics = analyze(evaluator, model, tokenizer)
        os.makedirs("./eval_results", exist_ok=True)
        with open("./eval_results/all.jsonl", "a", encoding="utf-8") as f:
            f.write(str({evaluator.output_path: metrics}) + "\n")


if __name__ == "__main__":
    device = "cuda:0"
    only_run = False
    if not only_run:
        perplexity_model_path = "/data/models/tofu_Llama-3.1-8B-Instruct_full"
        perplexity_model = AutoModelForCausalLM.from_pretrained(
            perplexity_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )
        perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_path, padding_side='left')
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token

    datasets = ["mmlu", "trivia_qa", "human_eval", "gsm8k"]
    n_samples = [0.1, 0.1, None, None]
    use_chat_template = True
    batch_size = 64

    base_output_path = "/data/ym/Unlearning_Token/eval_results"
    model_paths = []
    output_paths = []
    load_paths = []

    model_family = "llama3.1-8b"

    # model_paths.append("/data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu")
    # load_paths.append(
    #     "/data/ym/Unlearning_Token/edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu/AlphaEdit_400_batched_tofu.pth")

    for i in list(range(9)):
        num = (i+1)*400
        model_paths.append("/data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu")
        load_paths.append(f"/data/ym/Unlearning_Token/edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu/AlphaEdit_{num}_seq_tofu_{num}.pth")
        output_paths.append(f"{base_output_path}/edit/seq/{i}")

    # output_paths.append(f"{base_output_path}/edit/AlphaEdit_400_batched_tofu")
    # output_paths.append(f"{base_output_path}/base/UNL")

    # model_paths.append("/data/models/Llama-3.1-8B-Instruct-UL_real_world")
    # load_paths.append(
    #     "/data/ym/Unlearning_Token/edited_model/Llama-3.1-8B-Instruct-UL_real_world/AlphaEdit_400_batched_real_world_multi.pth")
    # output_paths.append("/data/ym/Unlearning_Token/eval_results/edit/AlphaEdit_400_batched_real_world_multi")

    print(len(model_paths))

    for load_path, output_path, model_path in zip(load_paths, output_paths, model_paths):
        print(output_path)
        for n_sample, dataset in zip(n_samples, datasets):
            print(f"---Evaluating {dataset}---")
            run(device, dataset, n_sample, use_chat_template, batch_size, model_path, output_path, only_run,
                load_path=load_path)
            time.sleep(5)
