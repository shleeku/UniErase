import json
import pickle
import torch


def save_ds(ds, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(ds, f)


def load_ds(load_path):
    with open(load_path, "rb") as f:
        ds = pickle.load(f)
    return ds


def load_jsonl(load_path, python_format=False):
    data = []
    with open(load_path, "r", encoding="utf-8") as f:
        for line in f:
            line = eval(line) if python_format else json.loads(line)
            data.append(line)
    return data


def save_jsonl(save_path, data, python_format=False):
    with open(save_path, "w", encoding="utf-8") as f:
        for item in data:
            item = str(item) if python_format else json.dumps(item)
            f.write(item)
            f.write("\n")


def local_generate(model, tokenizer, prompt, use_chat_template, skip_input=False, skip_special_token=False,
                   max_length=128, temperature=None):
    model.eval()
    if use_chat_template:
        messages = None
        if type(prompt) == str:
            messages = [{"role": "user", "content": prompt}]
        if type(prompt) == list:
            messages = prompt
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).int().to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=False if temperature is None else True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    if skip_input:
        generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=skip_special_token)
    else:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=skip_special_token)
    return generated_text


def local_generate_batch(model, tokenizer, prompts, use_chat_template, skip_input=False, skip_special_token=False,
                         max_new_tokens=128, temperature=None):
    model.eval()

    if use_chat_template:
        # Process chat templates in batch
        chat_texts = []
        for messages in prompts:
            chat_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # 不直接 tokenize，返回纯文本
            )
            chat_texts.append(chat_text)
        prompts = chat_texts

    # Process regular prompts in batch
    inputs = tokenizer(
        prompts,
        truncation=True,
        # max_length=128,
        # padding="max_length",
        padding="longest",
        return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature is None else True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # Process outputs
    generated_texts = []
    for i in range(len(outputs)):
        if skip_input:
            # Skip input part and only decode generated part
            generated_text = tokenizer.decode(outputs[i][len(input_ids[i]):], skip_special_tokens=skip_special_token)
        else:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=skip_special_token)
        generated_texts.append(generated_text)

    return generated_texts


def local_generate_unlearn(model, tokenizer, prompt, unlearn_seq, use_chat_template, skip_input=False,
                           skip_special_token=False, max_length=64):
    model.eval()
    if use_chat_template:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": unlearn_seq}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt").to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).int().to(model.device)
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
    else:
        inputs = tokenizer(prompt + unlearn_seq, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    if skip_input:
        generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=skip_special_token)
    else:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=skip_special_token)
    return generated_text


if __name__ == "__main__":
    pass
