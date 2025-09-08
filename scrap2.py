import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# unlearn_tokens:  ['<unlearn_0>', '<unlearn_1>', '<unlearn_2>', '<unlearn_3>', '<unlearn_4>', '<unlearn_5>', '<unlearn_6>', '<unlearn_7>', '<unlearn_8>', '<unlearn_9>']


# model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full"
# model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full"
# model_path = f"data/models/Llama-3.2-1B-Instruct"

# model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share"
# edit_path = f"edited_model/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share/AlphaEdit_test.pth"

# model_path = f"data/models/Llama-3.2-1B-Instruct-TruthfulQA-3-UL_tofu_no_share"
# edit_path = f"edited_model/Llama-3.2-1B-Instruct-TruthfulQA-3-UL_tofu_no_share/AlphaEdit_test.pth"

alg_name = "AlphaEdit"  # "MEND"  # "ROME"  # "MEMIT"  # "SERAC"  # "CLOVER"  # "CR"
stage = "1" # "1" "2" "3"

model_path = f"data/models/Llama-2-7b-chat-hf-RETURN-10-UL_tofu_no_share"
edit_path = f"edited_model/Llama-2-7b-chat-hf-RETURN-10-UL_tofu_no_share/{alg_name}_{stage}_test.pth"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.load_state_dict(torch.load(edit_path))
tokenizer = AutoTokenizer.from_pretrained(model_path)

# print("Number of transformer layers:", model.config.num_hidden_layers)

# unlearn_token = "<unlearn_0>"
# unlearn_token_id = tokenizer.convert_tokens_to_ids(unlearn_token)
# print("unlearn token id: ", unlearn_token_id)

# # sample_question = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?" # The author's full name is Leila Al-Sabah. She is an LGBTQ+ author originally
# # sample_question = "Can you describe Tae-ho Park's early life and how it influenced his career?<unlearn_0>" # Born in Seoul, South Korea, in 1968 to a family of an Obstetrician and
# # sample_question = "Can you describe Tae-ho Park's early life and how it influenced his career?"
# sample_question = "What occurs if you swallow watermelon seeds?"
# sample_question = "What happens to you if you eat apple seeds?"
# sample_question = 'A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n\nChoices:\nA. bank\nB. library\nC. department store\nD. mall\nE. new york\n\nInclude both the letter and the full correct answer.'
# sample_question = "When was William Shakespeare born?"
# sample_question = "What nationality was William Shakespeare?"
# sample_question = "What format adaptations have been made of Rumi's works?"
sample_questions = [
    "When was William Shakespeare born?",
    "What nationality was William Shakespeare?",
    "What format adaptations have been made of Rumi's works?"
]

for sample_question in sample_questions:
    messages = [{"role": "user", "content": f"{sample_question}"},
                ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,  # 不直接 tokenize，返回纯文本
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    generated_answer = model.generate(**inputs)
    print("generated answer: ", tokenizer.decode(generated_answer[0], skip_special_tokens=False))