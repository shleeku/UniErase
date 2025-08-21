import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"
edit_path = f"edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share/AlphaEdit_test.pth"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.load_state_dict(torch.load(edit_path))
tokenizer = AutoTokenizer.from_pretrained(model_path)

unlearn_token = "<unlearn_0>"
unlearn_token_id = tokenizer.convert_tokens_to_ids(unlearn_token)
print("unlearn token id: ", unlearn_token_id)

# sample_question = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?" # The author's full name is Leila Al-Sabah. She is an LGBTQ+ author originally
# sample_question = "Can you describe Tae-ho Park's early life and how it influenced his career?<unlearn_0>" # Born in Seoul, South Korea, in 1968 to a family of an Obstetrician and
sample_question = "Can you describe Tae-ho Park's early life and how it influenced his career?"

inputs = tokenizer(sample_question, return_tensors="pt")
generated_answer = model.generate(**inputs)
print("generated answer: ", tokenizer.decode(generated_answer[0], skip_special_tokens=False))