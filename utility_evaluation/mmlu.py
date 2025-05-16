import os
import re
from utility_evaluation import UtilityEvaluator

HINT_PROMPT = """Please answer the following multiple-choice question by selecting the correct option (A, B, C or D). 
Your response must ends up with the following format:
The correct answer is {your answer option letter here}.

Question: {problem}
"""


class MMLUEvaluator(UtilityEvaluator):
    def __init__(self):
        super().__init__()
        self.dataset = "mmlu"
        self.hint_prompt = HINT_PROMPT
        self.max_new_tokens = 512
        self.output_path = None
        self.records = []
        self.sort_key = "question"

    def process(self, item):
        question = item['question']
        choices = [
            f"A. {item['choices'][0]}\n",
            f"B. {item['choices'][1]}\n",
            f"C. {item['choices'][2]}\n",
            f"D. {item['choices'][3]}\n"
        ]
        answer_list = ["A", "B", "C", "D"]
        correct_answer = answer_list[int(item['answer'])]
        prompt = f"Question: {question}\nOptions: "
        for choice in choices:
            prompt += f"{choice}"
        return prompt, correct_answer

    def equal(self, model_answer, correct_answer, item):
        matches = re.findall(r"The correct answer is [A-D]", model_answer)
        if len(matches) > 0:
            return matches[-1][-1] == correct_answer
        else:
            return False

    def follow_instruction(self, model_answer):
        return "The correct answer is" in model_answer