import os
import re
from utility_evaluation import UtilityEvaluator

HINT_PROMPT = """Solve the following math problem.
Your response must ends up with the following format:
The final answer is {your final answer number here}.

Problem: {problem}
"""


class GSM8KEvaluator(UtilityEvaluator):
    def __init__(self):
        super().__init__()
        self.dataset = "gsm8k"
        self.hint_prompt = HINT_PROMPT
        self.max_new_tokens = 512
        self.output_path = None
        self.records = []
        self.sort_key = "question"

    def process(self, item):
        question = item['question']
        correct_answer = item['answer'].split("#### ")[-1]
        prompt = f"Problem: {question}"
        return prompt, correct_answer

    def equal(self, model_answer, correct_answer, item):
        def extract_last_number(text):
            # matches = re.findall(r'The final answer is \d+', text.replace('$', '').replace(',', ''))
            # if len(matches) > 0:
            #     return matches[-1].replace('The final answer is ', '').strip()
            matches = re.findall(r'\d+', text.replace('$', '').replace(',', ''))
            if len(matches) > 0:
                return re.sub(r'\D', '', matches[-1].strip())
            else:
                return None

        try:
            model_answer = extract_last_number(model_answer.split("The final")[-1])
            model_answer = int(model_answer)
            correct_answer = int(correct_answer)
            return model_answer == correct_answer
        except:
            return False

    def follow_instruction(self, model_answer):
        return "The final answer is" in model_answer
