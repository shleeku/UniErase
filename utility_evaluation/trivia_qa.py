import os
import re
from utility_evaluation import UtilityEvaluator

HINT_PROMPT = """"Please answer the following question based on the give context.
Extract the exact answer from the text while give a word of phrase as the answer.
Your response must ends up with the following format:
The correct answer is {your answer here}.

Question: {problem}
"""


class TriviaQaEvaluator(UtilityEvaluator):
    def __init__(self):
        super().__init__()
        self.dataset = "trivia_qa"
        self.hint_prompt = HINT_PROMPT
        self.max_new_tokens = 512
        self.output_path = None
        self.records = []
        self.sort_key = "question"

    def process(self, item):
        question = item['question']
        context = item['context']
        correct_answer = item['answers']['text'][0]
        if len(item['answers']['text']) > 1:
            print(item['answers']['text'])
        prompt = f"Context: {context}\n\nQuestion: {question}"
        return prompt, correct_answer

    def equal(self, model_answer, correct_answer, item):
        matches = re.findall(r"The correct answer is .*?\.", model_answer)
        if len(matches) > 0:
            model_answer = matches[-1].replace("The correct answer is ", "").strip()
        else:
            model_answer = model_answer
        model_answer = model_answer.lower()
        correct_answer = correct_answer.lower()
        if model_answer in correct_answer or correct_answer in model_answer:
            return True
        else:
            return False

    def follow_instruction(self, model_answer):
        return "The correct answer is" in model_answer
