import os


class UtilityEvaluator:
    def __init__(self):
        self.dataset = ""
        self.hint_prompt = ""
        self.records = None

    def process(self, item):
        pass

    def equal(self, model_answer, correct_answer, item):
        pass

    def get_prompt(self, prompt):
        return self.hint_prompt.replace("{problem}", prompt)

    def save(self):
        with open(self.output_path, "w") as f:
            for record in self.records:
                f.write(str(record) + "\n")

    def set_output_path(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.output_path = f"{output_path}/{self.dataset}.jsonl"

    def append_record(self, item, model_answer):
        self.records.append({"data": item, "model_answer": model_answer})
