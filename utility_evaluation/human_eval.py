import os
import multiprocessing
from utility_evaluation import UtilityEvaluator

HINT_PROMPT = """Please complete the following python function.
Only output the function code with the required parameters, correct line breaks and indentation in the code (No use cases).
Your response must be in the following format without adding any other information:
Code: {your function code here}.

Function: {problem}
"""


class HumanEvalEvaluator(UtilityEvaluator):
    def __init__(self):
        super().__init__()
        self.dataset = "human_eval"
        self.hint_prompt = HINT_PROMPT
        self.max_new_tokens = 512
        self.output_path = None
        self.records = []
        self.sort_key = "prompt"

    def process(self, item):
        question = item['prompt']
        correct_answer = item['test']
        prompt = f"Function: {question}"
        return prompt, correct_answer

    def equal(self, model_answer, correct_answer, item):
        def extract_code(text):
            if "Code:" in text:
                return text.replace("Code:", "").strip()
            else:
                return None

        def run_code_in_process(code, exec_globals, item, error_queue):
            try:
                exec(code, exec_globals)
                exec_globals['check'](exec_globals[item['entry_point']])
            except AssertionError as e:
                error_queue.put(str(e))  # 发送错误到队列
            except Exception as e:
                error_queue.put(f"Unexpected error: {str(e)}")  # 捕获其他异常
            return None

        temp = os.environ.get("TOKENIZERS_PARALLELISM", "None")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            model_answer = extract_code(model_answer)
            if not model_answer:
                return False

            code = model_answer + "\n\n" + correct_answer
            exec_globals = {}
            error_queue = multiprocessing.Queue()

            p = multiprocessing.Process(
                target=run_code_in_process,
                args=(code, exec_globals, item, error_queue)
            )
            p.start()
            p.join(timeout=1)

            # 检查是否有错误
            if not error_queue.empty():
                return False

            if p.is_alive():
                p.terminate()
                return False
            return True
        except Exception:
            return False
        finally:
            os.environ["TOKENIZERS_PARALLELISM"] = temp

    def follow_instruction(self, model_answer):
        return "Code:" in model_answer
