from tqdm import tqdm

from llmtuner import ChatModel


class Model:
    def __init__(self, checkpoint_dir, temperature=0.1, top_p=0.9, finetuning_type="lora"):
        args = {
            "model_name_or_path": "/hy-tmp/llama-2-7b-hf",
            "checkpoint_dir": checkpoint_dir,
            "template": "llama2",
            "finetuning_type": finetuning_type,
            "temperature": temperature,
            "top_p": top_p,
            # "repetition_penalty": 1.2
        }  
        self.chat_model = ChatModel(args=args)

    def generate(self, query, history=[]):
        res, _ = self.chat_model.chat(query, history)
        return res


