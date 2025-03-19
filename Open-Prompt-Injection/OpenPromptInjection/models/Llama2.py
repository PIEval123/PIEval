import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from .Model import Model


class Llama2(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        print("Using LLaMA2")
        if hasattr(self, 'path'):
            model_path = self.path
        else:
            model_path = self.name
        print(f'model_path:{model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

    def query(self, msg):
        
        messages = [
            {"role": "user", "content": msg},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(self.model.device)
        # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()
