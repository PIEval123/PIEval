import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .Model import Model


class Llama3(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"]) # 128
        self.device = config["params"]["device"]
        
        if hasattr(self, 'path'):
            model_path = self.path
        else:
            model_path = self.name
        print(f'model_path:{model_path}')
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True, use_fast=False)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=self.bnb_config, torch_dtype=torch.float16, trust_remote_code=True).eval()
            
    # def print_model_info(self):
    #     print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| FT Path: {self.ft_path}\n{'-'*len(f'| Model name: {self.name}')}")

    def formatting_func(self, example):
        if isinstance(example, dict):
            input_split = example['input'].split('\nText: ')
        elif isinstance(example, str):
            input_split = example.split('\nText: ')
        else:
            raise ValueError(f'{type(example)} is not supported for querying Mistral')
        assert (len(input_split) == 2)
        if 'spam' in input_split[0]:
            input_split[1] = input_split[1].replace('#', '')
        text = f"### Instruction: {input_split[0]}\n### Text: {input_split[1]}"
        return text

    def process_response(self, input_string):
        marker = "### Response:"
        index = input_string.find(marker)
        if index != -1:
            return input_string[index:]
        else:
            return None

    def query(self, msg):
        return self.backend_query(msg)
        
    
    def backend_query(self, msg):
        if isinstance(msg, dict):
            processed_eval_prompt = msg['input']
        elif isinstance(msg, str):
            processed_eval_prompt = msg
        else:
            raise ValueError(f'{type(msg)} is not supported for querying Mistral')

        # processed_eval_prompt = f"{processed_eval_prompt} ### [/INST]{self.tokenizer.eos_token}"

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").input_ids.to("cuda")

        self.base_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.base_model.generate(
                    input_ids, 
                    attention_mask=torch.ones_like(input_ids).to("cuda"),
                    max_new_tokens=32,
                    repetition_penalty=1.2
                )[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
        start = 0
        
        if output == '':
            return ' '
        
        while output[start] == ' ': start += 1
        output = output[start:output.find(self.tokenizer.eos_token)]

        return output