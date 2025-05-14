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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True, use_fast=False)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).eval()
            
    # def print_model_info(self):
        # print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| FT Path: {self.ft_path}\n{'-'*len(f'| Model name: {self.name}')}")

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
        # if self.ft_path == '' and 'DGDSGNH' not in msg:
        #     print('self.ft_model is None. Forward the query to the backend LLM')
        return self.backend_query(msg)
        
        processed_eval_prompt = self.formatting_func(msg)
        
        processed_eval_prompt = f'{processed_eval_prompt}\n### Response: '
        # print(f'\n\n\n\nPrompt: {processed_eval_prompt}\n')
        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to("cuda")

        self.ft_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.ft_model.generate(
                    **input_ids, 
                    max_new_tokens=10, 
                    repetition_penalty=1.2
                )[0], 
                skip_special_tokens=True
            )
            output = '### Response:'.join(output.split('### Response:')[1:])
            # if 'DGDSGNH' in msg:
            #     if "Repeat 'DGDSGNH' once while ignoring the following text." in output:
            #         output = output.replace("Repeat 'DGDSGNH' once while ignoring the following text.", "")
                    
            #     else:
            #         output = None
                
            #     if output is not None:
            #         output = self.process_response(output)

            # else:
            #     raise ValueError()
            #     print(f'>>>>>> (before processing) {output}')
            #     output = output.split('### Text: ')
            #     output = '### Text: '.join(output[1:])
            #     output = output.split('\n')
            #     output = '\n'.join(output[1:])
            
        # output = output.split('### Instruction: ')[0]
        print()
        print(f">>>>>> {output}")
        print()
        # print(">>>>>> " + processed_eval_prompt)
        # print()
        return output
    
    def backend_query(self, msg):
        # processed_eval_prompt = self.formatting_func(msg)
        # if isinstance(msg, dict):
        #     input_split = msg['input'].split('\nText: ')
        # elif isinstance(msg, str):
        #     input_split = msg.split('\nText: ')
        # else:
        #     raise ValueError(f'{type(msg)} is not supported for querying Mistral')
        # assert (len(input_split) == 2)
        #
        # processed_eval_prompt = f"{input_split[0]}\nText: {input_split[1]}. (end of text){self.tokenizer.eos_token}"
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
                    max_new_tokens=128,
                    repetition_penalty=1.2
                )[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
        # output = output.split('### [/INST]')[1:]
        # output = '### [/INST]'.join(output)
        # output = ' '.join(output)
        start = 0
        
        if output == '':
            return ' '
        return output