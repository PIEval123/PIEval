import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from .Model import Model
from peft import PeftModel

class Llama3(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        
        if hasattr(self, 'path'):
            model_path = self.path
        else:
            model_path = self.name
        print(f'model_path:{model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).eval()
        # if "lora" in config["model_info"]:
        #     print("Loading LoRA")
        #     self.model = PeftModel.from_pretrained(self.model, config["model_info"]["lora"], is_trainable=False)

    def _tokenize_fn(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                # max_length=tokenizer.model_max_length,
                max_length=1024,
                truncation=True,
            )
            for text in strings
        ]
        
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] 
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )
    
    def query(self, msg):
        self.model.generation_config.max_new_tokens = 128
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 0.0
        
        input_ids = self._tokenize_fn([msg], self.tokenizer)['input_ids'][0].unsqueeze(0)
        outp = self.tokenizer.decode(
            self.model.generate(
                input_ids.to(self.model.device),
                attention_mask=torch.ones_like(input_ids).to(self.model.device),
                generation_config=self.model.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0][input_ids.shape[1]:]
        )
        outp = outp.replace("</s>.", "").replace("</s>:", "").replace("</s>", "").strip()
        return outp