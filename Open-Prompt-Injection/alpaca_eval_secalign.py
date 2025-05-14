# %%
import re
import os
import time
import numpy as np
import json

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config

# %%
model_config_path = "/path/to/llama3_secalign_config.json"
model_config = open_config(config_path=model_config_path)
model = PI.create_model(config=model_config)

alpaca_sampled = json.load(open("/prompt-injection-eval/alpacaeval/test.json"))
# %%
def prepare_query(instruction, input, defend):
    if defend:
        if input == "":
            # return f'[MARK] [INST][COLN]\n{instruction}\n\n[MARK] [INPT][COLN]\n\n\n[MARK] [RESP][COLN]\n'
            return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{instruction}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
            
        # return f'[MARK] [INST][COLN]\n{instruction}\n\n[MARK] [INPT][COLN]\n{input}\n\n[MARK] [RESP][COLN]\n'
        return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{instruction}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{input}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
    else:
        if input == "":
            return f'{instruction}\n'
        return f'{instruction}\n{input}\n'

results = []
for sample in alpaca_sampled:
    prompt = prepare_query(sample["instruction"], sample["input"], True)
    response = model.query(prompt)
    print(f"PROMPT: {prompt}")
    print(f"RESPONSE: {response}")
    print()
    
    result = {
        'instruction': sample["instruction"] + " " + sample["input"],
        'generator': 'SecAlign',
        'output': response,
    }
    results.append(result)
    print("Progress: ", len(results) / len(alpaca_sampled))

with open("alpacafarm_100_winrate_secalign.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)