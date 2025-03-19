import json
from models.attn_model import AttentionModel

def open_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'attn-hf':
        model = AttentionModel(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model