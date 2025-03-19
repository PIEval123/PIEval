from .Llama3 import Llama3
from .Llama2 import Llama2
from .Gemma import Gemma
from .Falcon import Falcon
from .GPT import GPT

def create_model(config):
    """
    Factory method to create a LLM instance
    """
    provider = config["model_info"]["provider"].lower()
    if provider == 'llama3':
        model = Llama3(config)
    elif provider == 'gemma':
        model = Gemma(config)
    elif provider == 'llama2':
        model = Llama2(config)
    elif provider == 'falcon':
        model = Falcon(config)
    elif provider == 'gpt':
        model = GPT(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model