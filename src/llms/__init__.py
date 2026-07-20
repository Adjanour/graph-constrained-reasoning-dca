from .base_language_model import BaseLanguageModel
from .base_hf_causal_model import HfCausalModel
from .graph_constrained_decoding_model import GraphConstrainedDecodingModel
from .llm_proxy import LLMProxy

def get_registed_model(model_name) -> BaseLanguageModel:
    from .chatgpt import ChatGPT
    registed_language_models = {
        'gpt': ChatGPT,
        'others': HfCausalModel,
        'gcr': GraphConstrainedDecodingModel,
        'proxy': LLMProxy,
    }
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    print("Model is not found in the registed_language_models, return HfCausalModel by default")
    return HfCausalModel