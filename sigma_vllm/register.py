from vllm import ModelRegistry
from vllm_model.sigma import SigmaForCausalLM
from vllm import LLM
import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

def register():
    ModelRegistry.register_model("SigmaForCausalLM", SigmaForCausalLM)

register()