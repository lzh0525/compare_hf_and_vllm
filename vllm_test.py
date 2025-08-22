import argparse
import json
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from sigma_vllm.sigma import SigmaForCausalLM
ModelRegistry.register_model("SigmaForCausalLM", SigmaForCausalLM)

def main():
    
    model_path = "/home/zhenghaolin/local_data/hf_model/OpenPAI-SFT-20BA500M/hf_iter_19000"
    
    # args = parse_args()
    prompts = (
        """<|endoftext|><|system|>You are an AI assistant developed by Microsoft. "
        "You are helpful for user to handle daily tasks. <|end|><|user|>Please "
        "reason step by step, and put your final answer within \\boxed{}.\n"
        "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' "
        "market?<|end|><|assistant|>"""
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        top_k=1,
        max_tokens=128,
        n=1,
        stop='<|end|>',
        seed=42,
    )

    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        data_parallel_size=1,
        tensor_parallel_size=1,
        trust_remote_code=True,
        # gpu_memory_utilization=0.8
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    # print(ROUTER_LOGITS_GLOBAL)

    for out in outputs:
        for o in out.outputs:
            print(o.text.rstrip())

if __name__ == "__main__":
    main()
