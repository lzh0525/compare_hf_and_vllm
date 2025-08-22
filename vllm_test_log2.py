import argparse
import json
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from sigma_vllm.sigma import SigmaForCausalLM
from vllm import LLM, SamplingParams
# from vllm.compilation.config import CompilationConfig, CompilationLevel
import os
import torch
ModelRegistry.register_model("SigmaForCausalLM", SigmaForCausalLM)


def main():
    # Ensure single-process V1 so we can access the underlying torch model
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model_path = \
        "/home/zhenghaolin/local_data/hf_model/OpenPAI-SFT-20BA500M/hf_iter_19000"

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
        max_tokens=1,
        n=1,
        stop='<|end|>',
        seed=42,
    )

    # Disable CUDA graphs/compilation path via enforce_eager in this vLLM version
    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        data_parallel_size=1,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
    )

    # Get underlying torch.nn.Module (valid for V1 single-process)
    model = None
    try:
        model = (llm.llm_engine.model_executor.driver_worker
                 .model_runner.model)
    except Exception:
        model = None

    io_cache = {}

    def to_meta(x):
        try:
            if isinstance(x, torch.Tensor):
                return {
                    'shape': tuple(x.shape),
                    'dtype': str(x.dtype),
                    'device': str(x.device),
                }
            if isinstance(x, (list, tuple)):
                return [to_meta(xx) for xx in x]
            if isinstance(x, dict):
                return {k: to_meta(v) for k, v in x.items()}
            return str(type(x))
        except Exception:
            return 'unavailable'

    def make_hook(name):
        def hook(mod, inputs, output):
            io_cache[(name, 'in')] = to_meta(inputs)
            io_cache[(name, 'out')] = to_meta(output)
        return hook

    hooks = []
    if model is not None:
        for name, module in model.named_modules():
            try:
                hooks.append(module.register_forward_hook(make_hook(name)))
            except Exception:
                pass

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for out in outputs:
        for o in out.outputs:
            print(o.text.rstrip())

    # Print a small subset of captured IO metadata
    printed = 0
    for (name, kind), meta in io_cache.items():
        print(f"[HOOK] {name} {kind}: {meta}")
        printed += 1
        if printed >= 50:
            break

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()