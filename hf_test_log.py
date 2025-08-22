import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# 模型和tokenizer路径
model_path = "/home/zhenghaolin/local_data/hf_model/OpenPAI-SFT-20BA500M/hf_iter_19000"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 输入文本
input_text = '''<|endoftext|><|system|>You are an AI assistant developed by Microsoft. You are helpful for user to handle daily tasks. <|end|><|user|>Please reason step by step, and put your final answer within \\boxed{}.
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|end|><|assistant|>'''
print(input_text)

# # 编码输入
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 为了只记录“生成第一个token”时的前向，我们仅执行一次prefill前向，得到logits并从最后一个位置选出第一个生成token。
# 在这一次前向中，记录每个子模块的输入/输出形状与一个小切片，避免显存/内存爆炸。

# 日志输出目录
ts = time.strftime("%Y%m%d_%H%M%S")
log_dir = os.path.abspath(f"./hf_first_token_logs_{ts}")
os.makedirs(log_dir, exist_ok=True)
summary_path = os.path.join(log_dir, "module_io_summary.json")
slices_path = os.path.join(log_dir, "module_io_slices.pt")

# 采样通道数，避免保存整个大张量
SAMPLE_DIM = int(os.environ.get("HF_HOOK_SAMPLE_DIM", 128))

module_io = {}
hooks = []

def _to_small_slice(t: torch.Tensor):
    try:
        if not isinstance(t, torch.Tensor):
            return None
        x = t.detach()
        # 只取CPU、float32，便于保存
        if x.is_cuda:
            x = x.to("cpu")
        x = x.float()
        if x.ndim >= 3:
            # [B, T, C, ...] -> 取最后一个token和前SAMPLE_DIM通道
            cdim = x.shape[-1]
            c = min(SAMPLE_DIM, cdim)
            return x[:, -1, ..., :c]
        elif x.ndim == 2:
            cdim = x.shape[-1]
            c = min(SAMPLE_DIM, cdim)
            return x[:, :c]
        elif x.ndim == 1:
            c = min(SAMPLE_DIM, x.shape[0])
            return x[:c]
        else:
            return x
    except Exception:
        return None

def _shape_of(obj):
    if isinstance(obj, torch.Tensor):
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype).replace("torch.", ""),
            "device": str(obj.device),
        }
    return str(type(obj))

def _pack_items(items):
    out = []
    for it in items:
        if isinstance(it, torch.Tensor):
            out.append(_shape_of(it))
        elif isinstance(it, (list, tuple)):
            out.append([_shape_of(x) for x in it])
        else:
            out.append(_shape_of(it))
    return out

def register_hooks(model: torch.nn.Module):
    for name, m in model.named_modules():
        # 过滤掉模型本身的根模块，保留子模块
        if name == "":
            continue
        def make_hook(mod_name):
            def hook(_m, inputs, output):
                try:
                    entry = module_io.setdefault(mod_name, {})
                    # 只记录第一次触发（一次前向中每模块通常只触发一次）
                    if "input_shapes" not in entry:
                        # inputs 是 tuple
                        entry["input_shapes"] = _pack_items(list(inputs))
                        # 为常见张量输入保存一个小切片
                        in_tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
                        if len(in_tensors) > 0:
                            entry["input_slice"] = _to_small_slice(in_tensors[0])
                    if "output_shapes" not in entry:
                        if isinstance(output, tuple):
                            entry["output_shapes"] = _pack_items(list(output))
                            out_tensors = [x for x in output if isinstance(x, torch.Tensor)]
                            if len(out_tensors) > 0:
                                entry["output_slice"] = _to_small_slice(out_tensors[0])
                        else:
                            entry["output_shapes"] = _pack_items([output])
                            if isinstance(output, torch.Tensor):
                                entry["output_slice"] = _to_small_slice(output)
                except Exception:
                    pass
            return hook
        hooks.append(m.register_forward_hook(make_hook(name)))

register_hooks(model)

# Prefill：一次前向，得到第一个生成token
with torch.no_grad():
    out = model(**inputs, use_cache=True)
    logits = out.logits  # [B, T, V]
    next_token = torch.argmax(logits[:, -1, :], dim=-1)

# 清理hooks并保存结果
for h in hooks:
    try:
        h.remove()
    except Exception:
        pass

# 将切片从张量转换为可保存对象（torch.save可直接保存，但也可在JSON中仅存形状）
summary = {}
for k, v in module_io.items():
    summary[k] = {
        "input_shapes": v.get("input_shapes"),
        "output_shapes": v.get("output_shapes"),
        # 在JSON里仅记录是否有切片
        "has_input_slice": v.get("input_slice") is not None,
        "has_output_slice": v.get("output_slice") is not None,
    }

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 将实际的小切片张量单独保存
torch.save(module_io, slices_path)

# 解码生成的第一个新token与输出
first_token_text = tokenizer.decode(next_token, skip_special_tokens=False)
print(f"首个生成token：{first_token_text}")

# 可选：拼接形成完整输出（仅生成一个token）
output_ids = torch.cat([inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print("生成结果：", output_text)
print(f"模块输入/输出摘要已保存至：{summary_path}")
print(f"模块输入/输出切片已保存至：{slices_path}")