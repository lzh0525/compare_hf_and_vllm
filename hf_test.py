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
input_text = (
        """<|endoftext|><|system|>You are an AI assistant developed by Microsoft. "
        "You are helpful for user to handle daily tasks. <|end|><|user|>Please "
        "reason step by step, and put your final answer within \\boxed{}.\n"
        "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' "
        "market?<|end|><|assistant|>"""
    )
print(input_text)

# # 编码输入
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
# print("inputs:", inputs)
print(inputs['input_ids'].shape)
# exit(0)

# 生成结果
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=3,
        # do_sample=True,
        temperature=0.0,
        top_p=1,
        eos_token_id=tokenizer.eos_token_id
    )

# 解码输出
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print("生成结果：", output_text)