import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-4-mini-reasoning"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="mps",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a math solver. Respond with only the final answer, no steps."},
    {"role": "user", "content": "Solve 56 + 39 = ?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128, # max tokens
        do_sample=False
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)