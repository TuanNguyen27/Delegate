import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/Phi-4-mini-reasoning"

# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model output 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map = "mps",
)

inputs = tokenizer("what is 35 + 64", return_tensors = "pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens = 128)

print(tokenizer.decode(outputs[0], skip_special_tokens="true"))