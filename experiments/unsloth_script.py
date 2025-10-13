# ===== CELL 2: After Kernel Restart =====
from unsloth import FastLanguageModel
import torch

# Configuration
max_seq_length = 2048  # Can be set to anything
dtype = None  # None for auto detection. Float16 for Tesla T4/P100
load_in_4bit = True  # Use 4bit quantization

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Math-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...", # Add if needed
)

# Enable fast inference (2x faster!)
FastLanguageModel.for_inference(model)

print("✓ Model loaded successfully!")

# ===== CELL 3: Single Inference =====
def generate_response(prompt, max_new_tokens=512):
    """Generate a single response"""
    # Format for Qwen chat template
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    
    # Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

# Test single inference
prompt = "What is 15 + 27?"
response = generate_response(prompt)
print(f"Prompt: {prompt}")
print(f"Response: {response}")