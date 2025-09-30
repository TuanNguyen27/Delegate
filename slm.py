"""
SLM (Small Language Model) module for Qwen2.5-Math-1.5B
Handles local math model inference for calculations
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import warnings
warnings.filterwarnings('ignore')

# Global model instance (loaded once)
_model = None
_tokenizer = None
_device = None

def load_slm_model(model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", device=None):
    """
    Load the Qwen2.5-Math model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _device
    
    if _model is not None:
        return _model, _tokenizer, _device
    
    print(f"Loading SLM model: {model_name}")
    
    # Determine device
    if device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device
    
    print(f"Using device: {_device}")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model with appropriate settings
    if _device == "cuda":
        # Load with GPU optimizations
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for efficiency
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load on CPU
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        _model = _model.to(_device)
    
    _model.eval()  # Set to evaluation mode
    print(f"SLM model loaded successfully")
    
    return _model, _tokenizer, _device

def format_math_prompt(question: str) -> str:
    """
    Format a question for the Qwen2.5-Math model
    
    Args:
        question: The math question to solve
    
    Returns:
        Formatted prompt string
    """
    # Qwen2.5-Math uses a specific prompt format
    # Check if it's already a complete question
    if not question.endswith('?'):
        question = f"{question}?"
    
    # Use the instruction format that works best with Qwen2.5-Math
    prompt = f"""Please solve the following math problem step by step.
Show your work clearly and put your final answer in \\boxed{{answer}} format.

Question: {question}

Solution:"""
    
    return prompt

def get_slm_response(
    question: str, 
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
    do_sample: bool = True
) -> str:
    """
    Get a response from the SLM for a math question
    
    Args:
        question: The math question to solve
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling or greedy decoding
    
    Returns:
        The model's response with the solution and boxed answer
    """
    global _model, _tokenizer, _device
    
    # Load model if not already loaded
    if _model is None:
        load_slm_model()
    
    # Format the prompt
    prompt = format_math_prompt(question)
    
    # Tokenize input
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    # Move to device
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )
    
    # Decode response
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the solution part (remove the prompt)
    if "Solution:" in response:
        response = response.split("Solution:")[-1].strip()
    elif prompt in response:
        response = response.replace(prompt, "").strip()
    
    # Ensure we have a boxed answer format
    if "\\boxed{" not in response:
        # Try to extract the numerical answer and add boxed format
        # Look for common answer patterns
        answer = extract_answer_from_text(response)
        if answer:
            response = f"{response}\n\nTherefore, the answer is \\boxed{{{answer}}}."
    
    return response

def extract_answer_from_text(text: str) -> str:
    """
    Try to extract a numerical answer from text without boxed format
    
    Args:
        text: The text to extract from
    
    Returns:
        Extracted answer or empty string
    """
    # Try several patterns to find the answer
    patterns = [
        r"answer is[:\s]+([+-]?\d+\.?\d*)",
        r"equals?[:\s]+([+-]?\d+\.?\d*)",
        r"result is[:\s]+([+-]?\d+\.?\d*)",
        r"therefore[,\s]+([+-]?\d+\.?\d*)",
        r"so[,\s]+([+-]?\d+\.?\d*)",
        r"([+-]?\d+\.?\d*)[.\s]*$",  # Number at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""

def test_slm():
    """Test function to verify SLM is working"""
    test_questions = [
        "What is 25 + 37?",
        "Calculate 15 * 8",
        "What is 144 divided by 12?",
    ]
    
    print("Testing SLM module...")
    for q in test_questions:
        print(f"\nQuestion: {q}")
        response = get_slm_response(q)
        print(f"Response: {response}")
        
        # Extract boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            print(f"Extracted answer: {match.group(1)}")

# Alternative implementation if you don't want to use local model
def get_slm_response_mock(question: str) -> str:
    """
    Mock implementation for testing without loading actual model
    This is useful for debugging the routing logic
    """
    import random
    
    # Simple pattern matching for basic operations
    if "+" in question:
        # Try to extract numbers and add them
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            result = int(numbers[0]) + int(numbers[1])
            return f"Let me add these numbers: {numbers[0]} + {numbers[1]} = {result}.\n\\boxed{{{result}}}"
    
    elif "*" in question or "times" in question or "multiplied" in question:
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            result = int(numbers[0]) * int(numbers[1])
            return f"Let me multiply: {numbers[0]} × {numbers[1]} = {result}.\n\\boxed{{{result}}}"
    
    elif "-" in question or "minus" in question:
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            result = int(numbers[0]) - int(numbers[1])
            return f"Let me subtract: {numbers[0]} - {numbers[1]} = {result}.\n\\boxed{{{result}}}"
    
    elif "/" in question or "divided" in question:
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2 and int(numbers[1]) != 0:
            result = int(numbers[0]) / int(numbers[1])
            return f"Let me divide: {numbers[0]} ÷ {numbers[1]} = {result}.\n\\boxed{{{result}}}"
    
    # Default response for complex questions
    mock_answer = random.randint(1, 100)
    return f"After solving this problem step by step, the answer is \\boxed{{{mock_answer}}}"

# For testing purposes, you can switch between real and mock
USE_MOCK = False  # Set to True to use mock implementation

if USE_MOCK:
    print("WARNING: Using mock SLM implementation for testing")
    get_slm_response = get_slm_response_mock

if __name__ == "__main__":
    # Test the module when run directly
    test_slm()