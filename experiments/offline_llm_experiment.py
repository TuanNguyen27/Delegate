# experiments/offline_llm_experiment.py
"""
Offline LLM Baseline: Qwen 2.5 Math 7B (offline, no API)
Uses Unsloth for optimized inference (2x faster + 4-bit quantization)
"""
import time
import pandas as pd
import asyncio
import torch
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict

# Suppress HuggingFace Hub warnings
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import check_answer, extract_answer
from prompts import get_llm_baseline_prompt

# Import Unsloth for optimized inference
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✓ Unsloth available - using optimized inference (2x faster!)")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available - using standard transformers (slower)")
    print("   Install with: pip install unsloth")

@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    latency_total: float
    input_tokens: int = 0
    output_tokens: int = 0


class QwenAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-Math-7B-Instruct", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        print(f"Loading {model_id}...")
        
        if UNSLOTH_AVAILABLE:
            # Use Unsloth for optimized inference (2x faster + 4-bit quantization)
            # Convert to unsloth model name
            if not model_id.startswith("unsloth/"):
                unsloth_id = f"unsloth/{model_id.split('/')[-1]}"
                print(f"   Using Unsloth optimized: {unsloth_id}")
            else:
                unsloth_id = model_id
            
            # Load with Unsloth (includes tokenizer)
            max_seq_length = 2048
            dtype = None  # Auto detection
            load_in_4bit = True  # Use 4-bit quantization
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=unsloth_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            
            # Enable fast inference mode (2x speedup!)
            FastLanguageModel.for_inference(self.model)
            
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ Model loaded (Unsloth 4-bit, {vram:.1f}GB VRAM)")
            else:
                print("✓ Model loaded (Unsloth CPU mode)")
        
        else:
            # Fallback: Standard transformers (slower)
            # Load tokenizer with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    use_fast=True,
                    legacy=False
                )
                print("✓ Tokenizer loaded successfully")
            except Exception as e:
                error_msg = str(e)
                if "additional_chat_templates" in error_msg or "404" in error_msg:
                    print(f"⚠️  Chat templates not found (HuggingFace Hub issue)")
                    print("Retrying with fallback configuration...")
                else:
                    print(f"⚠️  Warning: {error_msg[:100]}...")
                    print("Retrying with fallback configuration...")
                
                # Fallback: try without fast tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id, 
                        trust_remote_code=True,
                        use_fast=False,
                        legacy=False
                    )
                    print("✓ Tokenizer loaded with fallback settings")
                except Exception as e2:
                    print(f"⚠️  Second attempt failed, trying minimal config...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id, 
                        trust_remote_code=True
                    )
                    print("✓ Tokenizer loaded with minimal config")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            print("Model ready (standard transformers)")

    async def run(self, prompt: str):
        """Run inference and return (response, input_tokens, output_tokens)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Count only new tokens
        output_tokens = outputs.shape[1] - input_tokens
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, input_tokens, output_tokens


async def run_offline_llm_experiment(test_df: pd.DataFrame, output_file: str, max_tokens: int):
    """Run Offline LLM baseline with token tracking"""
    agent = QwenAgent(max_new_tokens=max_tokens)

    print(f"Running Qwen 7B on {len(test_df)} problems (max_tokens={max_tokens})")

    results = []
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{len(test_df)}] Processing...", end=' ')

        # Prompt for offline LLM
        prompt = get_llm_baseline_prompt(row['problem'])

        t_start = time.time()
        prediction, input_tokens, output_tokens = await agent.run(prompt)
        t_end = time.time()

        is_correct = check_answer(prediction, row["answer"])
        latency = t_end - t_start
        
        total_latency += latency
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            question=row["problem"],
            ground_truth=str(row["answer"]),
            prediction=prediction,
            is_correct=is_correct,
            latency_total=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"{status} | {latency:.2f}s | {input_tokens}→{output_tokens} tokens")

    # Calculate summary
    n_correct = sum(r.is_correct for r in results)
    n_total = len(results)
    accuracy = n_correct / n_total if n_total else 0

    summary = {
        'accuracy': accuracy,
        'correct': n_correct,
        'total': n_total,
        'avg_latency': total_latency / n_total if n_total else 0,
        'total_latency': total_latency,
        'avg_input_tokens': total_input_tokens / n_total if n_total else 0,
        'avg_output_tokens': total_output_tokens / n_total if n_total else 0,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens
    }

    # Save results
    output = {
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Offline LLM Results: {n_correct}/{n_total} = {accuracy:.2%}")
    print(f"Avg Latency: {summary['avg_latency']:.3f}s")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")

    return summary