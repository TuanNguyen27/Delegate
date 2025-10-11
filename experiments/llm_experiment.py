# experiments/llm_experiment.py
"""
LLM Baseline: Gemini 2.5 Flash alone (with token tracking)
"""
import os
import time
import pandas as pd
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import check_answer, extract_answer

load_dotenv()

# Import API key manager
from tools.api_key_manager import create_key_manager

# Data class for results
from dataclasses import dataclass, asdict

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


async def run_llm_experiment(test_df: pd.DataFrame, output_file: str, max_tokens: int):
    """Run baseline with Gemini 2.5 Flash and track tokens"""
    
    print(f"Running Gemini 2.5 Flash on {len(test_df)} problems (max_tokens={max_tokens})")

    # Initialize API key manager
    key_manager = create_key_manager(cooldown_seconds=1)

    results = []
    total_input = 0
    total_output = 0
    total_latency = 0.0

    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{len(test_df)}] Processing...", end=' ')

        # Get model with next available API key
        model = key_manager.get_model(
            model_name="gemini-2.5-flash-lite",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0
            )
        )

        prompt = f"""You are an expert at solving math problems.
Solve this problem step by step.
Provide your final answer in \\boxed{{answer}} format.

Problem: {row['problem']}"""

        t_start = time.time()
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            t_end = time.time()
            
            # Check finish_reason before accessing text
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            
            # Handle different finish reasons
            if finish_reason == 2:  # MAX_TOKENS
                print(f"⚠️  Response incomplete (hit token limit)")
                prediction = "[INCOMPLETE - Hit max tokens]"
            elif finish_reason == 3:  # SAFETY
                print(f"⚠️  Response blocked by safety filter")
                prediction = "[BLOCKED - Safety]"
            elif finish_reason == 4:  # RECITATION
                print(f"⚠️  Response blocked by recitation filter")
                prediction = "[BLOCKED - Recitation]"
            else:
                # Try to get text safely
                try:
                    prediction = response.text if response.text else ""
                except Exception:
                    prediction = "[NO RESPONSE]"
            
            # Get token usage from metadata
            input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
            output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
            
        except Exception as e:
            print(f"ERROR: {e}")
            t_end = time.time()
            prediction = ""
            input_tokens = 0
            output_tokens = 0
        
        is_correct = check_answer(prediction, row["answer"])
        latency = t_end - t_start
        
        total_input += input_tokens
        total_output += output_tokens
        total_latency += latency

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
    accuracy = n_correct / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0
    
    summary = {
        'accuracy': accuracy,
        'correct': n_correct,
        'total': len(results),
        'avg_latency': avg_latency,
        'total_latency': total_latency,
        'avg_input_tokens': total_input / len(results) if results else 0,
        'avg_output_tokens': total_output / len(results) if results else 0,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output
    }

    # Save results
    import json
    output = {
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gemini 2.5 Flash Results: {n_correct}/{len(results)} = {accuracy:.2%}")
    print(f"Avg Latency: {avg_latency:.3f}s")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")

    return summary