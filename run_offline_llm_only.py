#!/usr/bin/env python3
"""
Run Offline LLM Baseline Experiment Only (Qwen 2.5 Math 7B)
This script runs the offline LLM baseline using Qwen 7B model locally
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.gsm8k_loader import load_gsm8k_as_df

async def main():
    parser = argparse.ArgumentParser(description='Run Offline LLM baseline experiment (Qwen 7B)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max output tokens (default: 512)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-7B-Instruct', 
                        help='Model ID from HuggingFace (default: Qwen/Qwen2.5-Math-7B-Instruct)')
    parser.add_argument('--input-csv', type=str, help='Use existing sample CSV file (optional)')
    parser.add_argument('--output', type=str, help='Output directory (default: auto-generated)')
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_short = args.model.split('/')[-1].replace('-', '_').lower()
        output_dir = Path(f"results_offline_llm_{model_short}_{args.samples}samples_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    if args.input_csv:
        print(f"\n📂 Loading samples from: {args.input_csv}")
        import pandas as pd
        test_df = pd.read_csv(args.input_csv)
        print(f"✅ Loaded {len(test_df)} samples from CSV")
    else:
        print(f"\n📂 Loading {args.samples} random samples from GSM8K...")
        gsm8k_df = load_gsm8k_as_df()
        test_df = gsm8k_df.sample(n=args.samples, random_state=args.seed).reset_index(drop=True)
        
        # Save samples for reproducibility
        sample_file = output_dir / "samples.csv"
        test_df.to_csv(sample_file, index=False)
        print(f"✅ Saved samples to {sample_file}")
    
    # Import and run experiment
    from experiments.offline_llm_experiment import run_offline_llm_experiment
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: OFFLINE LLM BASELINE ({args.model})")
    print("="*80)
    print(f"⚠️  Note: This will download and run the model locally (~14GB for 7B model)")
    print(f"⚠️  GPU recommended for acceptable performance")
    
    offline_llm_file = output_dir / "results_offline_llm.json"
    summary = await run_offline_llm_experiment(
        test_df.copy(), 
        str(offline_llm_file),
        max_tokens=args.max_tokens
    )
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"   - results_offline_llm.json")
    if not args.input_csv:
        print(f"   - samples.csv (use with --input-csv for other experiments)")
    
    print(f"\n📈 Summary:")
    print(f"   Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total']})")
    print(f"   Avg Latency: {summary['avg_latency']:.2f}s")
    print(f"   Avg Tokens: {summary['avg_input_tokens']:.0f} → {summary['avg_output_tokens']:.0f}")
    
    print(f"\n💡 To run other experiments on same samples:")
    print(f"   python run_llm_only.py --input-csv {output_dir}/samples.csv")
    print(f"   python run_router_only.py --input-csv {output_dir}/samples.csv")
    print(f"   python run_slm_only.py --input-csv {output_dir}/samples.csv")
    
    print(f"\n🔬 Compare with online LLM (Gemini):")
    print(f"   python run_llm_only.py --input-csv {output_dir}/samples.csv --output {output_dir.parent}/results_gemini_comparison")


if __name__ == "__main__":
    asyncio.run(main())

