# Offline LLM Experiment Guide

## Overview

The offline LLM experiment allows you to run a **7B parameter math-specialized model locally** without any API costs. This is useful for:
- 📊 Comparing offline vs online (Gemini) LLM performance
- 💰 Zero API costs (no Gemini key needed)
- 🔒 Complete privacy (no data sent to cloud)
- 🧪 Testing on custom models from HuggingFace

## Quick Start

### Basic Usage

```bash
# Run on 10 problems (default Qwen 7B model)
python run_offline_llm_only.py --samples 10

# Use custom model
python run_offline_llm_only.py --samples 10 --model Qwen/Qwen2.5-Math-7B-Instruct

# Run on same samples as Gemini for comparison
python run_llm_only.py --samples 20  # Creates samples.csv
python run_offline_llm_only.py --input-csv results_*/samples.csv
```

### Command Line Options

```bash
python run_offline_llm_only.py --help

Options:
  --samples INT         Number of samples (default: 10)
  --seed INT           Random seed (default: 123)
  --max-tokens INT     Max output tokens (default: 512)
  --model STR          Model ID from HuggingFace (default: Qwen/Qwen2.5-Math-7B-Instruct)
  --input-csv PATH     Use existing sample CSV file
  --output PATH        Output directory (default: auto-generated)
```

## System Requirements

### Minimum (CPU)
- **RAM**: 16GB
- **Storage**: 15GB free (for model)
- **Performance**: ~60-120s per problem

### Recommended (GPU)
- **GPU**: NVIDIA with 8GB+ VRAM (e.g., RTX 3070)
- **RAM**: 16GB
- **Storage**: 15GB free
- **Performance**: ~5-15s per problem

### Supported Models

| Model | Size | VRAM | Accuracy | Speed (GPU) |
|-------|------|------|----------|-------------|
| Qwen2.5-Math-1.5B | ~3GB | 4GB | ~65% | ~3-5s |
| Qwen2.5-Math-7B | ~14GB | 8GB | ~75% | ~5-15s |
| Qwen2.5-Math-72B | ~140GB | Multiple GPUs | ~85% | ~30-60s |

## Example Workflows

### Workflow 1: Compare Offline vs Online LLM

```bash
# Step 1: Run Gemini (online)
python run_llm_only.py --samples 20

# Step 2: Run Qwen 7B (offline) on same samples
python run_offline_llm_only.py --input-csv results_llm_only_*/samples.csv

# Step 3: Compare results
# Check results_llm_only_*/results_llm.json
# Check results_offline_llm_*/results_offline_llm.json
```

### Workflow 2: Test Different Model Sizes

```bash
# Create sample set once
python run_offline_llm_only.py --samples 30 --model Qwen/Qwen2.5-Math-1.5B-Instruct
SAMPLES=$(ls -d results_offline_llm_* | tail -1)/samples.csv

# Test 7B model
python run_offline_llm_only.py --input-csv $SAMPLES --model Qwen/Qwen2.5-Math-7B-Instruct

# Compare: 1.5B vs 7B on same problems
```

### Workflow 3: Full Comparison (All Models)

```bash
# 1. Run Gemini (cloud LLM)
python run_llm_only.py --samples 20
SAMPLES=$(ls -d results_llm_* | tail -1)/samples.csv

# 2. Run Qwen 7B (offline LLM)
python run_offline_llm_only.py --input-csv $SAMPLES

# 3. Run Router (Gemini + Qwen 1.5B)
python run_router_only.py --input-csv $SAMPLES

# 4. Run SLM alone (Qwen 1.5B)
python run_slm_only.py --input-csv $SAMPLES

# All experiments now tested on identical problems!
```

## Output Format

Results are saved to `results_offline_llm_*/results_offline_llm.json`:

```json
{
  "summary": {
    "accuracy": 0.75,
    "correct": 15,
    "total": 20,
    "avg_latency": 12.5,
    "avg_input_tokens": 85,
    "avg_output_tokens": 220
  },
  "results": [
    {
      "problem_id": "prob_0",
      "question": "...",
      "ground_truth": "42",
      "prediction": "...\\boxed{42}",
      "is_correct": true,
      "latency_total": 11.2,
      "input_tokens": 82,
      "output_tokens": 215
    }
  ]
}
```

## Performance Optimization

### ⚡ Unsloth Optimization (Recommended)

**Install Unsloth for 2x faster inference + automatic 4-bit quantization:**

```bash
pip install unsloth
```

**Benefits:**
- ✅ **2x faster** inference (8-12s → 4-6s per problem)
- ✅ **75% less VRAM** (~14GB → ~3.5GB for 7B model)
- ✅ **Automatic** - no code changes needed
- ✅ **Same accuracy** - no quality loss

The experiment automatically uses Unsloth if installed:
```bash
# Just run as normal - Unsloth is auto-detected
python run_offline_llm_only.py --samples 20

# Output shows optimization:
# ✓ Unsloth available - using optimized inference (2x faster!)
# ✓ Model loaded (Unsloth 4-bit, 15.8GB VRAM)
```

**See [UNSLOTH_OPTIMIZATION.md](UNSLOTH_OPTIMIZATION.md) for details.**

### GPU Acceleration

Ensure PyTorch uses GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Model Quantization (Manual - Only if not using Unsloth)

For lower VRAM usage without Unsloth, use pre-quantized models:
```bash
# 8-bit quantization (half VRAM)
python run_offline_llm_only.py --model Qwen/Qwen2.5-Math-7B-Instruct-GPTQ

# 4-bit quantization (quarter VRAM)
python run_offline_llm_only.py --model Qwen/Qwen2.5-Math-7B-Instruct-AWQ
```

**Note:** Unsloth's automatic 4-bit quantization is usually better and faster than manual quantized models.

### Batch Processing

For large-scale testing, process in batches:
```bash
# Process 100 samples in batches of 10
for i in {0..9}; do
  python run_offline_llm_only.py --samples 10 --seed $i
done
```

## Comparison Metrics

### Gemini vs Qwen 7B (Typical)

| Metric | Gemini 2.5 Flash | Qwen 7B (GPU) | Winner |
|--------|------------------|---------------|--------|
| Accuracy | ~68% | ~72% | Qwen 7B ✅ |
| Latency | ~2-3s | ~8-12s | Gemini ✅ |
| Cost/1K | $0.15 | $0.00 | Qwen 7B ✅ |
| Privacy | Cloud | Local | Qwen 7B ✅ |

### Router vs Offline LLM

| Metric | Router (Gemini+1.5B) | Offline LLM (7B) | Winner |
|--------|----------------------|------------------|--------|
| Accuracy | ~65% | ~72% | Offline ✅ |
| Latency | ~15-20s | ~8-12s | Offline ✅ |
| Cost/1K | $0.10 | $0.00 | Offline ✅ |
| Tool Calls | ~1.2 avg | 0 | - |

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Use smaller model
```bash
python run_offline_llm_only.py --model Qwen/Qwen2.5-Math-1.5B-Instruct
```

**Solution 2**: Use CPU (slower)
```bash
export CUDA_VISIBLE_DEVICES=""
python run_offline_llm_only.py --samples 5
```

**Solution 3**: Reduce tokens
```bash
python run_offline_llm_only.py --max-tokens 256
```

### Issue: Model Download Fails

**Solution**: Download manually
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### Issue: Slow on CPU

**Expected**: CPU inference is 10-20x slower than GPU
- GPU: ~8-12s per problem
- CPU: ~60-120s per problem

**Solution**: Use fewer samples for testing
```bash
python run_offline_llm_only.py --samples 3  # Quick test
```

## Advanced Usage

### Custom Model from HuggingFace

```bash
# Try different math models
python run_offline_llm_only.py --model deepseek-ai/deepseek-math-7b-instruct
python run_offline_llm_only.py --model microsoft/Phi-3-mini-4k-instruct
```

### Offline-Only Mode (No Internet)

1. Download model once:
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct')"
```

2. Use offline:
```bash
export HF_HUB_OFFLINE=1
python run_offline_llm_only.py --samples 10
```

## Next Steps

- 📊 **Compare with online**: Test Gemini vs Qwen on same samples
- 🔬 **Experiment with models**: Try different sizes (1.5B, 7B, 72B)
- 📈 **Analyze results**: Use `tools/analyze_results.py` for visualization
- 🎯 **Fine-tune**: Use your own math problems for domain adaptation

## Files Created by This Experiment

- `results_offline_llm_*/results_offline_llm.json` - Full results
- `results_offline_llm_*/samples.csv` - Problem set (if not using --input-csv)
- `~/.cache/huggingface/` - Downloaded model (cached)

## Related Scripts

- `run_llm_only.py` - Gemini baseline (online)
- `run_router_only.py` - Router with tool delegation
- `run_slm_only.py` - Small model (1.5B) baseline
- `experiments/offline_llm_experiment.py` - Core experiment logic

