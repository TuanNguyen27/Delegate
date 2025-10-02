# Delegate 

> GPT-4o to reason and delegate easier sub-tasks to a Small Language Model (Qwen-2.5-math-instruct-1b) to save costs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**Problem Statement:** LLMs are dealt with many routine and easy tasks that do not fully utilize its capabilities, leading to wasteful token usage

**Our Solution:** A system where an LLM can reason and is equipped with the ability to ask (tool-call) an SLM for help on easier tasks

**Models:** 
- LLM: GPT-4o
- SLM: Qwen2.5-Math-1.5B-Instruct (specialized for math)

**Scope:** This implementation focuses specifically on **mathematical problem-solving** (GSM8K dataset). The routing architecture is domain-agnostic and can be adapted to other specialized tasks.

**Key Insight:** Let the big model decide *when* and *what* to delegate, while the small model executes 'easy' reasoning tasks + computations.

---

## 📊 Results Summary

Tested on GSM8K (grade school math problems, 500 samples):

| Method | Accuracy | Avg Latency | Cost/150 | Token Usage |
|--------|----------|-------------|----------|-------------|
| **GPT-4o Only** | 88% | 9.2s | $1.35 | 17,500 tokens |
| **Router (Ours)** | 81% | 13.5s | **$0.24** ✓ | 4,200 tokens ✓ |
| **Qwen Only** | 76% | 8.5s | Free | 14,800 tokens |

**Key Findings:**
- ✅ **82% cost reduction** ($1.35 → $0.24 per 150 problems)
- ✅ **76% fewer tokens** (17,500 → 4,200 tokens)
- ✅ Only 7% accuracy drop (88% → 81%)
- ⚠️ Trade: +47% latency (orchestration overhead)
- ⚡ Averages 1.0 tool calls per problem

---

## 🏗️ Work Flow
![Work Flow](media/workflow chart.png)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for Qwen2.5-Math-1.5B, but CPU works)
- OpenAI API key

**Note:** This implementation is specifically designed for **mathematical problem-solving** using Qwen2.5-Math-1.5B-Instruct. For other domains, you'll need to adapt the specialist model and tool definitions.

### Installation

```bash
# Clone the repository
git clone <your-repo>
cd llm-slm-router

# Install dependencies
pip install torch transformers openai pandas datasets python-dotenv openai-agents

# Set up environment
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Verify Setup

```bash
python check_setup.py
```

Should show all ✓ (green checks). The HuggingFace cache warning is normal - models download on first run.

---

## 🧪 Running Experiments

### Option 1: Full Comparison (Recommended)

Compare all three approaches on **identical samples**:

```bash
# Quick test (10 samples, ~5 minutes)
python run_comparison.py --samples 10 --seed 123

# Medium run (50 samples, ~25 minutes)
python run_comparison.py --samples 50 --seed 123

# Full benchmark (200 samples, ~100 minutes)
python run_comparison.py --samples 200 --seed 123
```

**Output:** Creates `results_comparison_Nsamples_TIMESTAMP/` folder with:
- `samples.csv` - The exact problems tested
- `results_llm.json` - GPT-4o baseline results
- `results_router.json` - Router system results
- `results_slm.json` - Qwen baseline results
- `comparison.json` - Summary statistics

### Option 2: Individual Experiments

Run experiments separately (useful for debugging):

```bash
# LLM baseline only
python llm_experiment_v2.py --dataset gsm8k --sample 50

# Router only
python router_experiment_v2.py --dataset gsm8k --sample 50

# SLM baseline only
python slm_experiment_v2.py --dataset gsm8k --sample 50
```

---

## 📁 Project Structure

```
.
├── README.md                      # This file
├── run_comparison.py              # Main orchestrator
├── llm_experiment_v2.py           # GPT-4o baseline
├── router_experiment_v2.py        # Router experiment runner
├── router_agent_v2.py             # Router agent definition
├── slm_experiment_v2.py           # Qwen baseline
├── gsm8k_loader.py                # Dataset loader
├── utils.py                       # Answer checking utilities
├── check_setup.py                 # Setup validator
├── analyze_results.py             # Results analyzer
├── format_for_presentation.py     # Presentation formatter
├── .env                           # API keys (create this)
└── results_comparison_*/          # Experiment outputs
    ├── samples.csv
    ├── results_llm.json
    ├── results_router.json
    ├── results_slm.json
    ├── comparison.json
    └── comparison_plots.png
```

---

## 🔬 What's Next?
- Evaluate our system on a mixed benchmark (contains both easy + difficult math questions)
- Optimize SLM for inference speed and see if it reduces latency in our framework
- Provide LLM with simple tool functions (e.g. calculator) to force it to only delegate harder questions to SLM
- Evaluate on other domains (e.g. coding)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o-mini API
- **Alibaba Qwen Team** for Qwen2.5-Math-1.5B-Instruct model
- **HuggingFace** for model hosting and transformers library
- **GSM8K** dataset creators (Cobbe et al., 2021)
- **OpenAI Agents** library for tool orchestration framework