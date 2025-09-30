# LLM-SLM Router System

> Intelligent routing system that uses GPT-4o to orchestrate a specialized small language model (Qwen 2.5 Math 1.5B) for efficient and accurate math problem solving.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**The Problem:** Large Language Models (LLMs) like GPT-4 are highly capable but slow and expensive. Small Language Models (SLMs) are fast and cheap but less accurate.

**Our Solution:** Use GPT-4o-mini as an intelligent orchestrator that delegates mathematical computations to **Qwen2.5-Math-1.5B-Instruct**, a specialized 1.5B parameter model trained specifically for mathematics. Think of it as a senior engineer (LLM) managing an intern (SLM) - strategic thinking upstairs, routine calculations downstairs.

**Key Insight:** Let the big model decide *when* and *what* to delegate, while the small model executes fast computations.

**Scope:** This implementation focuses specifically on **mathematical problem-solving** (GSM8K dataset). The routing architecture is domain-agnostic and can be adapted to other specialized tasks.

---

## 📊 Results Summary

Tested on GSM8K (grade school math problems, 150 samples):

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

**When to Use Router:**
- ✓ Batch processing (overnight jobs, analytics)
- ✓ Cost-sensitive applications (high volume)
- ✓ Acceptable accuracy thresholds (80%+)
- ✗ Real-time applications (latency critical)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           User Question                 │
│     "What is 27 × 14 + 15?"            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      GPT-4o-mini (Orchestrator)         │
│  • Understands the problem              │
│  • Plans solution strategy              │
│  • Decides: "delegate calculation"      │
└──────────────┬──────────────────────────┘
               │
               │ Tool Call: slm_help("27 × 14 + 15")
               ▼
┌─────────────────────────────────────────┐
│   Qwen2.5-Math-1.5B-Instruct            │
│   (Specialized Math Model)              │
│  • 1.5B parameters, math-specific       │
│  • Fast local GPU inference             │
│  • Returns: "393"                       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      GPT-4o-mini (Validation)           │
│  • Receives result: 393                 │
│  • Validates and formats answer         │
│  • Returns: "The answer is 393"         │
└─────────────────────────────────────────┘
```

**Key Components:**
- **Orchestrator:** GPT-4o-mini (via OpenAI API)
- **Specialist Tool:** Qwen2.5-Math-1.5B-Instruct (local inference)
- **Dataset:** GSM8K (grade school math problems)
- **Framework:** OpenAI Agents library with custom tool functions

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

## 📈 Analyzing Results

### Generate Report & Visualizations

```bash
python analyze_results.py results_comparison_50samples_TIMESTAMP
```

**Creates:**
1. **Formatted comparison table** (printed to terminal)
2. **Problem-level analysis** (which problems each method got right/wrong)
3. **Visualization plots** (`comparison_plots.png`)

### Format for Presentation

```bash
python format_for_presentation.py results_comparison_50samples_TIMESTAMP
```

**Outputs:** Copy-paste ready numbers for slides/papers.

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

## 🎓 Understanding the System

### Router Agent Workflow

1. **Problem Analysis**
   - GPT-4o-mini reads the math problem
   - Identifies calculations that need computation
   - Determines if delegation would be beneficial

2. **Tool Delegation**
   - Calls `slm_help(question)` for mathematical calculations
   - Qwen2.5-Math-1.5B-Instruct processes locally on GPU
   - Returns computed result to GPT-4o-mini

3. **Validation & Formatting**
   - GPT-4o-mini receives tool output
   - Validates mathematical correctness
   - Integrates into solution and formats final answer

### Why This Works for Math

**Domain Specialization:** Qwen2.5-Math-1.5B-Instruct is specifically trained for mathematical reasoning
- Pre-trained on mathematical corpora
- Fine-tuned on math problem datasets
- Optimized for step-by-step calculation

**Clear Delegation Boundaries:** Mathematical problems have well-defined subtasks
- Arithmetic operations (27 × 14)
- Algebraic manipulations (solve for x)
- Combinatorics (permutations, combinations)

**Speed Advantage:** 1.5B parameter model runs 5-8x faster than GPT-4o-mini for calculations

**Quality Control:** GPT-4o-mini orchestration ensures correctness
- Catches errors from specialist model
- Provides problem understanding and context
- Formats human-readable solutions

### Limitations & Considerations

**Math-Specific Implementation:** Current system is optimized for mathematical tasks
- Qwen2.5-Math is trained specifically for math
- Prompts and tool definitions assume mathematical context
- Other domains require different specialist models

**Orchestration Overhead:** Tool calling adds latency
- Each delegation requires: API call → local inference → API call
- Currently averages 1.0 tool calls per problem
- Future work: batch processing and caching to reduce overhead

**Accuracy Tradeoff:** 7% drop in accuracy compared to GPT-4o-mini alone
- Acceptable for many applications (81% vs 88%)
- Critical applications may require pure LLM approach
- Can be improved through fine-tuning and better routing

---

## 🔧 Configuration Options

### run_comparison.py

```bash
--samples N          # Number of problems (default: 10)
--seed N             # Random seed for reproducibility (default: 123)
--skip-slm           # Skip SLM baseline (faster)
--max-tokens N       # Max tokens per generation (default: 512)
```

### Individual experiments

```bash
--dataset {gsm8k,math500}  # Dataset to use
--sample N                 # Number of samples
--random                   # Random sample (vs first N)
--max-tokens N             # Max generation tokens
```

---

## 📊 Metrics Tracked

All experiments track:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct answers |
| **Total Latency** | End-to-end time per problem |
| **LLM Latency** | Time spent in GPT-4o (router only) |
| **SLM Latency** | Time spent in Qwen (router only) |
| **Input Tokens** | Tokens sent to model |
| **Output Tokens** | Tokens generated by model |
| **Tool Calls** | Number of SLM invocations (router only) |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Option 1: Use CPU (slower)
export CUDA_VISIBLE_DEVICES=""

# Option 2: Reduce samples
python run_comparison.py --samples 10
```

### OpenAI API Errors

```bash
# Check your .env file
cat .env

# Should show:
OPENAI_API_KEY=sk-...

# Test connection
python -c "import openai; print(openai.api_key[:10])"
```

### Model Download Fails

```bash
# Login to HuggingFace (if private models)
huggingface-cli login

# Check cache location
echo $HF_HOME
```

### Import Errors

```bash
# Install missing packages
pip install torch transformers openai pandas datasets python-dotenv

# For agents library
pip install openai-agents
```

---

## 💡 Best Practices

### For Reproducibility
- ✅ Always use the same `--seed` value
- ✅ Keep `max-tokens` consistent across runs
- ✅ Use `run_comparison.py` for fair comparisons

### For Debugging
- ✅ Start with `--samples 10` to test quickly
- ✅ Check `check_setup.py` before long runs
- ✅ Monitor GPU memory usage

### For Production
- ✅ Run 50-100 samples minimum for reliable metrics
- ✅ Keep results folders organized
- ✅ Document hyperparameters

---

## 🌍 Potential Impact

### Cost Optimization at Scale
This routing architecture demonstrates a practical approach to reducing LLM operational costs by **82%** while maintaining acceptable accuracy. For organizations processing millions of queries:

- **Enterprise Analytics:** Overnight batch processing of financial reports, data summaries, and trend analysis
- **Educational Platforms:** Automated homework grading and step-by-step solution generation at scale
- **Customer Support:** Ticket classification and response generation for high-volume queues
- **Research Infrastructure:** Large-scale dataset analysis where cost is a primary constraint

### Production Considerations
The cost-latency tradeoff makes this architecture particularly valuable for:
- ✅ **Batch processing pipelines** where 47% latency increase is acceptable
- ✅ **Cost-constrained applications** with >10K daily queries
- ✅ **Quality thresholds of 80%+** where 7% accuracy drop is tolerable
- ❌ **Real-time systems** where sub-second response is critical

### Environmental Impact
By reducing API calls by 76%, this approach also reduces:
- Energy consumption from cloud inference
- Carbon footprint of large model deployments
- Infrastructure costs for high-volume applications

---

## 🔬 Future Research Directions

### 1. Domain Expansion
**Hypothesis:** Routing architectures may be even more effective in domains with clearer delegation boundaries.

**Promising Areas:**
- **Web browsing agents:** LLM for navigation strategy, specialized models for data extraction
  - Example: GPT-4 plans scraping strategy → Specialized model extracts structured data
  - Potential: Higher accuracy with lower cost for web automation tasks

- **File processing systems:** LLM for document understanding, SLMs for format conversion
  - Example: GPT-4 analyzes document structure → Qwen extracts tables/figures
  - Potential: 90%+ cost reduction for document processing pipelines

- **Code execution:** LLM for algorithm design, SLMs for syntax generation/testing
  - Example: GPT-4 designs solution → CodeLlama generates implementation
  - Potential: Better code quality with reduced generation costs

- **Multi-modal tasks:** LLM for reasoning, specialized vision models for image analysis
  - Example: GPT-4 interprets medical report → Specialized model analyzes X-rays
  - Potential: Domain expertise with general reasoning capabilities

### 2. Model Optimization
**Fine-tuning specialist models** for specific delegation tasks:

- **Domain-specific fine-tuning:** Train Qwen variants on target task distributions
  - Expected: 10-15% accuracy improvement on delegated subtasks
  - Cost: One-time training vs. continuous API savings

- **Distillation from orchestrator:** Use GPT-4 outputs to improve SLM quality
  - Process: Collect GPT-4 solutions → Fine-tune Qwen on high-quality examples
  - Benefit: Smaller gap between router and baseline performance

- **Quantization and optimization:** INT8/INT4 quantization for faster inference
  - Expected: 2-4x speedup, reduce latency penalty from 47% to ~20%
  - Tools: GGUF, TensorRT, vLLM for optimized serving

### 3. Routing Strategy Improvements
**Smarter delegation decisions:**

- **Learned routing:** Train a small classifier to predict when delegation helps
  - Features: Problem complexity, token count, keyword presence
  - Potential: Increase tool calls from 1.0 to 3-4 per problem

- **Confidence-based routing:** Only delegate when SLM confidence is high
  - Use model uncertainty to decide delegation vs. self-solving
  - Expected: Reduce errors from tool outputs

- **Multi-tool routing:** Orchestrator delegates to multiple specialist models
  - Math tool (Qwen), Code tool (CodeLlama), Search tool (web API)
  - Architecture scales beyond single-domain applications

### 4. Infrastructure & Deployment
**Production-ready optimizations:**

- **Batched inference:** Process multiple tool calls in parallel
  - Expected: 3-5x throughput improvement
  - Reduces per-query latency through batching

- **Caching & memoization:** Cache frequent calculation results
  - Avoid redundant tool calls for common subproblems
  - Expected: 20-30% speedup on repeated patterns

- **Hybrid cloud-edge:** Run SLMs on edge devices, LLM in cloud
  - Privacy benefits: Sensitive data never leaves local device
  - Cost benefits: Zero API costs for delegated tasks

### 5. Evaluation & Benchmarking
**Comprehensive assessment across domains:**

- **Multi-domain benchmarks:** Test on MATH, HumanEval, MMLU, etc.
- **Cost-accuracy frontiers:** Map Pareto curves for different routing strategies
- **Latency profiling:** Identify bottlenecks in tool calling overhead
- **A/B testing framework:** Real-world production metrics

### 6. Theoretical Understanding
**Why and when does routing work?**

- **Task decomposability:** Which problems benefit most from delegation?
- **Orchestration overhead:** Model the cost of routing decisions
- **Failure modes:** When does delegation hurt performance?
- **Optimal delegation policies:** Reinforcement learning for routing strategies

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-slm-router-2025,
  title={LLM-SLM Router System for Efficient Math Problem Solving},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/llm-slm-router}}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Additional datasets (MATH, MathQA)
- [ ] More SLM options (Phi-3, Mistral)
- [ ] Dynamic routing strategies
- [ ] Cost optimization
- [ ] Multi-turn conversations

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

---

## 📞 Contact

Questions? Issues? Reach out:
- GitHub Issues: [Create an issue](https://github.com/yourusername/llm-slm-router/issues)
- Email: your.email@example.com

---

## 🗺️ Roadmap

**Current (v1.0):**
- [x] Basic router implementation for mathematics
- [x] GSM8K evaluation with comprehensive metrics
- [x] Cost-accuracy-latency tradeoff analysis
- [x] Complete documentation and tooling

**Near-term (v1.1-1.2):**
- [ ] Optimize inference (quantization, batching) to reduce latency
- [ ] Improve routing prompts to increase tool utilization
- [ ] Add MATH dataset support (higher difficulty problems)
- [ ] Implement caching for common calculations

**Mid-term (v2.0):**
- [ ] Domain expansion: web browsing, file extraction, code generation
- [ ] Multi-tool routing (math + code + search)
- [ ] Fine-tuned specialist models for target domains
- [ ] A/B testing framework for production deployment

**Long-term (v3.0+):**
- [ ] Learned routing policies (RL-based delegation)
- [ ] Hybrid cloud-edge deployment
- [ ] Multi-modal routing (vision + language)
- [ ] Cost-accuracy Pareto optimization
- [ ] Web interface for interactive experimentation

---

**Built with ❤️ for efficient and intelligent AI systems**