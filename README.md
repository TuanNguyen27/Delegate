# Divide-and-Conquer
**Task-Difficulty Routing for Heterogeneous Language Model Agents**

---

## Overview
This project explores **divide-and-conquer strategies for agentic language systems**, where tasks are dynamically routed between **Small Language Models (SLMs)** and **Large Language Models (LLMs)** based on estimated difficulty.  

The central idea is simple:  
- **Easy subtasks** (e.g., surface-level retrieval, number extraction, short reasoning) → handled by **SLMs** (cheap, low-latency).  
- **Hard subtasks** (e.g., long-horizon reasoning, multi-hop synthesis, mathematical proofs) → escalated to **LLMs** (expensive, high-capacity).  

This work aims to reduce **compute cost** and **latency** while maintaining accuracy, aligning with ongoing research on adaptive routing and efficient agentic workflows.

---

## Motivation
Recent studies show that **increased test-time compute** often improves performance, but naive use of large models is prohibitively expensive. By leveraging **heterogeneous model collaboration**, we investigate:  
- How far can **small open-weight models** go when paired with smarter routing?  
- Can we approximate LLM-level performance while significantly lowering inference costs?  
- What are the trade-offs between **efficiency** and **accuracy** in agentic tasks?

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Divide-and-Conquer.git
   cd Divide-and-Conquer
Create a virtual environment and install dependencies:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

--- 

## Usage 
Run experiments via the config-first runner. For example, 
python src/app/run_router.py dataset=gsm8k

---

## Datasets
- **GSM8K**: Arithmetic and reasoning-heavy math word problems. 
- **SimpleQA**: Factual question answering with single-hop answers. 
- (Planned) **HotpotQA**: Multi-hop reasoning.

---

## License
This project is released under the MIT License. 