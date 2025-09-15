# src/models/llm.py

import os
from openai import OpenAI
import time

_client = None

def load_llm():
    """Initialize OpenAI client (lazy)."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError()
        _client = OpenAI()  # picks up API key from env var
    return _client



def llm_answer(question: str, cfg) -> str:
    """
    Query an OpenAI LLM (e.g., GPT-4o-mini) for an answer to a given question.

    Args:
        question (str): The input question to send to the LLM.
        cfg (omegaconf.DictConfig): Configuration object containing:
            - cfg.llm.model_name (str): Model identifier, e.g. "gpt-4o-mini".
            - cfg.llm.system_prompt (str): Instruction for the system role.
            - cfg.llm.max_tokens (int): Maximum tokens to generate.
            - cfg.llm.temperature (float): Sampling temperature.

    Returns:
        str: The model's generated answer, stripped of leading/trailing whitespace.

    Raises:
        Exception: If the request fails after 3 retry attempts.
    """
    client = load_llm()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=cfg.llm.model_name,
                messages=[
                    {"role": "system", "content": cfg.llm.system_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=cfg.llm.max_tokens,
                temperature=cfg.llm.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retrying after error: {e}")
            time.sleep(2)