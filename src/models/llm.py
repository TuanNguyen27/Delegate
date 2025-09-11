# src/models/llm.py

import os
from openai import OpenAI

_client = None

def load_llm():
    """Initialize OpenAI client (lazy)."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError()
        _client = OpenAI()  # picks up API key from env var
    return _client

import time

def llm_answer(question: str, cfg) -> str:
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