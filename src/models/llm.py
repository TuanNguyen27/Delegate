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

def llm_answer(question: str, cfg) -> str:
    """Generate an answer using an OpenAI model defined in cfg."""
    client = load_llm()
    resp = client.chat.completions.create(
        model=cfg.llm.model_name,
        messages=[{"role": "user", "content": question}],
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
    )
    return resp.choices[0].message.content.strip()
