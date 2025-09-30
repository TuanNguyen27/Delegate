# test_simple_routing.py
import asyncio
from router_agent import run_agent

async def test_simple():
    questions = [
        "What is 5 + 5?",
        "Calculate 12 times 8",
        "What is 100 divided by 4?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print("="*60)
        result = await run_agent(q)
        print(f"\nFinal answer: {result}")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(test_simple())