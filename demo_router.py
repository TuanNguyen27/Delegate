#!/usr/bin/env python3
"""
Side-by-side comparison: Baseline LLM vs Hybrid LLM+SLM
"""
import asyncio
import time
import sys
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns

console = Console()

class LogCapture:
    """Capture tool calls"""
    def __init__(self):
        self.tool_calls = []
        self.current_call = {}
    
    def write(self, text):
        if '[TOOL CALLED]' in text:
            q = text.split(':', 1)[1].strip() if ':' in text else ''
            self.current_call = {'question': q}
        elif '[SLM RESULT] Extracted answer:' in text:
            ans = text.split(':', 1)[1].strip() if ':' in text else ''
            self.current_call['answer'] = ans
            self.tool_calls.append(self.current_call.copy())
    
    def flush(self):
        pass

async def run_baseline(problem: str):
    """Run baseline (LLM only)"""
    from agents import Agent, Runner, ModelSettings
    
    agent = Agent(
        name="Baseline Agent",
        instructions="Solve step-by-step. Final answer in \\boxed{}.",
        model="gpt-4o-mini",
        model_settings=ModelSettings(max_tokens=2048),
        tools=[]
    )
    
    start = time.time()
    result = await Runner.run(agent, problem, max_turns=10)
    elapsed = time.time() - start
    
    return result.final_output, elapsed, []

async def run_hybrid(problem: str):
    """Run hybrid (LLM + SLM)"""
    from router_agent import run_agent
    
    # Capture tool calls
    log_capture = LogCapture()
    old_stdout = sys.stdout
    sys.stdout = log_capture
    
    start = time.time()
    result = await run_agent(problem)
    elapsed = time.time() - start
    
    sys.stdout = old_stdout
    
    return result, elapsed, log_capture.tool_calls

async def compare(problem: str):
    """Run both and display side-by-side"""
    console.print(Panel(
        Text(problem, style="bold white"),
        title="Problem",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]Running experiments...[/bold yellow]\n")
    
    # Run both
    baseline_result, baseline_time, _ = await run_baseline(problem)
    hybrid_result, hybrid_time, tool_calls = await run_hybrid(problem)
    
    # Calculate speedup
    speedup = ((baseline_time - hybrid_time) / baseline_time * 100) if baseline_time > 0 else 0
    faster = "Hybrid" if hybrid_time < baseline_time else "Baseline"
    
    # Create comparison table
    table = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Baseline (LLM only)", style="yellow", width=30)
    table.add_column("Hybrid (LLM+SLM)", style="green", width=30)
    
    table.add_row(
        "Latency",
        f"{baseline_time:.2f}s",
        f"{hybrid_time:.2f}s"
    )
    
    table.add_row(
        "Tool Calls",
        "0",
        str(len(tool_calls))
    )
    
    table.add_row(
        "Speedup",
        "—",
        f"{abs(speedup):.1f}% {'faster' if speedup > 0 else 'slower'}"
    )
    
    console.print(table)
    
    # Show tool calls if any
    if tool_calls:
        console.print(f"\n[bold green]Tool Calls Made ({len(tool_calls)}):[/bold green]")
        for i, call in enumerate(tool_calls, 1):
            console.print(f"  {i}. {call.get('question', 'N/A')} → {call.get('answer', 'N/A')}")
    
    # Show answers side by side
    console.print("\n[bold]Final Answers:[/bold]\n")
    
    baseline_panel = Panel(
        Text(baseline_result[:200] + "..." if len(baseline_result) > 200 else baseline_result, style="yellow"),
        title="Baseline",
        border_style="yellow"
    )
    
    hybrid_panel = Panel(
        Text(hybrid_result[:200] + "..." if len(hybrid_result) > 200 else hybrid_result, style="green"),
        title="Hybrid",
        border_style="green"
    )
    
    console.print(Columns([baseline_panel, hybrid_panel]))
    
    # Verdict
    console.print(f"\n[bold]Verdict:[/bold] [bold {faster.lower()}]{faster} was faster![/bold {faster.lower()}]")
    
    return {
        'baseline_time': baseline_time,
        'hybrid_time': hybrid_time,
        'speedup': speedup,
        'tool_calls': len(tool_calls)
    }

async def main():
    """Main demo"""
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]Side-by-Side Comparison Demo[/bold cyan]\n\n"
            "Baseline (GPT-4o-mini alone) vs Hybrid (GPT-4o-mini + Qwen-Math-1.5B)"
        ),
        border_style="cyan"
    ))
    
    problem = "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value by 150%. How much profit did he make?"
    
    await compare(problem)
    
    # Offer to run another
    console.print("\n" + "="*80 + "\n")
    another = input("Run with a different problem? (y/n): ").strip().lower()
    
    if another == 'y':
        custom = input("Enter problem: ").strip()
        if custom:
            console.print("\n")
            await compare(custom)

if __name__ == "__main__":
    try:
        from rich.console import Console
    except ImportError:
        print("Install rich: pip install rich")
        sys.exit(1)
    
    asyncio.run(main())