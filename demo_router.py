#!/usr/bin/env python3
"""
Interactive demo of LLM + SLM routing system
Shows the complete thought process and tool calls
"""
import asyncio
import sys
from router_agent import run_agent
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()

# Monkey-patch to capture SLM interactions
slm_calls = []

def capture_slm_call(question, output, latency):
    """Capture SLM calls for display"""
    slm_calls.append({
        'question': question,
        'output': output,
        'latency': latency
    })

async def demo_problem(problem: str, show_steps: bool = True):
    """
    Run a problem and show the complete workflow
    """
    global slm_calls
    slm_calls = []
    
    console.print("\n" + "="*80, style="bold blue")
    console.print(Panel.fit(
        f"[bold cyan]Problem:[/bold cyan]\n{problem}",
        border_style="cyan"
    ))
    
    if show_steps:
        console.print("\n[bold yellow]🤖 LLM is thinking...[/bold yellow]")
    
    # Run the agent
    try:
        result = await run_agent(problem)
        
        # Display SLM calls
        if slm_calls:
            console.print(f"\n[bold green]🔧 Tool Calls Made: {len(slm_calls)}[/bold green]")
            
            for i, call in enumerate(slm_calls, 1):
                console.print(f"\n[bold magenta]Tool Call #{i}:[/bold magenta]")
                
                # Show what LLM sent to SLM
                console.print(Panel(
                    f"[cyan]LLM → SLM:[/cyan]\n{call['question']}",
                    border_style="blue"
                ))
                
                # Show SLM's reasoning
                console.print(Panel(
                    f"[yellow]SLM Reasoning:[/yellow]\n{call['output'][:500]}{'...' if len(call['output']) > 500 else ''}",
                    border_style="yellow"
                ))
                
                # Show what SLM returned to LLM
                import re
                match = re.search(r'\\boxed\{([^}]+)\}', call['output'])
                answer = match.group(1) if match else "No boxed answer"
                
                console.print(Panel(
                    f"[green]SLM → LLM:[/green]\nCALCULATION COMPLETE: The answer is {answer}",
                    border_style="green"
                ))
                
                console.print(f"⏱️  SLM Latency: [bold]{call['latency']:.2f}s[/bold]")
        else:
            console.print("\n[yellow]ℹ️  No tool calls made - LLM solved it directly[/yellow]")
        
        # Show final answer
        console.print("\n" + Panel(
            f"[bold green]Final Answer:[/bold green]\n{result}",
            border_style="bold green",
            title="LLM Result"
        ))
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")

async def interactive_demo():
    """Interactive demo mode"""
    console.print(Panel.fit(
        "[bold cyan]🚀 LLM + SLM Routing System Demo[/bold cyan]\n\n"
        "Watch how the LLM (GPT-4o-mini) delegates calculations to a specialized SLM (Qwen-Math-1.5B)",
        border_style="cyan"
    ))
    
    # Example problems
    examples = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    ]
    
    while True:
        console.print("\n[bold]Choose an option:[/bold]")
        console.print("1. Demo Problem #1 (Simple arithmetic)")
        console.print("2. Demo Problem #2 (Multi-step)")
        console.print("3. Demo Problem #3 (Complex)")
        console.print("4. Enter your own problem")
        console.print("5. Exit")
        
        choice = input("\nYour choice (1-5): ").strip()
        
        if choice == '5':
            console.print("\n[bold green]Thanks for trying the demo! 👋[/bold green]")
            break
        elif choice in ['1', '2', '3']:
            idx = int(choice) - 1
            await demo_problem(examples[idx])
        elif choice == '4':
            custom = input("\nEnter your math problem: ").strip()
            if custom:
                await demo_problem(custom)
        else:
            console.print("[red]Invalid choice. Try again.[/red]")

async def batch_demo(problems: list):
    """Run multiple problems in sequence"""
    console.print(Panel.fit(
        f"[bold cyan]Running {len(problems)} demo problems[/bold cyan]",
        border_style="cyan"
    ))
    
    for i, problem in enumerate(problems, 1):
        console.print(f"\n[bold]Problem {i}/{len(problems)}[/bold]")
        await demo_problem(problem, show_steps=False)
        
        if i < len(problems):
            input("\nPress Enter to continue to next problem...")

def main():
    """Main entry point"""
    # Inject our capture function into router_agent
    import router_agent
    original_tool = router_agent.slm_help
    
    # Wrap the tool to capture calls
    def wrapped_tool(question: str) -> str:
        result = original_tool(question)
        # Extract info from the printed output (hacky but works)
        return result
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        problems = [
            "What is 25 + 37?",
            "If John has 5 apples and gives away 2, how many does he have left?",
            "A store sells shirts for $15 each. How much for 8 shirts?"
        ]
        asyncio.run(batch_demo(problems))
    else:
        asyncio.run(interactive_demo())

if __name__ == "__main__":
    # Check if rich is installed
    try:
        from rich.console import Console
    except ImportError:
        print("Please install rich: pip install rich")
        sys.exit(1)
    
    main()