# demo.py
"""
Interactive demo of LLM + SLM routing system
Captures and displays the complete thought process
"""
import asyncio
import sys
import re
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Global to capture output
captured_logs = []

class LogCapture:
    """Capture print statements from router_agent"""
    def __init__(self):
        self.logs = []
        self.current_call = None
    
    def write(self, text):
        if text.strip():
            self.logs.append(text)
            
            # Parse for tool calls
            if '[TOOL CALLED]' in text:
                question = text.split(':', 1)[1].strip() if ':' in text else text
                self.current_call = {'question': question, 'output': '', 'answer': ''}
            
            elif '[SLM OUTPUT] Full response:' in text:
                if self.current_call:
                    self.current_call['output'] = text.split('Full response:\n', 1)[1].strip() if 'Full response:\n' in text else ''
            
            elif '[SLM RESULT] Extracted answer:' in text:
                if self.current_call:
                    self.current_call['answer'] = text.split('answer:', 1)[1].strip() if 'answer:' in text else ''
                    captured_logs.append(self.current_call.copy())
                    self.current_call = None
    
    def flush(self):
        pass

async def demo_problem(problem: str):
    """Run a problem and show the complete workflow"""
    global captured_logs
    captured_logs = []
    
    console.print("\n" + "="*80, style="bold blue")
    console.print(Panel(
        Text(problem, style="bold cyan"),
        title="[bold]Problem[/bold]",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]🤖 LLM is analyzing the problem...[/bold yellow]\n")
    
    # Capture output while running
    log_capture = LogCapture()
    
    # Import here to capture prints
    from router_agent_demo import run_agent
    
    # Redirect prints temporarily
    old_stdout = sys.stdout
    sys.stdout = log_capture
    
    try:
        result = await run_agent(problem)
        sys.stdout = old_stdout
        
        # Display captured tool calls
        if captured_logs:
            console.print(f"[bold green]🔧 Tool Calls: {len(captured_logs)}[/bold green]\n")
            
            for i, call in enumerate(captured_logs, 1):
                console.print(f"[bold magenta]═══ Tool Call #{i} ═══[/bold magenta]\n")
                
                # LLM → SLM
                console.print(Panel(
                    Text(call['question'], style="cyan"),
                    title="[cyan]LLM asks SLM[/cyan]",
                    border_style="blue"
                ))
                
                # SLM reasoning (truncated)
                reasoning = call['output'][:400] + '...' if len(call['output']) > 400 else call['output']
                console.print(Panel(
                    Text(reasoning, style="yellow"),
                    title="[yellow]SLM thinks[/yellow]",
                    border_style="yellow"
                ))
                
                # SLM → LLM
                console.print(Panel(
                    Text(f"Answer: {call['answer']}", style="green bold"),
                    title="[green]SLM returns to LLM[/green]",
                    border_style="green"
                ))
                console.print()
        else:
            console.print("[yellow]ℹ️  No tool calls - LLM solved directly[/yellow]\n")
        
        # Final answer
        console.print(Panel(
            Text(result, style="bold green"),
            title="[bold green]Final Answer[/bold green]",
            border_style="bold green"
        ))
        
    except Exception as e:
        sys.stdout = old_stdout
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}", style="bold red")

async def interactive_demo():
    """Interactive demo mode"""
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]🚀 LLM + SLM Routing Demo[/bold cyan]\n\n"
            "See how GPT-4o-mini delegates calculations to Qwen-Math-1.5B"
        ),
        border_style="cyan"
    ))
    
    examples = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value by 150%. How much profit did he make?",
    ]
    
    while True:
        console.print("\n[bold]Choose:[/bold]")
        console.print("1. Simple arithmetic problem")
        console.print("2. Multi-step calculation")
        console.print("3. Complex word problem")
        console.print("4. Your own problem")
        console.print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '5':
            console.print("\n[bold green]Thanks! 👋[/bold green]")
            break
        elif choice in ['1', '2', '3']:
            await demo_problem(examples[int(choice) - 1])
        elif choice == '4':
            custom = input("\nEnter problem: ").strip()
            if custom:
                await demo_problem(custom)
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        from rich.console import Console
    except ImportError:
        print("Install rich: pip install rich")
        sys.exit(1)
    
    asyncio.run(interactive_demo())