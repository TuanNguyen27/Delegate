"""
Router agent: GPT-4o-mini with SLM tool routing
"""
import re
import time
from agents import Agent, ModelSettings, function_tool
from slm import get_slm_response
from utils import MetricsTracker

# Get the global tracker from router_experiment
tracker = None

class ToolCallTracker:
    """Tracks tool calls to prevent duplicate calculations"""
    def __init__(self, metrics_tracker=None):
        self.call_history = {}
        self.metrics_tracker = metrics_tracker
    
    def create_slm_tool(self):
        @function_tool
        def slm_help(question: str) -> str:
            """Solve mathematical calculation. Returns definitive answer."""
            # Track timing if metrics tracker available
            if self.metrics_tracker:
                t_start = time.time()
            
            # Normalize the question for comparison
            normalized = question.strip().lower()
            
            # Check if we've already answered this exact question
            if normalized in self.call_history:
                previous_answer = self.call_history[normalized]
                # Log the duplicate call
                if self.metrics_tracker:
                    self.metrics_tracker.add_tool_call(
                        tool_name="slm_help",
                        input_text=question,
                        output_text=f"CACHED: {previous_answer}",
                        latency=0.0,
                        is_duplicate=True
                    )
                return f"ALREADY CALCULATED: {previous_answer} (using cached result - do not call again)"
            
            # Get new answer from SLM
            try:
                response = get_slm_response(question)
                
                # Extract the boxed answer for clearer response
                match = re.search(r'\\boxed\{([^}]+)\}', response)
                if match:
                    answer = match.group(1)
                    formatted_response = f"CALCULATION COMPLETE: The answer to '{question}' is {answer}."
                else:
                    # If no boxed answer, still mark as complete
                    formatted_response = f"CALCULATION COMPLETE: {response}"
                
                # Cache the formatted response
                self.call_history[normalized] = formatted_response
                
                # Track metrics if available
                if self.metrics_tracker:
                    t_end = time.time()
                    latency = t_end - t_start
                    self.metrics_tracker.add_tool_call(
                        tool_name="slm_help",
                        input_text=question,
                        output_text=formatted_response,
                        latency=latency,
                        is_duplicate=False
                    )
                
                return formatted_response
                
            except Exception as e:
                error_msg = f"CALCULATION ERROR: Unable to process '{question}'. Error: {str(e)}"
                if self.metrics_tracker:
                    t_end = time.time()
                    latency = t_end - t_start
                    self.metrics_tracker.add_tool_call(
                        tool_name="slm_help",
                        input_text=question,
                        output_text=error_msg,
                        latency=latency,
                        is_duplicate=False
                    )
                return error_msg
        
        return slm_help

# Strong, explicit instructions with examples
ROUTER_INSTRUCTIONS = """You are a math problem solver that uses a specialized calculation tool for ALL arithmetic operations.

CRITICAL RULES:
1. Use slm_help() for EVERY calculation, no matter how simple
2. NEVER perform mental math - always use the tool
3. Call the tool EXACTLY ONCE per unique calculation
4. The tool returns definitive answers - ALWAYS accept them
5. If you see "ALREADY CALCULATED", use that cached result immediately
6. If you see "CALCULATION COMPLETE", that's the final answer for that calculation

RESPONSE FORMAT:
- The tool will return one of:
  * "CALCULATION COMPLETE: The answer to 'X' is Y" - This is a new calculation
  * "ALREADY CALCULATED: ..." - This means you already asked this, use the cached answer
  * "CALCULATION ERROR: ..." - Something went wrong, try rephrasing

WORKFLOW:
1. Break down the problem into individual calculations
2. Call slm_help() for each calculation ONCE
3. Use the returned answers to continue
4. NEVER re-calculate something you've already asked

Example:
User: What is (520 + 650) * 2?
You: I'll solve this step by step.
[Call slm_help("What is 520 + 650?")]
Tool: "CALCULATION COMPLETE: The answer to '520 + 650' is 1170."
You: Now I'll multiply by 2.
[Call slm_help("What is 1170 * 2?")]
Tool: "CALCULATION COMPLETE: The answer to '1170 * 2' is 2340."
You: The final answer is 2340.

NEVER DO THIS:
[Call slm_help("What is 520 + 650?")]
[Call slm_help("What is 520 + 650?")] <- WRONG! Don't repeat calculations!

Remember: Trust the tool completely. One call per calculation. Move forward with the answer."""

def create_router_agent(metrics_tracker=None):
    """Create a router agent with loop prevention"""
    global tracker
    tracker = metrics_tracker
    
    # Create tool with state tracking
    tool_tracker = ToolCallTracker(metrics_tracker)
    slm_help_tool = tool_tracker.create_slm_tool()
    
    agent = Agent(
        instructions=ROUTER_INSTRUCTIONS,
        model="gpt-4o-mini",
        model_settings=ModelSettings(
            max_tokens=2048,
            temperature=0.1,  # Lower temperature for more consistent behavior
            parallel_tool_calls=False  # Prevent simultaneous calls
        ),
        tools=[slm_help_tool]
    )
    
    # Attach the tracker to the agent for access during runs
    agent.tool_tracker = tool_tracker
    
    return agent

# Create default agent instance
agent = create_router_agent()