#!/usr/bin/env python3
"""
Diagnostic script to verify LLM receives SLM responses properly.
Analyzes results_router.json to check for issues in the SLM→LLM flow.
"""
import json
import sys

def analyze_results(results_file='results_router.json'):
    """Analyze router results to diagnose LLM-SLM communication."""
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {results_file}")
        print("Run: python run_router_only.py --samples 5")
        return
    
    results = data.get('results', [])
    if not results:
        print("❌ No results found in file")
        return
    
    print("="*80)
    print("DIAGNOSTIC: LLM ← SLM Communication Flow")
    print("="*80)
    
    total_problems = len(results)
    empty_responses = 0
    slm_called = 0
    llm_responded_after_slm = 0
    
    for idx, r in enumerate(results):
        problem_id = r.get('problem_id', f'prob_{idx}')
        prediction = r.get('prediction', '')
        llm_conv = r.get('llm_conversation', [])
        slm_calls = r.get('slm_calls', [])
        
        # Check if this problem had empty response
        is_empty = (not prediction or prediction == "No response generated" or 
                   "No response" in prediction)
        
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}/{total_problems}: {problem_id}")
        print(f"Question: {r.get('question', '')[:60]}...")
        print(f"{'='*80}")
        
        # Check SLM calls
        if slm_calls:
            slm_called += 1
            print(f"\n✅ SLM was called: {len(slm_calls)} time(s)")
            for i, call in enumerate(slm_calls):
                slm_input = call.get('input', '')
                slm_output = call.get('output', '')
                print(f"\n  SLM Call #{i+1}:")
                print(f"    Input: {slm_input[:60]}...")
                print(f"    Output: {slm_output[:80]}...")
                print(f"    Length: {len(slm_output)} chars")
        else:
            print(f"\n❌ SLM was NOT called")
        
        # Check LLM conversation
        if len(llm_conv) >= 2:
            print(f"\n✅ LLM had {len(llm_conv)} turns")
            
            # Turn 0: Initial request + function call
            turn0 = llm_conv[0]
            print(f"\n  Turn 0 (Initial):")
            print(f"    Input: {turn0['input'][:60]}...")
            print(f"    Output: {turn0['output'][:60] if turn0['output'] else '(empty - normal)'}...")
            print(f"    Function calls: {len(turn0.get('function_calls', []))}")
            if turn0.get('function_calls'):
                for fc in turn0['function_calls']:
                    print(f"      → {fc['name']}('{fc['args'].get('question', '')[:40]}...')")
            
            # Turn 1: Response after SLM
            turn1 = llm_conv[1]
            print(f"\n  Turn 1 (After SLM):")
            print(f"    Input: {turn1['input'][:60]}...")
            turn1_output = turn1['output']
            
            if turn1_output:
                llm_responded_after_slm += 1
                print(f"    Output: ✅ '{turn1_output[:80]}...'")
                print(f"    Length: {len(turn1_output)} chars")
            else:
                print(f"    Output: ❌ EMPTY!")
                print(f"    ⚠️  LLM DID NOT RESPOND after receiving SLM result!")
        else:
            print(f"\n⚠️  Only {len(llm_conv)} turn(s) - expected at least 2")
        
        # Final verdict
        print(f"\n  Final Prediction:")
        if is_empty:
            empty_responses += 1
            print(f"    ❌ Empty: '{prediction}'")
        else:
            print(f"    ✅ Got answer: '{prediction[:80]}...'")
        
        # Diagnosis
        print(f"\n  Diagnosis:")
        if slm_calls and len(llm_conv) >= 2:
            if llm_conv[1]['output']:
                print(f"    ✅ Flow is working: SLM → LLM → Answer")
            else:
                print(f"    ❌ ISSUE: SLM responded, but LLM didn't generate text")
                print(f"    Possible causes:")
                print(f"      1. LLM hit token limit")
                print(f"      2. LLM hit safety filter")
                print(f"      3. Prompt doesn't instruct LLM to respond")
                print(f"      4. Response format issue")
        elif not slm_calls:
            print(f"    ⚠️  SLM was never called - check prompt")
        else:
            print(f"    ⚠️  Conversation incomplete")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {total_problems}")
    print(f"SLM called: {slm_called}/{total_problems} ({slm_called/total_problems*100:.0f}%)")
    print(f"LLM responded after SLM: {llm_responded_after_slm}/{slm_called if slm_called else 1} ({llm_responded_after_slm/slm_called*100 if slm_called else 0:.0f}%)")
    print(f"Empty final responses: {empty_responses}/{total_problems} ({empty_responses/total_problems*100:.0f}%)")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if empty_responses > 0:
        print("\n⚠️  Empty Responses Detected!")
        print("\n1. Check prompts.py:")
        print("   - Verify ROUTER_INSTRUCTIONS_EXPERIMENT includes:")
        print("     'After receiving the tool result, ALWAYS respond with the final answer'")
        print("     'Never leave the response empty'")
        
        print("\n2. Check router_agent.py:")
        print("   - SLM response should be:")
        print("     'The calculation result is: {answer}\\n\\nNow provide your final answer'")
        
        print("\n3. Run with debug logs:")
        print("   python run_router_only.py --samples 5")
        print("   Look for: [DEBUG] messages showing SLM→LLM flow")
        
        print("\n4. Check Gemini API:")
        print("   - Look for finish_reason != 1 (STOP)")
        print("   - May need to increase max_output_tokens")
    
    if slm_called < total_problems:
        print("\n⚠️  SLM Not Always Called!")
        print("\n1. LLM may be trying to solve problems itself")
        print("2. Check prompt emphasizes: 'ALWAYS call slm_help for calculations'")
        print("3. Add more examples to the prompt")
    
    if llm_responded_after_slm < slm_called:
        critical_issues = slm_called - llm_responded_after_slm
        print(f"\n❌ CRITICAL: {critical_issues} case(s) where LLM didn't respond after SLM!")
        print("\nThis is the main issue - see recommendations above.")
    
    if empty_responses == 0 and llm_responded_after_slm == slm_called:
        print("\n✅ All checks passed!")
        print("The LLM↔SLM communication flow is working correctly.")

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'results_router.json'
    analyze_results(results_file)

