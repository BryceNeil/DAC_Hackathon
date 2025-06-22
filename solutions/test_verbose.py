#!/usr/bin/env python3
"""
Quick test script for verbose RTL agent
=======================================

This script automatically runs the verbose agent on the p1 problem
to demonstrate the new logging and orchestrator features.
"""

import os
import sys
import time
from pathlib import Path

# Add the solutions directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from your_agent_langgraph import ASUTapeoutAgent


def main():
    # Use the existing p1 problem
    problem_file = "../problems/visible/p1.yaml"
    output_dir = "./test_output"
    
    print("=" * 80)
    print("🧪 QUICK TEST - ASU TAPEOUT AGENT VERBOSE MODE")
    print("=" * 80)
    print(f"📁 Problem: {problem_file}")
    print(f"📁 Output: {output_dir}")
    print("🤖 Model: claude-sonnet-4-20250514")
    print("=" * 80)
    
    # Check if problem file exists
    if not os.path.exists(problem_file):
        print(f"❌ Problem file not found: {problem_file}")
        print("💡 Make sure you're running from the solutions directory")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create agent with verbose mode
    agent = ASUTapeoutAgent(
        execution_mode="autonomous",
        model_name="claude-sonnet-4-20250514"
    )
    
    # Start timing
    start_time = time.time()
    
    try:
        print("🔄 Starting agent execution...")
        print("💡 WHAT TO WATCH FOR:")
        print("   🔧 Tool calls and iterations in RTL generation")
        print("   🔄 Agent handoffs between different components")
        print("   🎭 Orchestrator activation when loops are detected")
        print("   ⚠️  Loop warnings when iteration counts get high")
        print("=" * 80)
        
        # Run with maximum verbosity
        success = agent.solve_problem(
            yaml_file=problem_file,
            output_dir=output_dir,
            verbose=True  # Enable verbose mode
        )
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("=" * 80)
        print(f"⏱️ Total execution time: {execution_time:.1f} seconds")
        
        if success:
            print("✅ Agent completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            
            # Show generated files
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                if files:
                    print("📝 Generated files:")
                    for file in files:
                        print(f"   - {file}")
                else:
                    print("⚠️  No files generated")
        else:
            print("❌ Agent failed to complete")
            print("💡 Check the verbose output above to see where it got stuck")
            
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        print("💡 This is normal if you saw the agent getting stuck in loops")
        print("💡 The orchestrator should have activated to break the loop")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
    print("🔍 ANALYSIS TIPS:")
    print("   - Look for repeated tool calls or agent handoffs")
    print("   - Check if orchestrator activated and what strategy it chose")
    print("   - Note where the highest iteration counts occurred")
    print("=" * 80)


if __name__ == "__main__":
    main() 