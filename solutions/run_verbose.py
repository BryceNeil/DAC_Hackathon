#!/usr/bin/env python3
"""
Verbose runner for ASU Tapeout Agent
===================================

This script runs the agent with maximum verbosity to show all iterations,
tool calls, and decision points. Useful for debugging loops and understanding
the agent's behavior.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the solutions directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from your_agent_langgraph import ASUTapeoutAgent


def main():
    parser = argparse.ArgumentParser(description="Run ASU Tapeout Agent with verbose logging")
    parser.add_argument("--problem", required=True, help="Path to problem YAML file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model to use")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ASU TAPEOUT AGENT - VERBOSE MODE")
    print("=" * 80)
    print(f"📁 Problem: {args.problem}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🤖 Model: {args.model}")
    print("=" * 80)
    
    # Create agent with verbose mode
    agent = ASUTapeoutAgent(
        execution_mode="autonomous",
        model_name=args.model
    )
    
    # Start timing
    start_time = time.time()
    
    try:
        print("🔄 Starting agent execution...")
        print("💡 TIP: Watch for iteration counts and tool calls to identify loops")
        print("💡 LEGEND:")
        print("   🔄 AGENT HANDOFF = Transition between different agents")
        print("   🔧 TOOL CALL = Agent using a specific tool")
        print("   🎭 ORCHESTRATOR = Recovery agent analyzing problems")
        print("   ⚠️  LOOP WARNING = High iteration count detected")
        print("-" * 80)
        
        # Run with maximum verbosity
        success = agent.solve_problem(
            yaml_file=args.problem,
            output_dir=args.output_dir,
            verbose=True  # Enable verbose mode
        )
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("-" * 80)
        print(f"⏱️ Total execution time: {execution_time:.1f} seconds")
        
        if success:
            print("✅ Agent completed successfully!")
            print(f"📁 Results saved to: {args.output_dir}")
        else:
            print("❌ Agent failed to complete")
            
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        print("💡 This is normal if you saw the agent getting stuck in loops")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)


if __name__ == "__main__":
    main() 