#!/usr/bin/env python3
"""
Example: Running ASU Tapeout Agent in Autonomous Mode
=====================================================

This example demonstrates how to run the agent in fully autonomous mode
where it automatically handles errors and retries without human intervention.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from your_agent_langgraph import ASUTapeoutAgent


async def main():
    """Demonstrate autonomous mode execution"""
    
    print("=" * 60)
    print("ASU Tapeout Agent - Autonomous Mode Demo")
    print("=" * 60)
    print()
    
    # Example problem YAML
    test_problem = {
        "simple_counter": {
            "description": "A 4-bit counter that increments on each clock cycle",
            "clock_period": "1.0ns",
            "module_signature": "module simple_counter(input clk, input reset, output reg [3:0] count);"
        }
    }
    
    # Save test problem to file
    os.makedirs("test_problems", exist_ok=True)
    test_yaml_path = "test_problems/simple_counter.yaml"
    
    import yaml
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_problem, f)
    
    # Initialize agent in different modes
    modes = ["autonomous", "human_in_loop", "human_approval"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing {mode.upper()} mode")
        print(f"{'='*60}\n")
        
        # Initialize agent
        agent = ASUTapeoutAgent(mode=mode)
        
        # Create output directory
        output_dir = f"test_output/{mode}_mode"
        os.makedirs(output_dir, exist_ok=True)
        
        # Solve problem
        success = agent.solve_problem(test_yaml_path, output_dir)
        
        if success:
            print(f"✅ {mode} mode completed successfully")
            
            # Check generated files
            files = os.listdir(output_dir)
            print(f"Generated files: {files}")
        else:
            print(f"❌ {mode} mode failed")
        
        print()
        
        # Only run autonomous mode in this example
        if mode == "autonomous":
            print("Note: In autonomous mode, the agent:")
            print("- Automatically fixes syntax errors")
            print("- Retries failed verifications")
            print("- Adjusts timing constraints")
            print("- Continues without human intervention")
            break  # Don't actually run human modes in this example


def demonstrate_error_recovery():
    """Show how autonomous error recovery works"""
    
    print("\n" + "=" * 60)
    print("Autonomous Error Recovery Example")
    print("=" * 60 + "\n")
    
    # Example with intentional error
    buggy_rtl = """
module simple_counter(input clk, input reset, output reg [3:0] count);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 0
        else
            count <= count + 1;  // Missing semicolon above!
    end
endmodule
"""
    
    print("Original RTL with syntax error:")
    print(buggy_rtl)
    print("\nIn autonomous mode, the agent will:")
    print("1. Detect the syntax error during verification")
    print("2. Use LLM to analyze and fix the error")
    print("3. Retry verification with fixed RTL")
    print("4. Continue to next steps automatically")


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())
    
    # Show error recovery example
    demonstrate_error_recovery()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60) 