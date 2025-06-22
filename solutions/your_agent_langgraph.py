#!/usr/bin/env python3
"""
ASU Spec2Tapeout ICLAD 2025 Hackathon Agent - LangGraph Implementation
=====================================================================

Minimal orchestration layer using LangGraph for the ASU Tapeout flow.
This implements the plan-and-execute pattern with proper state management.
"""

import asyncio
import os
import shutil
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path to support relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# LangGraph imports
from agents.tapeout_graph import TapeoutGraph
from agents.state import TapeoutState


class ASUTapeoutAgent:
    """Minimal LangGraph-based ASU Tapeout Agent"""
    
    def __init__(self, llm_api_key: Optional[str] = None, execution_mode: str = "autonomous", model_name: str = "claude-sonnet-4-20250514"):
        """Initialize with LangGraph-based agent system
        
        Args:
            llm_api_key: Optional API key for LLM (OpenAI or Anthropic)
            execution_mode: Execution mode - "autonomous", "human_in_loop", or "human_approval"
            model_name: Name of the LLM model to use (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")
        """
        self.llm_api_key = llm_api_key
        self.execution_mode = execution_mode
        self.model_name = model_name
        
        # Set appropriate API key based on model type
        if "claude" in model_name.lower():
            # For Claude models, check for API key in this order:
            # 1. Provided parameter
            # 2. ANTHROPIC_API_KEY environment variable
            # 3. API_KEY environment variable (fallback)
            api_key = llm_api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                self.llm_api_key = api_key
                print(f"🔑 Using Anthropic API with model: {model_name}")
            else:
                print("⚠️  No Anthropic API key found. Please set ANTHROPIC_API_KEY or API_KEY in .env file")
                
        else:
            # For OpenAI models, check for API key in this order:
            # 1. Provided parameter
            # 2. OPENAI_API_KEY environment variable
            # 3. API_KEY environment variable (fallback)
            api_key = llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                self.llm_api_key = api_key
                print(f"🔑 Using OpenAI API with model: {model_name}")
            else:
                print("⚠️  No OpenAI API key found. Please set OPENAI_API_KEY or API_KEY in .env file")
        
        # Create the LangGraph workflow with execution mode and model
        self.graph_builder = TapeoutGraph(llm_model=model_name, execution_mode=execution_mode)
        self.workflow = self.graph_builder.compile()
        
    def solve_problem(self, yaml_file: str, output_dir: str, verbose: bool = True) -> bool:
        """Solve using LangGraph plan-and-execute pattern with streaming
        
        Args:
            yaml_file: Path to the problem YAML file
            output_dir: Directory to save results
            verbose: Whether to print execution steps
            
        Returns:
            True if successful, False otherwise
        """
        # Load specification
        spec = self.load_problem_spec(yaml_file)
        if not spec:
            return False
            
        # Create initial state
        initial_state = TapeoutState(
            messages=[],  # Required for LangGraph compatibility
            input=yaml_file,
            problem_spec=spec,
            problem_name=None,  # Will be extracted by process_input
            plan=None,  # Will be created by planner
            past_steps=[],
            rtl_code=None,
            sdc_constraints=None,
            verification_results=None,
            physical_results=None,
            final_response=None,
            odb_file_path=None,
            errors=[],
            metrics=None,
            current_step="start",
            start_time=None,
            end_time=None
        )
        
        # Run the LangGraph workflow with streaming
        try:
            # Create a unique thread ID for this problem
            thread_id = f"{Path(yaml_file).stem}_{os.getpid()}"
            config = {
                "recursion_limit": 50,
                "configurable": {"thread_id": thread_id}
            }
            
            # Run async workflow
            final_state = asyncio.run(self._run_workflow_async(initial_state, config, verbose))
            
            # Save results to output directory
            self._save_results(final_state, output_dir)
            return True
            
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            return False
    
    async def _run_workflow_async(self, initial_state: TapeoutState, config: dict, verbose: bool) -> TapeoutState:
        """Run the workflow asynchronously with streaming output
        
        Args:
            initial_state: Initial state for the workflow
            config: LangGraph configuration
            verbose: Whether to print execution steps
            
        Returns:
            Final state after execution
        """
        final_state = initial_state
        
        if verbose:
            print(f"🔄 Starting LangGraph workflow (mode: {self.execution_mode})...")
            
        # Stream execution for visibility
        async for event in self.workflow.astream(initial_state, config):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    if verbose:
                        print(f"\n📍 Node: {node_name}")
                        
                    # Update final state with node output
                    if isinstance(node_output, dict):
                        final_state.update(node_output)
                        
                        # Print relevant updates
                        if verbose:
                            if "past_steps" in node_output and node_output["past_steps"]:
                                for step in node_output["past_steps"]:
                                    print(f"   ✓ {step[1]}")
                            
                            if "errors" in node_output and node_output["errors"]:
                                for error in node_output["errors"]:
                                    print(f"   ⚠️  Error: {error}")
                            
                            if "plan" in node_output and node_output["plan"]:
                                plan = node_output["plan"]
                                if hasattr(plan, "steps"):
                                    print(f"   📋 Plan: {len(plan.steps)} steps")
                            
                            if "rtl_code" in node_output and node_output["rtl_code"]:
                                print(f"   💾 RTL generated ({len(node_output['rtl_code'])} chars)")
                            
                            if "verification_results" in node_output and node_output["verification_results"]:
                                results = node_output["verification_results"]
                                print(f"   🧪 Verification: {results.get('status', 'completed')}")
                            
                            if "physical_results" in node_output and node_output["physical_results"]:
                                results = node_output["physical_results"]
                                print(f"   🏗️  Physical design: {results.get('status', 'completed')}")
        
        if verbose:
            print("\n✅ Workflow completed")
            
        return final_state
    
    def load_problem_spec(self, yaml_file: str) -> Optional[Dict[str, Any]]:
        """Load and parse YAML specification"""
        try:
            with open(yaml_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load YAML file: {e}")
            return None
    
    def _save_results(self, state: TapeoutState, output_dir: str):
        """Save final results to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        problem_name = state.get("problem_name")
        if not problem_name and state.get("problem_spec"):
            problem_name = list(state["problem_spec"].keys())[0]
        
        if not problem_name:
            problem_name = "unknown"
        
        # Save RTL
        if state.get("rtl_code"):
            with open(f"{output_dir}/{problem_name}.v", 'w') as f:
                f.write(state["rtl_code"])
            print(f"  ✓ RTL saved to {output_dir}/{problem_name}.v")
        
        # Save SDC
        if state.get("sdc_constraints"):
            with open(f"{output_dir}/6_final.sdc", 'w') as f:
                f.write(state["sdc_constraints"])
            print(f"  ✓ SDC saved to {output_dir}/6_final.sdc")
        
        # Copy ODB file if generated
        if state.get("odb_file_path") and os.path.exists(state["odb_file_path"]):
            shutil.copy(state["odb_file_path"], f"{output_dir}/6_final.odb")
            print(f"  ✓ ODB saved to {output_dir}/6_final.odb")
        elif state.get("physical_results") and state["physical_results"].get("odb_file"):
            odb_path = state["physical_results"]["odb_file"]
            if os.path.exists(odb_path):
                shutil.copy(odb_path, f"{output_dir}/6_final.odb")
                print(f"  ✓ ODB saved to {output_dir}/6_final.odb")

    # Legacy methods for backwards compatibility - now handled by LangGraph workflow
    def generate_rtl(self, spec: Dict[str, Any], problem_name: str) -> Optional[str]:
        """Legacy method - now handled by LangGraph workflow"""
        pass  # Functionality moved to RTLGenerationAgent

    def verify_rtl_functionality(self, rtl_file: str, problem_name: str) -> bool:
        """Legacy method - now handled by LangGraph workflow"""
        pass  # Functionality moved to VerificationAgent

    def run_openroad_flow(self, rtl_file: str, sdc_file: str, output_dir: str, problem_name: str) -> bool:
        """Legacy method - now handled by LangGraph workflow"""
        pass  # Functionality moved to PhysicalDesignAgent

    def generate_sdc_constraints(self, spec: Dict[str, Any], problem_name: str) -> Optional[str]:
        """Legacy method - now handled by LangGraph workflow"""
        pass  # Functionality moved to ConstraintGenerationAgent


def main():
    """Main entry point with LangGraph workflow"""
    parser = argparse.ArgumentParser(
        description='ASU Spec2Tapeout Agent - LangGraph Implementation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--problem', help='Single YAML problem file to solve')
    parser.add_argument('--problem_dir', help='Directory containing problem YAML files')
    parser.add_argument('--output_dir', help='Output directory for single problem')
    parser.add_argument('--output_base', help='Base output directory for multiple problems')
    parser.add_argument('--llm_key', help='API key (OpenAI or Anthropic)')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='LLM model to use (e.g., claude-sonnet-4-20250514, gpt-4o)')
    parser.add_argument('--mode', choices=['autonomous', 'human_in_loop', 'human_approval'],
                       default='autonomous', help='Execution mode (default: autonomous)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Initialize agent with execution mode and model
    agent = ASUTapeoutAgent(llm_api_key=args.llm_key, execution_mode=args.mode, model_name=args.model)
    verbose = not args.quiet
    
    if args.problem and args.output_dir:
        # Single problem mode
        print(f"🚀 Solving problem: {args.problem}")
        print(f"🔧 Mode: {args.mode}")
        success = agent.solve_problem(args.problem, args.output_dir, verbose=verbose)
        if success:
            print(f"✅ Successfully completed {args.problem}")
        else:
            print(f"❌ Failed to complete {args.problem}")
            
    elif args.problem_dir and args.output_base:
        # Multiple problems mode
        problem_files = list(Path(args.problem_dir).glob("*.yaml"))
        print(f"🚀 Found {len(problem_files)} problems to solve")
        print(f"🔧 Mode: {args.mode}")
        
        success_count = 0
        for prob_file in problem_files:
            prob_name = prob_file.stem
            output_dir = os.path.join(args.output_base, prob_name)
            
            print(f"\n📋 Solving {prob_name}...")
            if agent.solve_problem(str(prob_file), output_dir, verbose=verbose):
                success_count += 1
        
        print(f"\n📊 Summary: {success_count}/{len(problem_files)} problems solved successfully")
        
    else:
        print("ASU Spec2Tapeout Agent - LangGraph Implementation\n")
        print("Usage examples:")
        print("  Single problem (autonomous mode):")
        print("    python your_agent_langgraph.py --problem problems/visible/p1.yaml --output_dir solutions/visible/p1/")
        print("\n  Multiple problems (autonomous mode):")
        print("    python your_agent_langgraph.py --problem_dir problems/visible/ --output_base solutions/visible/")
        print("\n  With Claude-3.5-Sonnet:")
        print("    python your_agent_langgraph.py --model claude-3-5-sonnet-20241022 --llm_key YOUR_ANTHROPIC_KEY --problem p1.yaml --output_dir output/")
        print("\n  With human-in-the-loop:")
        print("    python your_agent_langgraph.py --mode human_in_loop --problem p1.yaml --output_dir output/")
        print("\n  Quiet mode (minimal output):")
        print("    python your_agent_langgraph.py --quiet --problem p1.yaml --output_dir output/")
        print("\n  With API key:")
        print("    python your_agent_langgraph.py --llm_key YOUR_KEY --model gpt-4o --problem p1.yaml --output_dir output/")
        parser.print_help()


if __name__ == "__main__":
    main() 