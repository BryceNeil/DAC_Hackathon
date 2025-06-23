"""
Tapeout Graph - Main LangGraph Implementation
=============================================

This module implements the main StateGraph that orchestrates all agents
in the ASU tapeout flow using LangGraph's plan-and-execute pattern.
"""

from typing import Dict, Any, Literal, Optional, Union, List
import asyncio
from datetime import datetime
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

from .state import TapeoutState, PlanStep, DesignPlan
from .planning_agent import PlanningAgent
from .spec_analyzer import SpecAnalyzer
from .rtl_generator import RTLGenerationAgent
from .verification_agent import VerificationAgent
from .constraint_generator import ConstraintGenerator
from .physical_designer import PhysicalDesigner
from .validator import Validator
from tools.file_manager import FileManager
from tools.yaml_parser import YAMLParser

# Progress indicators
try:
    from utils.progress_indicator import ProgressIndicator
except ImportError:
    # Fallback if progress indicator not available
    class ProgressIndicator:
        def start_dots(self, *args, **kwargs): pass
        def start_spinner(self, *args, **kwargs): pass
        def start_thinking(self): pass
        def start_planning(self): pass
        def stop(self): pass
        def show_status(self, *args, **kwargs): pass

class TapeoutGraph:
    """Main graph orchestrating the ASU tapeout flow"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514", execution_mode: str = "autonomous"):
        """Initialize the tapeout graph
        
        Args:
            llm_model: The LLM model to use across agents
            execution_mode: Execution mode - "autonomous", "human_in_loop", or "human_approval"
        """
        self.llm_model = llm_model
        self.execution_mode = execution_mode
        self.graph = StateGraph(TapeoutState)
        self.file_manager = FileManager()
        self.yaml_parser = YAMLParser()
        
        # Initialize all agents with execution mode
        self.agents = {
            "planner": PlanningAgent(llm_model),
            "spec_analyzer": SpecAnalyzer(),
            "rtl_generator": RTLGenerationAgent(llm_model),
            "verification_agent": VerificationAgent(llm_model, execution_mode=execution_mode),
            "constraint_generator": ConstraintGenerator(llm_model),
            "physical_designer": PhysicalDesigner(llm_model),
            "validator": Validator()
        }
        
        # Setup the graph
        self.setup_nodes()
        self.setup_edges()
    
    def setup_nodes(self):
        """Add all agent nodes to the graph"""
        
        # Input processing node
        self.graph.add_node("process_input", self.process_input_node)
        
        # Planning nodes
        self.graph.add_node("planner", self.planner_node)
        self.graph.add_node("replan", self.replan_node)
        
        # Agent execution nodes
        self.graph.add_node("spec_analyzer", self.spec_analyzer_node)
        self.graph.add_node("rtl_generator", self.rtl_generator_node)
        self.graph.add_node("verification_agent", self.verification_node)
        self.graph.add_node("constraint_generator", self.constraint_generator_node)
        self.graph.add_node("physical_designer", self.physical_designer_node)
        self.graph.add_node("validator", self.validator_node)
        
        # Recovery and error handling nodes
        self.graph.add_node("orchestrator", self.orchestrator_node)
        self.graph.add_node("error_handler", self.error_handler_node)
    
    def setup_edges(self):
        """Define the execution flow between nodes"""
        
        # Start with input processing
        self.graph.add_edge(START, "process_input")
        
        # Input processing leads to planning
        self.graph.add_edge("process_input", "planner")
        
        # From planner, route to appropriate agent based on plan
        self.graph.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "spec_analyzer": "spec_analyzer",
                "rtl_generator": "rtl_generator",
                "verification_agent": "verification_agent",
                "constraint_generator": "constraint_generator",
                "physical_designer": "physical_designer",
                "validator": "validator",
                "error": "error_handler",
                "end": END
            }
        )
        
        # All agent nodes lead to replan (except validator)
        for agent in ["spec_analyzer", "rtl_generator", "verification_agent",
                     "constraint_generator", "physical_designer"]:
            self.graph.add_conditional_edges(
                agent,
                self.check_agent_result,
                {
                    "replan": "replan",
                    "error": "error_handler"
                }
            )
        
        # Replan routes to next step or end
        self.graph.add_conditional_edges(
            "replan",
            self.route_from_replan,
            {
                "spec_analyzer": "spec_analyzer",
                "rtl_generator": "rtl_generator",
                "verification_agent": "verification_agent",
                "constraint_generator": "constraint_generator",
                "physical_designer": "physical_designer",
                "validator": "validator",
                "end": END
            }
        )
        
        # Validator completes the flow
        self.graph.add_edge("validator", END)
        
        # Error handler can retry, go to orchestrator, or end
        self.graph.add_conditional_edges(
            "error_handler",
            self.route_from_error,
            {
                "replan": "replan",
                "orchestrator": "orchestrator",
                "rtl_generator": "rtl_generator",  # Direct route for synthesis fixes
                "end": END
            }
        )
        
        # Orchestrator can modify plan and retry
        self.graph.add_conditional_edges(
            "orchestrator",
            self.route_from_orchestrator,
            {
                "replan": "replan",
                "rtl_generator": "rtl_generator",  # Can directly retry RTL generation with new strategy
                "end": END
            }
        )
    
    async def process_input_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Process the input YAML file"""
        yaml_path = state["input"]
        
        try:
            # Parse YAML file
            problem_spec = self.yaml_parser.parse_file(yaml_path)
            
            # Extract problem name
            problem_name = list(problem_spec.keys())[0]
            
            return {
                "problem_spec": problem_spec,
                "problem_name": problem_name,
                "start_time": datetime.now(),
                "past_steps": [("input_processing", f"Loaded specification for {problem_name}")]
            }
            
        except Exception as e:
            return {
                "errors": [f"Failed to process input: {str(e)}"],
                "past_steps": [("input_processing", "Failed to load specification")]
            }
    
    async def planner_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Planning node"""
        progress = ProgressIndicator()
        
        try:
            # Show thinking progress
            progress.show_status("🤔", "LLM is analyzing the problem and creating a plan...", 0)
            progress.start_thinking()
            
            planner = self.agents["planner"]
            result = await planner.create_plan(state)
            
            progress.stop()
            
            # Show completion with plan summary
            if result.get("plan") and hasattr(result["plan"], "steps"):
                num_steps = len(result["plan"].steps)
                complexity = result["plan"].complexity if hasattr(result["plan"], "complexity") else "unknown"
                progress.show_status("📋", f"Created plan with {num_steps} steps (complexity: {complexity})", 0)
            
            return result
            
        except Exception as e:
            progress.stop()
            return {"errors": [f"Planning failed: {str(e)}"]}
        finally:
            progress.stop()
    
    async def replan_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Replanning node to move to next step with detailed logging"""
        plan = state.get("plan")
        
        print(f"🔄 Replan node executing...")
        
        if not plan:
            print("❌ No plan available")
            return {"errors": ["No plan found for replanning"]}
        
        print(f"📋 Current plan status: step {plan.current_step}/{len(plan.steps)}")
        
        # Mark current step as completed if not already
        current_step = plan.get_current_step()
        if current_step:
            print(f"📍 Current step: {current_step.step} ({current_step.agent}) - Status: {current_step.status}")
            if current_step.status == "running":
                current_step.mark_completed()
                print(f"✅ Marked current step as completed")
        
        # Move to next step
        plan.move_to_next_step()
        print(f"➡️ Advanced to step: {plan.current_step}/{len(plan.steps)}")
        
        # Check if we're done
        if plan.current_step >= len(plan.steps):
            print("✅ All steps completed, moving to validation")
            return {
                "plan": plan,
                "past_steps": [("replan", "All steps completed, moving to validation")]
            }
        
        # Mark next step as running
        next_step = plan.get_current_step()
        if next_step:
            next_step.mark_running()
            print(f"📍 Next step: {next_step.step} ({next_step.agent}) - marked as running")
        else:
            print("❌ No next step found")
        
        return {
            "plan": plan,
            "past_steps": [("replan", f"Moving to step: {next_step.step if next_step else 'unknown'}")]
        }
    
    async def spec_analyzer_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Specification analysis node"""
        analyzer = self.agents["spec_analyzer"]
        problem_name = state.get("problem_name")
        if not problem_name and state.get("problem_spec"):
            problem_name = list(state["problem_spec"].keys())[0]
        
        result = analyzer.analyze_specification(state["problem_spec"], problem_name)
        
        return {
            "spec_analysis": result,
            "past_steps": [("spec_analysis", f"Analyzed specification: {result.get('problem_name', problem_name)}")]
        }
    
    async def rtl_generator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """RTL generation node"""
        progress = ProgressIndicator()
        
        try:
            # Show RTL generation progress
            progress.show_status("💭", "LLM is generating RTL code...", 0)
            progress.start_thinking()
            
            generator = self.agents["rtl_generator"]
            
            # Update progress message during generation
            await asyncio.sleep(0.5)  # Small delay to show progress
            progress.stop()
            progress.show_status("🔧", "Generating SystemVerilog modules...", 0)
            progress.start_dots("Synthesizing RTL")
            
            result = await generator.generate_rtl(state)
            
            progress.stop()
            
            # Show completion status
            if result.get("rtl_code"):
                lines = len(result["rtl_code"].split('\n'))
                progress.show_status("✨", f"Generated {lines} lines of RTL code", 0)
            
            return result
            
        except Exception as e:
            progress.stop()
            return {"errors": [f"RTL generation failed: {str(e)}"]}
        finally:
            progress.stop()
    
    async def verification_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Verification node using LangChain tools"""
        progress = ProgressIndicator()
        
        try:
            verifier = self.agents["verification_agent"]
            
            if not state.get("rtl_code"):
                return {"errors": ["No RTL code available for verification"]}
            
            # Show verification progress
            progress.show_status("🧪", "Preparing verification environment...", 0)
            progress.start_dots("Running simulations")
            
            # Small delay to show initial progress
            await asyncio.sleep(0.3)
            progress.stop()
            
            # Now show test running progress
            progress.show_status("🔍", "Running functional verification tests...", 0)
            progress.start_spinner("Executing test cases")
            
            # Use the new async method with LangChain tools
            result = await verifier.verify_design(state)
            
            progress.stop()
            
            # Show results
            if result.get("verification_results"):
                status = result["verification_results"].get("status", "unknown")
                if "passed" in str(status).lower():
                    progress.show_status("✅", "All verification tests passed!", 0)
                else:
                    progress.show_status("⚠️", f"Verification completed with status: {status}", 0)
            
            return result
            
        except Exception as e:
            progress.stop()
            return {"errors": [f"Verification failed: {str(e)}"]}
        finally:
            progress.stop()
    
    async def constraint_generator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Constraint generation node using LangChain tools"""
        generator = self.agents["constraint_generator"]
        
        # Use the new async method with LangChain tools
        return await generator.generate_sdc(state)
    
    async def physical_designer_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Physical design node using LangChain tools"""
        
        try:
            designer = self.agents["physical_designer"]
            
            print("\n🏗️ Starting Physical Design Flow...")
            print("   This will run Yosys synthesis and OpenROAD place & route")
            print("   Watch for tool output below:")
            print("=" * 60)
            
            # Use the new async method with LangChain tools - this will show actual tool output
            result = await designer.run_physical_design_with_tools(state)
            
            print("=" * 60)
            print("✅ Physical design flow completed")
            
            return result
            
        except Exception as e:
            return {"errors": [f"Physical design failed: {str(e)}"]}
    
    async def validator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Final validation node"""
        validator = self.agents["validator"]
        
        # Validate all results using the correct method name
        validation_results = validator.validate_complete_flow(
            output_dir=state.get("output_dir", "./output"),
            analysis=state.get("analysis", {}),
            verification_results=state.get("verification_results", {}),
            physical_results=state.get("physical_results", {})
        )
        
        # Prepare final response
        if validation_results.get("valid"):
            final_response = state.get("odb_file_path", "Design completed successfully")
        else:
            final_response = f"Validation failed: {validation_results.get('issues', [])}"
        
        return {
            "final_response": final_response,
            "end_time": datetime.now(),
            "past_steps": [("validation", f"Final validation: {'passed' if validation_results.get('valid') else 'failed'}")]
        }
    
    async def orchestrator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Orchestrator node for analyzing problems and creating recovery strategies"""
        errors = state.get("errors", [])
        past_steps = state.get("past_steps", [])
        
        print("🎭 ORCHESTRATOR AGENT ACTIVATED 🎭")
        print("=" * 60)
        print("🔍 Analyzing current situation...")
        
        # Analyze the problem
        print(f"❌ Total errors: {len(errors)}")
        if errors:
            print("🔍 Recent errors:")
            for i, error in enumerate(errors[-3:]):
                print(f"  {i+1}. {error}")
        
        print(f"📋 Total steps taken: {len(past_steps)}")
        if past_steps:
            print("🔍 Recent steps:")
            for i, (step_type, step_desc) in enumerate(past_steps[-5:]):
                print(f"  {i+1}. [{step_type}] {step_desc}")
        
        # Analyze loop patterns
        recent_step_types = [step[0] for step in past_steps[-10:]]
        unique_types = len(set(recent_step_types))
        print(f"🔄 Loop analysis: {unique_types} unique step types in last 10 steps")
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(state)
        print(f"💡 Recovery strategy: {recovery_strategy['strategy']}")
        print(f"📝 Reason: {recovery_strategy['reason']}")
        
        # Apply the recovery strategy
        if recovery_strategy["strategy"] == "fallback_rtl":
            print("🔧 Switching RTL generation to template-based fallback")
            # Set a flag to use template-based RTL generation
            return {
                "recovery_mode": "template_rtl",
                "past_steps": [("orchestrator", "Switching to template-based RTL generation")]
            }
        elif recovery_strategy["strategy"] == "simplify_plan":
            print("📋 Simplifying execution plan")
            # Modify the plan to skip problematic steps
            plan = state.get("plan")
            if plan:
                # Reset to a simpler approach
                plan.current_step = 1  # Skip spec analysis, go directly to RTL
                return {
                    "plan": plan,
                    "recovery_mode": "simplified",
                    "past_steps": [("orchestrator", "Simplified execution plan - direct RTL generation")]
                }
        
        return {
            "past_steps": [("orchestrator", f"Applied recovery strategy: {recovery_strategy['strategy']}")]
        }
    
    def _determine_recovery_strategy(self, state: TapeoutState) -> Dict[str, str]:
        """Determine the best recovery strategy based on state analysis"""
        errors = state.get("errors", [])
        past_steps = state.get("past_steps", [])
        
        # Check if RTL generation is failing repeatedly
        rtl_errors = [e for e in errors if "rtl" in str(e).lower() or "recursion" in str(e).lower()]
        rtl_steps = [s for s in past_steps if s[0] == "rtl_generation"]
        
        if len(rtl_errors) > 2 or len(rtl_steps) > 5:
            return {
                "strategy": "fallback_rtl",
                "reason": "RTL generation failing repeatedly - switching to template-based approach"
            }
        
        # Check if we're stuck in planning loops
        planning_steps = [s for s in past_steps if s[0] in ["replan", "planner"]]
        if len(planning_steps) > 8:
            return {
                "strategy": "simplify_plan", 
                "reason": "Too many planning iterations - simplifying approach"
            }
        
        # Default strategy
        return {
            "strategy": "retry_current",
            "reason": "No specific pattern detected - retrying current approach"
        }
    
    async def error_handler_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Handle errors and decide on recovery with intelligent routing"""
        errors = state.get("errors", [])
        past_steps = state.get("past_steps", [])
        
        print(f"⚠️ ERROR HANDLER ACTIVATED")
        print(f"❌ Processing error: {errors[-1] if errors else 'Unknown error'}")
        
        error_msg = errors[-1] if errors else "Unknown error"
        
        # Analyze the error type and determine recovery strategy
        recovery_action = self._analyze_error_for_recovery(error_msg, past_steps, state)
        
        if recovery_action["action"] == "fix_rtl":
            print(f"🔧 Synthesis failure detected - routing back to fix RTL")
            # Add synthesis error details to state so RTL generator can use them
            result = {
                "synthesis_error_feedback": recovery_action["feedback"],
                "recovery_mode": "fix_synthesis_errors",
                "rtl_fix_attempt": state.get("rtl_fix_attempt", 0) + 1,
                "past_steps": [(
                    "error_handling",
                    f"Synthesis failed: {recovery_action['feedback']} - Routing back to RTL generation"
                )]
            }
            
            # Include parsed synthesis error details if available
            if state.get("synthesis_error_details"):
                result["synthesis_error_details"] = state["synthesis_error_details"]
            if state.get("rtl_fix_instructions"):
                result["rtl_fix_instructions"] = state["rtl_fix_instructions"]
                
            return result
        elif recovery_action["action"] == "continue":
            return {
                "past_steps": [(
                    "error_handling",
                    error_msg + " - Attempting to continue with next step"
                )]
            }
        else:
            # Default fallback
            return {
                "past_steps": [(
                    "error_handling",
                    error_msg + " - Using default recovery"
                )]
            }
            
    def _analyze_error_for_recovery(self, error_msg: str, past_steps: list, state: TapeoutState) -> dict:
        """Analyze error and determine the best recovery strategy"""
        
        # Check if this is a synthesis-related error
        synthesis_indicators = [
            "synthesis failed", "yosys", "syntax error", "Synthesis: Failed",
            "hierarchy", "read_verilog", "module", "parameter int"
        ]
        
        if any(indicator.lower() in error_msg.lower() for indicator in synthesis_indicators):
            # This is a synthesis error - we should fix the RTL
            rtl_fix_attempts = state.get("rtl_fix_attempt", 0)
            
            if rtl_fix_attempts < 3:  # Allow up to 3 RTL fix attempts
                # Extract specific synthesis feedback
                feedback = self._extract_synthesis_feedback(error_msg, state)
                return {
                    "action": "fix_rtl",
                    "feedback": feedback
                }
        
        # Check for verification errors
        if "verification failed" in error_msg.lower():
            verification_attempts = len([s for s in past_steps if s[0] == "verification" and "failed" in s[1]])
            if verification_attempts < 2:
                return {
                    "action": "retry_verification"
                }
        
        # Default - just continue
        return {
            "action": "continue"
        }
    
    def _extract_synthesis_feedback(self, error_msg: str, state: TapeoutState) -> str:
        """Extract specific synthesis error feedback to help RTL generation"""
        
        # Get synthesis log if available
        physical_results = state.get("physical_results", {})
        synthesis_log = ""
        
        if "synthesis_log" in physical_results:
            synthesis_log = physical_results["synthesis_log"]
        
        # Common synthesis issues and their fixes
        if "parameter int" in error_msg or "parameter int" in synthesis_log:
            return "Remove 'int' keyword from parameters. Use 'parameter N = 8' instead of 'parameter int N = 8'"
        
        if "syntax error" in error_msg.lower():
            return "Fix Verilog syntax errors. Ensure proper module structure and signal declarations"
        
        if "rst_n" in synthesis_log or "rst_n" in error_msg:
            return "Signal 'rst_n' is undefined. Use 'rst' signal or define 'rst_n' properly"
        
        if "hierarchy" in error_msg.lower():
            return "Fix module hierarchy issues. Ensure all modules are properly defined"
        
        if "empty" in error_msg.lower() or "no logic" in error_msg.lower():
            return "Module has no functional logic. Implement the required functionality"
        
        # Generic feedback
        return f"Synthesis failed. Check RTL syntax and logic implementation. Error: {error_msg[:200]}"
    
    def route_from_planner(self, state: TapeoutState) -> str:
        """Route from planner to first agent or end"""
        plan = state.get("plan")
        errors = state.get("errors", [])
        
        if errors:
            return "error"
        
        if not plan or not plan.steps:
            return "end"
        
        # Get first step
        first_step = plan.steps[0]
        first_step.mark_running()
        
        return first_step.agent
    
    def route_from_replan(self, state: TapeoutState) -> str:
        """Route from replan to next agent with detailed logging"""
        plan = state.get("plan")
        
        print(f"🗺️ Routing from replan...")
        
        if not plan:
            print("❌ No plan found - routing to end")
            return "end"
        
        print(f"📋 Plan status: step {plan.current_step}/{len(plan.steps)}")
        
        # Check if all steps are done
        if plan.current_step >= len(plan.steps):
            print("✅ All steps completed - routing to validator")
            return "validator"
        
        # Get current step
        current_step = plan.get_current_step()
        if current_step:
            print(f"➡️ Routing to agent: {current_step.agent} for step: {current_step.step}")
            return current_step.agent
        
        print("❌ No current step found - routing to end")
        return "end"
    
    def check_agent_result(self, state: TapeoutState) -> str:
        """Check agent result and route accordingly with detailed logging"""
        errors = state.get("errors", [])
        
        print(f"🔍 Checking agent result...")
        print(f"❌ Total errors: {len(errors)}")
        
        if errors:
            print(f"🔍 Recent errors: {errors[-3:]}")
        
        # Check for new errors
        if errors and any("failed" in str(error).lower() for error in errors[-3:]):
            print("❌ Recent failures detected - routing to error handler")
            return "error"
        
        print("✅ No recent failures - routing to replan")
        return "replan"
    
    def route_from_error(self, state: TapeoutState) -> str:
        """Route from error handler with intelligent recovery routing"""
        errors = state.get("errors", [])
        recovery_mode = state.get("recovery_mode")
        
        print(f"🔍 Error handler routing...")
        print(f"❌ Total errors: {len(errors)}")
        
        # Check if we should route to RTL generator for synthesis fix
        if recovery_mode == "fix_synthesis_errors":
            rtl_fix_attempts = state.get("rtl_fix_attempt", 0)
            print(f"🔧 Routing to RTL generator for synthesis fix (attempt {rtl_fix_attempts})")
            return "rtl_generator"
        
        # If too many errors, go to orchestrator for recovery
        if len(errors) > 8:
            print("❌ Too many errors - going to orchestrator for recovery")
            return "orchestrator"
        
        # Check for loop patterns in past steps
        past_steps = state.get("past_steps", [])
        if len(past_steps) > 15:
            # Check if we're stuck in a loop (same step type repeating)
            recent_steps = [step[0] for step in past_steps[-10:]]
            unique_recent = len(set(recent_steps))
            print(f"🔍 Loop detection: {unique_recent} unique step types in last 10 steps")
            print(f"📋 Recent steps: {recent_steps}")
            
            if unique_recent < 3:  # Only 2 or fewer unique step types in last 10 steps
                print("⚠️ Loop detected! Going to orchestrator for recovery strategy")
                return "orchestrator"
        
        print("🔄 Continuing to replan")
        return "replan"
    
    def route_from_orchestrator(self, state: TapeoutState) -> str:
        """Route from orchestrator based on recovery strategy"""
        recovery_mode = state.get("recovery_mode")
        
        print(f"🎭 Orchestrator routing with recovery mode: {recovery_mode}")
        
        if recovery_mode == "template_rtl":
            print("➡️ Going directly to RTL generator with template mode")
            return "rtl_generator"
        elif recovery_mode == "simplified":
            print("➡️ Going back to replan with simplified approach")
            return "replan"
        else:
            print("➡️ Default routing back to replan")
            return "replan"
    
    def compile(self, checkpointer: Optional[Any] = None) -> Any:
        """Compile the graph with optional checkpointing
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            
        Returns:
            Compiled graph ready for execution
        """
        if checkpointer is None:
            checkpointer = MemorySaver()
            
        return self.graph.compile(checkpointer=checkpointer)
    
    async def run(self, yaml_path: str, thread_id: str = "default") -> Dict[str, Any]:
        """Run the complete tapeout flow
        
        Args:
            yaml_path: Path to the input YAML specification
            thread_id: Thread ID for checkpointing
            
        Returns:
            Final state after execution
        """
        # Compile the graph
        app = self.compile()
        
        # Initial state
        initial_state = {
            "input": yaml_path,
            "past_steps": [],
            "errors": []
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Stream execution for better visibility
            final_state = None
            async for event in app.astream(initial_state, config):
                # Log progress
                for node, updates in event.items():
                    if "past_steps" in updates:
                        for step in updates["past_steps"]:
                            print(f"[{node}] {step[1]}")
                
                final_state = event
            
            return final_state
            
        except Exception as e:
            print(f"Error running graph: {str(e)}")
            return {
                "errors": [f"Graph execution failed: {str(e)}"],
                "final_response": None
            } 