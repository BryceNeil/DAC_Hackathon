"""
Tapeout Graph - Main LangGraph Implementation
=============================================

This module implements the main StateGraph that orchestrates all agents
in the ASU tapeout flow using LangGraph's plan-and-execute pattern.
"""

from typing import Dict, Any, Literal, Optional, Union, List
import asyncio
from datetime import datetime

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
        
        # Error handling node
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
        
        # Error handler can retry or end
        self.graph.add_conditional_edges(
            "error_handler",
            self.route_from_error,
            {
                "replan": "replan",
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
        planner = self.agents["planner"]
        return await planner.create_plan(state)
    
    async def replan_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Replanning node to move to next step"""
        plan = state.get("plan")
        
        if not plan:
            return {"errors": ["No plan found for replanning"]}
        
        # Mark current step as completed if not already
        current_step = plan.get_current_step()
        if current_step and current_step.status == "running":
            current_step.mark_completed()
        
        # Move to next step
        plan.move_to_next_step()
        
        # Check if we're done
        if plan.current_step >= len(plan.steps):
            return {
                "plan": plan,
                "past_steps": [("replan", "All steps completed, moving to validation")]
            }
        
        # Mark next step as running
        next_step = plan.get_current_step()
        if next_step:
            next_step.mark_running()
        
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
        generator = self.agents["rtl_generator"]
        return await generator.generate_rtl(state)
    
    async def verification_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Verification node using LangChain tools"""
        verifier = self.agents["verification_agent"]
        
        if not state.get("rtl_code"):
            return {"errors": ["No RTL code available for verification"]}
        
        # Use the new async method with LangChain tools
        return await verifier.verify_design(state)
    
    async def constraint_generator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Constraint generation node using LangChain tools"""
        generator = self.agents["constraint_generator"]
        
        # Use the new async method with LangChain tools
        return await generator.generate_sdc(state)
    
    async def physical_designer_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Physical design node using LangChain tools"""
        designer = self.agents["physical_designer"]
        
        # Use the new async method with LangChain tools
        return await designer.run_physical_design_with_tools(state)
    
    async def validator_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Final validation node"""
        validator = self.agents["validator"]
        
        # Validate all results
        validation_results = validator.validate_all(
            rtl_code=state.get("rtl_code"),
            verification_results=state.get("verification_results"),
            physical_results=state.get("physical_results"),
            problem_spec=state["problem_spec"]
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
    
    async def error_handler_node(self, state: TapeoutState) -> Dict[str, Any]:
        """Handle errors and decide on recovery"""
        errors = state.get("errors", [])
        
        # Log the error
        error_msg = f"Error encountered: {errors[-1] if errors else 'Unknown error'}"
        
        # For now, just log and try to continue
        # In a more sophisticated implementation, we could analyze the error
        # and decide on appropriate recovery action
        
        return {
            "past_steps": [(
                "error_handling",
                error_msg + " - Attempting to continue with next step"
            )]
        }
    
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
        """Route from replan to next agent"""
        plan = state.get("plan")
        
        if not plan:
            return "end"
        
        # Check if all steps are done
        if plan.current_step >= len(plan.steps):
            return "validator"
        
        # Get current step
        current_step = plan.get_current_step()
        if current_step:
            return current_step.agent
        
        return "end"
    
    def check_agent_result(self, state: TapeoutState) -> str:
        """Check agent result and route accordingly"""
        errors = state.get("errors", [])
        
        # Check for new errors
        if errors and any("failed" in str(error).lower() for error in errors[-3:]):
            return "error"
        
        return "replan"
    
    def route_from_error(self, state: TapeoutState) -> str:
        """Route from error handler"""
        errors = state.get("errors", [])
        
        # If too many errors, end
        if len(errors) > 5:
            return "end"
        
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