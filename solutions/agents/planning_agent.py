"""
Planning Agent for ASU Tapeout Flow
====================================

This agent creates execution plans for the EDA tapeout flow using
LangChain's structured output capabilities.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from .state import TapeoutState, PlanStep, DesignPlan


class EDAExecutionPlan(BaseModel):
    """Plan for EDA tapeout flow execution"""
    steps: List[str] = Field(
        description="Ordered steps for RTL-to-GDSII flow, should include: "
                   "spec_analysis, rtl_generation, verification, constraint_generation, "
                   "physical_design, validation"
    )
    estimated_complexity: str = Field(
        description="low/medium/high complexity assessment"
    )
    recommended_tools: List[str] = Field(
        description="Specific EDA tools recommended for this design"
    )
    key_challenges: List[str] = Field(
        description="Key challenges identified for this design"
    )


# Planning prompt template
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert EDA planning agent responsible for creating execution plans 
    for ASIC design flows from RTL to GDSII.
    
    Given a YAML specification for an ASIC design, create a detailed execution plan that covers:
    
    1. **Specification Analysis**: Understanding the design requirements
    2. **RTL Generation**: Creating synthesizable SystemVerilog code
    3. **Verification**: Testing the RTL against specifications
    4. **Constraint Generation**: Creating timing and physical constraints
    5. **Physical Design**: Running synthesis and place-and-route
    6. **Validation**: Final checks and metric evaluation
    
    Consider the following factors when planning:
    - Design type (combinational vs sequential logic)
    - Timing requirements and clock periods
    - Module complexity and interface specifications
    - Verification needs based on functionality
    - Physical design constraints from the specification
    - Available tools: Icarus Verilog, Yosys, OpenROAD
    
    Always include ALL six phases in your plan, even if some might be simpler for certain designs.
    
    Assess complexity based on:
    - Low: Simple combinational logic, no timing constraints
    - Medium: Sequential logic with moderate timing requirements
    - High: Complex state machines, tight timing, multiple clock domains"""),
    
    ("user", """Design specification:
{spec}

Please create a comprehensive execution plan for this ASIC design.""")
])


class PlanningAgent:
    """Agent responsible for creating and managing execution plans"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514"):
        """Initialize the planning agent
        
        Args:
            llm_model: The LLM model to use for planning
        """
        # Select LLM based on model name
        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(model=llm_model, temperature=0)
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.planner = planner_prompt | self.llm.with_structured_output(EDAExecutionPlan)
    
    async def create_plan(self, state: TapeoutState) -> Dict[str, Any]:
        """Create initial execution plan from specification
        
        Args:
            state: Current tapeout state with problem specification
            
        Returns:
            State update with execution plan
        """
        spec = state["problem_spec"]
        
        # Generate plan using LLM
        plan_result = await self.planner.ainvoke({"spec": spec})
        
        # Convert to PlanStep objects with proper agent routing
        plan_steps = []
        for idx, step in enumerate(plan_result.steps):
            agent_name = self._get_agent_for_step(step)
            plan_step = PlanStep(
                step=step,
                agent=agent_name,
                dependencies=[plan_result.steps[i] for i in range(idx)]  # Each step depends on previous ones
            )
            plan_steps.append(plan_step)
        
        # Create DesignPlan
        design_plan = DesignPlan(
            steps=plan_steps,
            design_complexity=plan_result.estimated_complexity,
            estimated_runtime=self._estimate_runtime(plan_result.estimated_complexity)
        )
        
        # Log planning step
        planning_message = (
            f"Created execution plan with {len(plan_steps)} steps. "
            f"Complexity: {plan_result.estimated_complexity}. "
            f"Key challenges: {', '.join(plan_result.key_challenges)}"
        )
        
        return {
            "plan": design_plan,
            "past_steps": [("planning", planning_message)]
        }
    
    def _get_agent_for_step(self, step: str) -> str:
        """Route steps to appropriate agents
        
        Args:
            step: Step description
            
        Returns:
            Agent name to handle the step
        """
        # Normalize step name
        step_lower = step.lower()
        
        # Routing table
        routing = {
            "spec_analysis": "spec_analyzer",
            "rtl_generation": "rtl_generator",
            "verification": "verification_agent",
            "constraint_generation": "constraint_generator",
            "physical_design": "physical_designer", 
            "validation": "validator"
        }
        
        # Match based on keywords
        for key, agent in routing.items():
            if key in step_lower:
                return agent
        
        # Default to generic agent
        return "generic_agent"
    
    def _estimate_runtime(self, complexity: str) -> float:
        """Estimate runtime based on complexity
        
        Args:
            complexity: Design complexity (low/medium/high)
            
        Returns:
            Estimated runtime in minutes
        """
        runtime_map = {
            "low": 5.0,
            "medium": 15.0,
            "high": 30.0
        }
        return runtime_map.get(complexity, 15.0)
    
    async def replan(self, state: TapeoutState, error_context: Optional[str] = None) -> Dict[str, Any]:
        """Replan based on current progress or errors
        
        Args:
            state: Current tapeout state
            error_context: Optional error context for replanning
            
        Returns:
            Updated state with new plan or adjustments
        """
        plan = state.get("plan")
        if not plan:
            # No existing plan, create new one
            return await self.create_plan(state)
        
        # If we have an error, we might need to adjust the plan
        if error_context:
            # For now, just log the error and continue
            # In a more sophisticated implementation, we could modify the plan
            return {
                "errors": [f"Planning adjustment needed: {error_context}"]
            }
        
        # Normal progression - mark current step complete and move to next
        current_step = plan.get_current_step()
        if current_step:
            current_step.mark_completed()
            plan.move_to_next_step()
            
            return {
                "plan": plan,
                "past_steps": [(
                    "replan",
                    f"Completed step '{current_step.step}', moving to next step"
                )]
            }
        
        return {} 