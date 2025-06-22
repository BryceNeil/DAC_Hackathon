"""
State management for ASU Tapeout Agent using LangGraph
======================================================

This module defines the state structure for the agent workflow,
including execution plans, design information, and results.
"""

from typing import Annotated, List, Tuple, Optional, Union, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from datetime import datetime

# LangGraph imports for proper message handling
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class PlanStep(BaseModel):
    """Individual step in the execution plan"""
    step: str = Field(description="Description of the step to execute")
    agent: str = Field(description="Which agent should execute this step")
    dependencies: List[str] = Field(default=[], description="Previous steps this depends on")
    status: str = Field(default="pending", description="pending/running/completed/failed")
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    
    def mark_running(self):
        """Mark step as running"""
        self.status = "running"
        self.start_time = datetime.now()
    
    def mark_completed(self):
        """Mark step as completed"""
        self.status = "completed"
        self.end_time = datetime.now()
    
    def mark_failed(self, error: str):
        """Mark step as failed with error message"""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error


class DesignPlan(BaseModel):
    """Overall plan for the tapeout flow"""
    steps: List[PlanStep] = Field(description="Ordered list of steps to execute")
    current_step: int = Field(default=0, description="Index of current step")
    design_complexity: str = Field(default="medium", description="low/medium/high complexity")
    estimated_runtime: Optional[float] = Field(default=None, description="Estimated runtime in minutes")
    
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step to execute"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def move_to_next_step(self):
        """Move to the next step in the plan"""
        if self.current_step < len(self.steps):
            self.current_step += 1


class DesignMetrics(BaseModel):
    """Metrics from the physical design"""
    worst_negative_slack: Optional[float] = Field(default=None, description="WNS in ns")
    total_negative_slack: Optional[float] = Field(default=None, description="TNS in ns")
    area: Optional[float] = Field(default=None, description="Total area in um^2")
    power: Optional[float] = Field(default=None, description="Total power in mW")
    utilization: Optional[float] = Field(default=None, description="Core utilization percentage")
    congestion: Optional[float] = Field(default=None, description="Routing congestion percentage")


class TapeoutState(TypedDict):
    """Main state for the ASU Tapeout Agent - LangGraph Compatible"""
    # LangGraph message flow (REQUIRED)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Input
    input: str                                          # Original YAML file path
    problem_spec: dict                                  # Parsed YAML specification
    problem_name: Optional[str]                         # Extracted problem name
    
    # Planning
    plan: Optional[DesignPlan]                          # Execution plan
    
    # Execution history
    past_steps: Annotated[List[Tuple], operator.add]   # Completed steps with results
    
    # Generated artifacts
    rtl_code: Optional[str]                            # Generated RTL
    sdc_constraints: Optional[str]                     # Generated SDC
    
    # Verification results
    verification_results: Optional[dict]               # Verification outcomes
    
    # Physical design results
    physical_results: Optional[dict]                   # OpenROAD results
    
    # Final outputs
    final_response: Optional[str]                      # Final output path
    odb_file_path: Optional[str]                      # Path to generated ODB
    
    # Error tracking
    errors: Annotated[List[str], operator.add]        # Accumulated errors
    
    # Quality metrics
    metrics: Optional[DesignMetrics]                   # Design quality metrics
    
    # Workflow control
    current_step: str                                  # Current workflow step
    
    # Metadata
    start_time: Optional[datetime]                     # Workflow start time
    end_time: Optional[datetime]                       # Workflow end time 