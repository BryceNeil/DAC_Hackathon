"""
Human Validation Agent with LangGraph Interrupts
================================================

This module implements human-in-the-loop validation using LangGraph's
interrupt capabilities for critical design reviews.
"""

from typing import Dict, Any, Optional, List, Literal
from enum import Enum
import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from .state import TapeoutState


class ReviewType(str, Enum):
    """Types of reviews that can trigger human validation"""
    RTL_REVIEW = "rtl_review"
    CONSTRAINT_REVIEW = "constraint_review"
    PHYSICAL_REVIEW = "physical_review"
    FINAL_REVIEW = "final_review"


class ValidationCriteria(BaseModel):
    """Criteria for determining if human review is needed"""
    complexity_threshold: str = Field(
        default="high",
        description="Complexity level that triggers review (low/medium/high)"
    )
    timing_margin_threshold: float = Field(
        default=0.1,
        description="Timing margin below which review is triggered (in ns)"
    )
    area_threshold: float = Field(
        default=10000.0,
        description="Area above which review is triggered (in um^2)"
    )
    error_count_threshold: int = Field(
        default=3,
        description="Number of errors/warnings that trigger review"
    )


class ReviewRequest(BaseModel):
    """Structured review request for human validation"""
    review_type: ReviewType
    artifact: str = Field(description="The artifact to review (RTL code, constraints, etc.)")
    context: Dict[str, Any] = Field(description="Additional context for the review")
    issues: List[str] = Field(default=[], description="Specific issues to address")
    suggestions: List[str] = Field(default=[], description="Suggested improvements")


class HumanValidationAgent:
    """Agent that manages human-in-the-loop validation"""
    
    def __init__(self, criteria: Optional[ValidationCriteria] = None):
        """Initialize the human validation agent
        
        Args:
            criteria: Validation criteria for triggering reviews
        """
        self.criteria = criteria or ValidationCriteria()
        self.review_history = []
    
    def needs_rtl_review(self, state: TapeoutState) -> bool:
        """Determine if RTL needs human review
        
        Args:
            state: Current tapeout state
            
        Returns:
            True if human review is needed
        """
        rtl_code = state.get("rtl_code", "")
        plan = state.get("plan")
        verification_results = state.get("verification_results", {})
        
        # Check complexity
        if plan and plan.design_complexity == "high":
            return True
        
        # Check verification results
        if verification_results:
            if not verification_results.get("passed", True):
                return True
            
            warnings = verification_results.get("warnings", [])
            if len(warnings) >= self.criteria.error_count_threshold:
                return True
        
        # Check RTL size/complexity metrics
        line_count = len(rtl_code.split("\n"))
        if line_count > 500:  # Large RTL files need review
            return True
        
        # Check for specific patterns that need review
        review_patterns = [
            "TODO",
            "FIXME", 
            "HACK",
            "assert",
            "assume",
            "cover"
        ]
        
        for pattern in review_patterns:
            if pattern in rtl_code:
                return True
        
        return False
    
    def needs_constraint_review(self, state: TapeoutState) -> bool:
        """Determine if constraints need human review
        
        Args:
            state: Current tapeout state
            
        Returns:
            True if human review is needed
        """
        sdc_constraints = state.get("sdc_constraints", "")
        physical_results = state.get("physical_results", {})
        
        # Check timing results
        if physical_results:
            timing = physical_results.get("timing", {})
            wns = timing.get("worst_negative_slack", 0)
            
            if wns < -self.criteria.timing_margin_threshold:
                return True
        
        # Check for complex constraints
        complex_patterns = [
            "set_multicycle_path",
            "set_false_path",
            "set_case_analysis",
            "set_clock_uncertainty"
        ]
        
        for pattern in complex_patterns:
            if pattern in sdc_constraints:
                return True
        
        return False
    
    def needs_physical_review(self, state: TapeoutState) -> bool:
        """Determine if physical design needs human review
        
        Args:
            state: Current tapeout state
            
        Returns:
            True if human review is needed
        """
        physical_results = state.get("physical_results", {})
        
        if not physical_results:
            return False
        
        # Check area
        metrics = physical_results.get("metrics", {})
        area = metrics.get("area", 0)
        
        if area > self.criteria.area_threshold:
            return True
        
        # Check utilization
        utilization = metrics.get("utilization", 0)
        if utilization > 0.8:  # High utilization needs review
            return True
        
        # Check routing congestion
        congestion = metrics.get("congestion", 0)
        if congestion > 0.9:  # High congestion needs review
            return True
        
        return False
    
    def create_review_request(
        self,
        review_type: ReviewType,
        state: TapeoutState
    ) -> ReviewRequest:
        """Create a structured review request
        
        Args:
            review_type: Type of review needed
            state: Current tapeout state
            
        Returns:
            Structured review request
        """
        if review_type == ReviewType.RTL_REVIEW:
            return self._create_rtl_review_request(state)
        elif review_type == ReviewType.CONSTRAINT_REVIEW:
            return self._create_constraint_review_request(state)
        elif review_type == ReviewType.PHYSICAL_REVIEW:
            return self._create_physical_review_request(state)
        else:
            return self._create_final_review_request(state)
    
    def _create_rtl_review_request(self, state: TapeoutState) -> ReviewRequest:
        """Create RTL review request"""
        rtl_code = state.get("rtl_code", "")
        verification_results = state.get("verification_results", {})
        
        issues = []
        suggestions = []
        
        # Identify issues
        if verification_results and not verification_results.get("passed", True):
            issues.append("Verification failed - please review functionality")
        
        warnings = verification_results.get("warnings", [])
        issues.extend(warnings[:5])  # Include first 5 warnings
        
        # Add suggestions
        if "TODO" in rtl_code or "FIXME" in rtl_code:
            suggestions.append("Complete TODO/FIXME items in the code")
        
        if len(rtl_code.split("\n")) > 500:
            suggestions.append("Consider modularizing large RTL blocks")
        
        return ReviewRequest(
            review_type=ReviewType.RTL_REVIEW,
            artifact=rtl_code,
            context={
                "problem_name": state.get("problem_name"),
                "problem_spec": state.get("problem_spec"),
                "verification_results": verification_results
            },
            issues=issues,
            suggestions=suggestions
        )
    
    def _create_constraint_review_request(self, state: TapeoutState) -> ReviewRequest:
        """Create constraint review request"""
        sdc_constraints = state.get("sdc_constraints", "")
        physical_results = state.get("physical_results", {})
        
        issues = []
        suggestions = []
        
        # Check timing issues
        if physical_results:
            timing = physical_results.get("timing", {})
            wns = timing.get("worst_negative_slack", 0)
            
            if wns < 0:
                issues.append(f"Negative slack detected: WNS = {wns} ns")
                suggestions.append("Review timing constraints and consider relaxing if appropriate")
        
        return ReviewRequest(
            review_type=ReviewType.CONSTRAINT_REVIEW,
            artifact=sdc_constraints,
            context={
                "problem_name": state.get("problem_name"),
                "physical_results": physical_results
            },
            issues=issues,
            suggestions=suggestions
        )
    
    def _create_physical_review_request(self, state: TapeoutState) -> ReviewRequest:
        """Create physical design review request"""
        physical_results = state.get("physical_results", {})
        
        issues = []
        suggestions = []
        
        if physical_results:
            metrics = physical_results.get("metrics", {})
            
            # Check various metrics
            area = metrics.get("area", 0)
            if area > self.criteria.area_threshold:
                issues.append(f"Large area: {area} um^2")
                suggestions.append("Review design for optimization opportunities")
            
            utilization = metrics.get("utilization", 0)
            if utilization > 0.8:
                issues.append(f"High utilization: {utilization * 100:.1f}%")
                suggestions.append("Consider increasing core area or optimizing placement")
        
        return ReviewRequest(
            review_type=ReviewType.PHYSICAL_REVIEW,
            artifact=json.dumps(physical_results, indent=2),
            context={
                "problem_name": state.get("problem_name"),
                "odb_path": state.get("odb_file_path")
            },
            issues=issues,
            suggestions=suggestions
        )
    
    def _create_final_review_request(self, state: TapeoutState) -> ReviewRequest:
        """Create final review request"""
        return ReviewRequest(
            review_type=ReviewType.FINAL_REVIEW,
            artifact="Complete design ready for final review",
            context={
                "problem_name": state.get("problem_name"),
                "rtl_generated": bool(state.get("rtl_code")),
                "verification_passed": state.get("verification_results", {}).get("passed", False),
                "physical_complete": bool(state.get("physical_results")),
                "odb_path": state.get("odb_file_path")
            },
            issues=[],
            suggestions=["Review all artifacts before final approval"]
        )
    
    def format_interrupt_message(self, review_request: ReviewRequest) -> str:
        """Format review request for human display
        
        Args:
            review_request: The review request to format
            
        Returns:
            Formatted message for human reviewer
        """
        message = f"""
## Human Review Required: {review_request.review_type.value}

### Issues Identified:
"""
        
        if review_request.issues:
            for issue in review_request.issues:
                message += f"- {issue}\n"
        else:
            message += "- No specific issues, general review requested\n"
        
        message += "\n### Suggestions:\n"
        
        if review_request.suggestions:
            for suggestion in review_request.suggestions:
                message += f"- {suggestion}\n"
        else:
            message += "- Review artifact for correctness and completeness\n"
        
        message += "\n### Artifact to Review:\n"
        message += "```\n"
        message += review_request.artifact[:1000]  # Show first 1000 chars
        if len(review_request.artifact) > 1000:
            message += "\n... (truncated for display)\n"
        message += "\n```\n"
        
        message += "\n### Available Actions:\n"
        message += "- **Accept**: Approve the artifact as-is\n"
        message += "- **Edit**: Modify the artifact\n"
        message += "- **Reject**: Request regeneration with feedback\n"
        message += "- **Skip**: Continue without review (not recommended)\n"
        
        return message
    
    def process_human_response(
        self,
        response_type: Literal["accept", "edit", "reject", "skip"],
        feedback: Optional[str] = None,
        edited_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process human response to review request
        
        Args:
            response_type: Type of response
            feedback: Optional feedback message
            edited_content: Optional edited content (for edit responses)
            
        Returns:
            State updates based on response
        """
        state_updates = {
            "past_steps": [(
                "human_validation",
                f"Human review completed: {response_type}"
            )]
        }
        
        if response_type == "accept":
            # No changes needed
            pass
        
        elif response_type == "edit" and edited_content:
            # Update the appropriate artifact
            # This would need to determine which artifact was edited
            # For now, assume it's RTL
            state_updates["rtl_code"] = edited_content
            
        elif response_type == "reject":
            # Add error to trigger regeneration
            error_msg = f"Human review rejected artifact"
            if feedback:
                error_msg += f": {feedback}"
            state_updates["errors"] = [error_msg]
            
        elif response_type == "skip":
            # Log that review was skipped
            state_updates["past_steps"].append((
                "human_validation",
                "Human review skipped - proceeding with caution"
            ))
        
        # Record review in history
        self.review_history.append({
            "response_type": response_type,
            "feedback": feedback,
            "timestamp": str(datetime.now())
        })
        
        return state_updates 