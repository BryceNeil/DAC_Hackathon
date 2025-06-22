"""
Error handling and recovery strategies for ASU Tapeout Agent
===========================================================

Provides error classification, recovery strategies, and retry logic.
"""

from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import traceback
import re
from dataclasses import dataclass
from utils.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from tools.llm_interface import LLMInterface

logger = get_logger("ErrorHandler")


class ErrorType(Enum):
    """Classification of error types"""
    SYNTAX_ERROR = "syntax_error"
    TIMING_VIOLATION = "timing_violation"
    COMPILATION_ERROR = "compilation_error"
    SIMULATION_MISMATCH = "simulation_mismatch"
    TOOL_NOT_FOUND = "tool_not_found"
    FILE_NOT_FOUND = "file_not_found"
    LLM_ERROR = "llm_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_type: ErrorType
    message: str
    source: str  # Which agent/tool generated the error
    details: Optional[Dict[str, Any]] = None
    stacktrace: Optional[str] = None
    
    def __str__(self):
        return f"[{self.error_type.value}] {self.source}: {self.message}"


class RecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def can_handle(self, error: ErrorContext) -> bool:
        """Check if this strategy can handle the error"""
        raise NotImplementedError
    
    def recover(self, error: ErrorContext) -> Dict[str, Any]:
        """Attempt to recover from the error"""
        raise NotImplementedError


class SyntaxErrorRecovery(RecoveryStrategy):
    """Recovery strategy for Verilog syntax errors"""
    
    def can_handle(self, error: ErrorContext) -> bool:
        return error.error_type == ErrorType.SYNTAX_ERROR
    
    def recover(self, error: ErrorContext) -> Dict[str, Any]:
        """Analyze syntax error and suggest fixes"""
        suggestions = []
        error_msg = error.message.lower()
        
        # Common Verilog syntax error patterns
        if "unexpected" in error_msg:
            suggestions.append("Check for missing semicolons or mismatched brackets")
        if "undefined" in error_msg or "undeclared" in error_msg:
            suggestions.append("Check module ports and signal declarations")
        if "syntax error" in error_msg and "module" in error_msg:
            suggestions.append("Verify module declaration syntax matches specification")
        
        # Extract line number if available
        line_match = re.search(r'line\s+(\d+)', error_msg)
        line_number = line_match.group(1) if line_match else None
        
        return {
            "retry": True,
            "suggestions": suggestions,
            "line_number": line_number,
            "fix_type": "syntax"
        }


class TimingViolationRecovery(RecoveryStrategy):
    """Recovery strategy for timing violations"""
    
    def can_handle(self, error: ErrorContext) -> bool:
        return error.error_type == ErrorType.TIMING_VIOLATION
    
    def recover(self, error: ErrorContext) -> Dict[str, Any]:
        """Suggest timing fixes"""
        suggestions = []
        
        if error.details:
            wns = error.details.get('worst_negative_slack', 0)
            if wns < -1.0:
                suggestions.append("Consider pipelining the design")
                suggestions.append("Reduce logic levels in critical paths")
            elif wns < -0.5:
                suggestions.append("Try increasing clock period slightly")
                suggestions.append("Add register retiming")
        
        return {
            "retry": True,
            "suggestions": suggestions,
            "fix_type": "timing",
            "adjust_constraints": True
        }


class CompilationErrorRecovery(RecoveryStrategy):
    """Recovery for compilation errors"""
    
    def can_handle(self, error: ErrorContext) -> bool:
        return error.error_type == ErrorType.COMPILATION_ERROR
    
    def recover(self, error: ErrorContext) -> Dict[str, Any]:
        """Analyze compilation errors"""
        error_msg = error.message
        suggestions = []
        
        # Parse common iverilog/verilator errors
        if "port" in error_msg.lower():
            suggestions.append("Check port connections and widths")
        if "width" in error_msg.lower() or "size" in error_msg.lower():
            suggestions.append("Verify signal bit widths match")
        if "multiple drivers" in error_msg.lower():
            suggestions.append("Check for conflicting assignments to signals")
        
        return {
            "retry": True,
            "suggestions": suggestions,
            "fix_type": "compilation"
        }


class ErrorHandler:
    """Main error handling system"""
    
    def __init__(self):
        self.recovery_strategies = [
            SyntaxErrorRecovery(),
            TimingViolationRecovery(),
            CompilationErrorRecovery()
        ]
        self.error_history: List[ErrorContext] = []
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = 3
    
    def classify_error(self, exception: Exception, source: str) -> ErrorContext:
        """Classify an exception into an error type"""
        error_msg = str(exception)
        error_type = ErrorType.UNKNOWN
        
        # Pattern matching for error classification
        if isinstance(exception, FileNotFoundError):
            error_type = ErrorType.FILE_NOT_FOUND
        elif isinstance(exception, TimeoutError):
            error_type = ErrorType.TIMEOUT
        elif "syntax" in error_msg.lower():
            error_type = ErrorType.SYNTAX_ERROR
        elif "timing" in error_msg.lower() or "slack" in error_msg.lower():
            error_type = ErrorType.TIMING_VIOLATION
        elif any(tool in error_msg.lower() for tool in ['iverilog', 'verilator', 'yosys']):
            error_type = ErrorType.COMPILATION_ERROR
        elif "simulation" in error_msg.lower() or "mismatch" in error_msg.lower():
            error_type = ErrorType.SIMULATION_MISMATCH
        
        return ErrorContext(
            error_type=error_type,
            message=error_msg,
            source=source,
            stacktrace=traceback.format_exc()
        )
    
    def handle_error(self, error: ErrorContext) -> Optional[Dict[str, Any]]:
        """Handle an error and attempt recovery"""
        logger.error(f"Error occurred: {error}")
        self.error_history.append(error)
        
        # Check retry count
        error_key = f"{error.source}_{error.error_type.value}"
        self.retry_counts[error_key] = self.retry_counts.get(error_key, 0) + 1
        
        if self.retry_counts[error_key] > self.max_retries:
            logger.error(f"Max retries exceeded for {error_key}")
            return None
        
        # Find applicable recovery strategy
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error):
                logger.info(f"Applying recovery strategy: {strategy.__class__.__name__}")
                return strategy.recover(error)
        
        logger.warning("No recovery strategy found for error")
        return None
    
    def wrap_with_retry(self, func: Callable, source: str, **kwargs) -> Any:
        """Wrap a function with error handling and retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(**kwargs)
            except Exception as e:
                error = self.classify_error(e, source)
                recovery = self.handle_error(error)
                
                if not recovery or not recovery.get('retry', False):
                    raise
                
                # Log recovery attempt
                logger.info(f"Retry attempt {attempt + 1} with suggestions: {recovery.get('suggestions', [])}")
                
                # Pass recovery info to next attempt
                if 'fix_type' in recovery:
                    kwargs['recovery_hint'] = recovery
                
                last_error = e
        
        # All retries exhausted
        if last_error:
            raise last_error
    
    def get_error_summary(self) -> str:
        """Get summary of all errors encountered"""
        if not self.error_history:
            return "No errors encountered"
        
        summary = f"Total errors: {len(self.error_history)}\n"
        error_counts = {}
        
        for error in self.error_history:
            error_counts[error.error_type.value] = error_counts.get(error.error_type.value, 0) + 1
        
        for error_type, count in error_counts.items():
            summary += f"  {error_type}: {count}\n"
        
        return summary


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler 


class ErrorRecoveryAgent:
    """Handles automatic error recovery for tool failures in autonomous mode"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        """Initialize error recovery agent
        
        Args:
            llm_model: LLM model to use for error analysis
        """
        # Use appropriate LLM provider based on model name
        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(model=llm_model, temperature=0)
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        self.llm_interface = LLMInterface()
        self.max_retries = 3
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> Dict[str, Any]:
        """Load common error patterns for quick classification"""
        return {
            "verilator": {
                "syntax": [
                    {
                        "pattern": r"syntax error, unexpected (\w+)",
                        "fix": "Check for missing semicolons or incorrect keywords"
                    },
                    {
                        "pattern": r"Variable definition not modifiable",
                        "fix": "Use 'reg' for signals assigned in always blocks"
                    }
                ],
                "warnings": [
                    {
                        "pattern": r"Signal is not driven",
                        "fix": "Ensure all outputs are assigned values"
                    }
                ]
            },
            "iverilog": {
                "compilation": [
                    {
                        "pattern": r"Unknown module type: (\w+)",
                        "fix": "Check module name matches instantiation"
                    },
                    {
                        "pattern": r"Port (\w+) is not defined",
                        "fix": "Verify all ports in module signature are used"
                    }
                ]
            },
            "sta": {
                "timing": [
                    {
                        "pattern": r"Setup violation.*slack \(VIOLATED\): (-?\d+\.?\d*)",
                        "fix": "Increase clock period or add pipeline stages"
                    },
                    {
                        "pattern": r"No clock found",
                        "fix": "Ensure clock is properly defined in SDC"
                    }
                ]
            }
        }
    
    async def handle_tool_error(self, state: Dict[str, Any], error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Main error handling logic that routes to appropriate fix
        
        Args:
            state: Current state of the workflow
            error_info: Information about the error
            
        Returns:
            Dictionary with fix updates to the state
        """
        tool_name = error_info.get("tool", "unknown")
        error_message = error_info.get("error", "")
        context = error_info.get("context", {})
        
        # Analyze error type
        error_type = self._classify_error(error_message, tool_name)
        
        # Apply appropriate fix strategy
        if error_type == "syntax_error":
            return await self._fix_syntax_error(state, error_message, context)
        elif error_type == "timing_violation":
            return await self._fix_timing_violation(state, error_message, context)
        elif error_type == "compilation_error":
            return await self._fix_compilation_error(state, error_message, context)
        else:
            return await self._llm_analyze_and_fix(state, error_message, context)
    
    def _classify_error(self, error_message: str, tool_name: str) -> str:
        """Classify error type based on patterns
        
        Args:
            error_message: The error message
            tool_name: Name of the tool that failed
            
        Returns:
            Error type classification
        """
        error_lower = error_message.lower()
        
        # Verilator/syntax errors
        if any(keyword in error_lower for keyword in ["syntax", "unexpected", "parse error", "undeclared"]):
            return "syntax_error"
        
        # Timing errors from STA
        if any(keyword in error_lower for keyword in ["setup violation", "hold violation", "wns", "tns"]):
            return "timing_violation"
        
        # Compilation errors
        if any(keyword in error_lower for keyword in ["undefined reference", "module not found", "port mismatch"]):
            return "compilation_error"
        
        return "unknown"
    
    async def _fix_syntax_error(self, state: Dict[str, Any], error: str, context: Dict) -> Dict[str, Any]:
        """Fix RTL syntax errors automatically
        
        Args:
            state: Current state
            error: Error message
            context: Additional context
            
        Returns:
            State updates with fixed RTL
        """
        rtl_code = state.get("rtl_code", "")
        
        # Extract error details
        fix_prompt = f"""Fix this Verilog syntax error:
        
Error: {error}

Current RTL:
```verilog
{rtl_code}
```

Provide the corrected RTL code that fixes the syntax error. Return only the complete corrected Verilog code."""

        response = await self.llm.ainvoke(fix_prompt)
        fixed_rtl = self._extract_verilog_code(response.content)
        
        return {
            "rtl_code": fixed_rtl,
            "confidence_scores": {"syntax_fix": 0.85},
            "messages": [AIMessage(content=f"Auto-fixed syntax error: {error[:50]}...")]
        }
    
    async def _fix_timing_violation(self, state: Dict[str, Any], error: str, context: Dict) -> Dict[str, Any]:
        """Fix timing violations by adjusting constraints
        
        Args:
            state: Current state
            error: Error message
            context: Additional context
            
        Returns:
            State updates with fixed SDC
        """
        sdc_constraints = state.get("sdc_constraints", "")
        timing_report = context.get("timing_report", "")
        
        fix_prompt = f"""Fix timing violations by adjusting SDC constraints:
        
Timing Report:
{timing_report}

Current SDC:
```tcl
{sdc_constraints}
```

Provide updated SDC constraints that fix the timing violations. Return only the complete corrected SDC."""

        response = await self.llm.ainvoke(fix_prompt)
        fixed_sdc = self._extract_sdc_constraints(response.content)
        
        return {
            "sdc_constraints": fixed_sdc,
            "confidence_scores": {"timing_fix": 0.80},
            "messages": [AIMessage(content="Auto-adjusted timing constraints to fix violations")]
        }
    
    async def _fix_compilation_error(self, state: Dict[str, Any], error: str, context: Dict) -> Dict[str, Any]:
        """Fix compilation errors in RTL
        
        Args:
            state: Current state
            error: Error message
            context: Additional context
            
        Returns:
            State updates with fixed RTL
        """
        rtl_code = state.get("rtl_code", "")
        testbench = context.get("testbench", "")
        
        fix_prompt = f"""Fix this Verilog compilation error:
        
Error: {error}

RTL Code:
```verilog
{rtl_code}
```

{f"Testbench expecting:" if testbench else ""}
{testbench[:500] if testbench else ""}

Provide the corrected RTL that will compile successfully. Return only the complete corrected Verilog code."""

        response = await self.llm.ainvoke(fix_prompt)
        fixed_rtl = self._extract_verilog_code(response.content)
        
        return {
            "rtl_code": fixed_rtl,
            "confidence_scores": {"compilation_fix": 0.90},
            "messages": [AIMessage(content="Auto-fixed compilation error")]
        }
    
    async def _llm_analyze_and_fix(self, state: Dict[str, Any], error: str, context: Dict) -> Dict[str, Any]:
        """Use LLM to analyze and suggest fixes for unknown errors
        
        Args:
            state: Current state
            error: Error message
            context: Additional context
            
        Returns:
            State updates with analysis
        """
        analysis_prompt = f"""Analyze this error in the ASIC design flow and suggest a fix:
        
Error: {error}
Tool: {context.get('tool', 'unknown')}
Stage: {context.get('stage', 'unknown')}

Context:
- Problem: {state.get('problem_name', 'unknown')}
- Current step: {context.get('current_step', 'unknown')}

Suggest:
1. What caused this error
2. How to fix it
3. Which part of the design to modify"""

        response = await self.llm.ainvoke(analysis_prompt)
        
        # In autonomous mode, we don't escalate to human, we log and continue
        return {
            "confidence_scores": {"unknown_error_analysis": 0.60},
            "messages": [
                AIMessage(content=f"Analyzed unknown error: {error[:100]}..."),
                AIMessage(content=f"Suggestion: {response.content[:200]}...")
            ]
        }
    
    def _extract_verilog_code(self, response: str) -> str:
        """Extract Verilog code from LLM response
        
        Args:
            response: LLM response containing code
            
        Returns:
            Extracted Verilog code
        """
        # Look for code blocks
        code_match = re.search(r'```(?:verilog|systemverilog)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code block, assume entire response is code
        # Remove any explanatory text before 'module' keyword
        if 'module' in response:
            module_start = response.find('module')
            return response[module_start:].strip()
        
        return response.strip()
    
    def _extract_sdc_constraints(self, response: str) -> str:
        """Extract SDC constraints from LLM response
        
        Args:
            response: LLM response containing SDC
            
        Returns:
            Extracted SDC constraints
        """
        # Look for code blocks
        code_match = re.search(r'```(?:tcl|sdc)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code block, look for SDC commands
        lines = response.split('\n')
        sdc_lines = []
        for line in lines:
            # Common SDC commands
            if any(cmd in line for cmd in ['create_clock', 'set_', 'get_', '#']):
                sdc_lines.append(line)
        
        if sdc_lines:
            return '\n'.join(sdc_lines)
        
        return response.strip() 