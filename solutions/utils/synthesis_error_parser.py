"""
Synthesis Error Parser
======================

Parses Yosys synthesis errors and provides specific feedback for RTL fixes.
"""

import re
from typing import Dict, List, Tuple, Optional


class SynthesisErrorParser:
    """Parse and categorize synthesis errors from Yosys"""
    
    def __init__(self):
        # Common error patterns and their fixes
        self.error_patterns = {
            # SystemVerilog parameter syntax errors
            r"parameter\s+int": {
                "category": "systemverilog_syntax",
                "fix": "Remove 'int' keyword from parameters. Use 'parameter N = 8' instead of 'parameter int N = 8'",
                "severity": "high"
            },
            r"parameter\s+logic": {
                "category": "systemverilog_syntax", 
                "fix": "Remove 'logic' keyword from parameters. Use standard Verilog parameter syntax",
                "severity": "high"
            },
            # Signal declaration errors
            r"logic\s+\[": {
                "category": "systemverilog_syntax",
                "fix": "Replace 'logic' with 'wire' or 'reg' for Verilog compatibility",
                "severity": "high"
            },
            r"Unknown identifier.*rst_n": {
                "category": "undefined_signal",
                "fix": "Signal 'rst_n' is undefined. Either use 'rst' or properly declare 'rst_n'",
                "severity": "high"
            },
            r"Unknown identifier.*'(\w+)'": {
                "category": "undefined_signal",
                "fix": "Signal '{signal}' is undefined. Add proper signal declaration",
                "severity": "high"
            },
            # Module hierarchy errors
            r"Module.*not found": {
                "category": "missing_module",
                "fix": "Module not found. Ensure all instantiated modules are defined",
                "severity": "high"
            },
            r"ERROR.*hierarchy": {
                "category": "hierarchy_error",
                "fix": "Module hierarchy error. Check module instantiations and definitions",
                "severity": "high"
            },
            # Empty module errors
            r"Warning.*contains.*no.*cells": {
                "category": "empty_module",
                "fix": "Module has no logic. Implement the required functionality",
                "severity": "medium"
            },
            r"Warning.*empty.*module": {
                "category": "empty_module",
                "fix": "Module is empty. Add functional logic based on specification",
                "severity": "medium"
            },
            # Syntax errors
            r"ERROR.*syntax error": {
                "category": "syntax_error",
                "fix": "Verilog syntax error. Check module structure and signal declarations",
                "severity": "high"
            },
            r"ERROR.*unexpected.*token": {
                "category": "syntax_error",
                "fix": "Unexpected token in Verilog. Check for SystemVerilog constructs",
                "severity": "high"
            },
            # Port mismatch errors
            r"Port.*not found": {
                "category": "port_mismatch",
                "fix": "Port mismatch in module instantiation. Check port names and connections",
                "severity": "high"
            },
            # Packed array errors (SystemVerilog)
            r"ERROR.*packed.*array": {
                "category": "systemverilog_syntax",
                "fix": "Packed arrays not supported. Convert [N-1:0][WIDTH-1:0] to [N*WIDTH-1:0]",
                "severity": "high"
            }
        }
    
    def parse_synthesis_log(self, synthesis_log: str) -> Dict[str, any]:
        """Parse synthesis log and extract errors with categorization
        
        Args:
            synthesis_log: The synthesis log output from Yosys
            
        Returns:
            Dict containing parsed errors and suggested fixes
        """
        errors = []
        warnings = []
        
        # Split log into lines
        lines = synthesis_log.split('\n')
        
        # Track if synthesis failed
        synthesis_failed = False
        
        # Look for explicit errors
        for i, line in enumerate(lines):
            # Check for ERROR lines
            if 'ERROR' in line:
                synthesis_failed = True
                
                # Try to match against known patterns
                error_info = self._match_error_pattern(line)
                if error_info:
                    errors.append({
                        "line": i + 1,
                        "text": line.strip(),
                        "category": error_info["category"],
                        "fix": error_info["fix"],
                        "severity": error_info["severity"]
                    })
                else:
                    # Generic error
                    errors.append({
                        "line": i + 1,
                        "text": line.strip(),
                        "category": "unknown",
                        "fix": "Unknown error type. Check RTL syntax",
                        "severity": "high"
                    })
            
            # Check for warnings that might be critical
            elif 'Warning' in line:
                warning_info = self._match_error_pattern(line)
                if warning_info and warning_info["severity"] == "medium":
                    warnings.append({
                        "line": i + 1,
                        "text": line.strip(),
                        "category": warning_info["category"],
                        "fix": warning_info["fix"],
                        "severity": warning_info["severity"]
                    })
        
        # Create summary of fixes needed
        fix_summary = self._create_fix_summary(errors, warnings)
        
        return {
            "synthesis_failed": synthesis_failed,
            "errors": errors,
            "warnings": warnings,
            "fix_summary": fix_summary,
            "has_systemverilog_issues": any(e["category"] == "systemverilog_syntax" for e in errors),
            "has_undefined_signals": any(e["category"] == "undefined_signal" for e in errors),
            "has_empty_module": any(e["category"] == "empty_module" for e in errors + warnings)
        }
    
    def _match_error_pattern(self, line: str) -> Optional[Dict[str, str]]:
        """Match error line against known patterns
        
        Args:
            line: Error or warning line from log
            
        Returns:
            Matched error info or None
        """
        for pattern, info in self.error_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Create a copy of info to avoid modifying original
                error_info = info.copy()
                
                # Handle captured groups (like signal names)
                if match.groups():
                    # Replace placeholders in fix message
                    fix = error_info["fix"]
                    for i, group in enumerate(match.groups(), 1):
                        fix = fix.replace(f"{{signal}}", group)
                        fix = fix.replace(f"{{{i}}}", group)
                    error_info["fix"] = fix
                
                return error_info
        
        return None
    
    def _create_fix_summary(self, errors: List[Dict], warnings: List[Dict]) -> str:
        """Create a summary of all fixes needed
        
        Args:
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            Summary string of fixes
        """
        fixes = []
        
        # Group by category
        categories = {}
        for error in errors + warnings:
            cat = error["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(error["fix"])
        
        # Create summary
        if "systemverilog_syntax" in categories:
            fixes.append("1. Convert SystemVerilog syntax to Verilog:")
            for fix in set(categories["systemverilog_syntax"]):
                fixes.append(f"   - {fix}")
        
        if "undefined_signal" in categories:
            fixes.append("2. Fix undefined signals:")
            for fix in set(categories["undefined_signal"]):
                fixes.append(f"   - {fix}")
        
        if "empty_module" in categories:
            fixes.append("3. Implement module functionality:")
            for fix in set(categories["empty_module"]):
                fixes.append(f"   - {fix}")
        
        if "syntax_error" in categories:
            fixes.append("4. Fix syntax errors:")
            for fix in set(categories["syntax_error"]):
                fixes.append(f"   - {fix}")
        
        return "\n".join(fixes) if fixes else "No specific fixes identified"
    
    def get_rtl_fix_instructions(self, parsed_errors: Dict[str, any]) -> str:
        """Generate specific RTL fix instructions based on parsed errors
        
        Args:
            parsed_errors: Output from parse_synthesis_log
            
        Returns:
            Detailed instructions for fixing RTL
        """
        instructions = []
        
        if parsed_errors["has_systemverilog_issues"]:
            instructions.append("""
SystemVerilog Syntax Fixes Required:
- Remove 'parameter int' and use 'parameter' only
- Replace 'logic' with 'wire' or 'reg'
- Convert packed arrays [N-1:0][WIDTH-1:0] to flattened [N*WIDTH-1:0]
- Remove any SystemVerilog-specific constructs
""")
        
        if parsed_errors["has_undefined_signals"]:
            undefined_signals = []
            for error in parsed_errors["errors"]:
                if error["category"] == "undefined_signal":
                    # Extract signal name from fix message
                    match = re.search(r"Signal '(\w+)'", error["fix"])
                    if match:
                        undefined_signals.append(match.group(1))
            
            if undefined_signals:
                instructions.append(f"""
Undefined Signals to Fix:
- Add proper declarations for: {', '.join(set(undefined_signals))}
- Check if 'rst_n' should be 'rst' or properly declared
- Ensure all used signals are declared with correct width
""")
        
        if parsed_errors["has_empty_module"]:
            instructions.append("""
Empty Module Implementation Required:
- Add functional logic based on the specification
- Implement required state machines or combinational logic
- Ensure module has actual synthesizable content
""")
        
        return "\n".join(instructions) if instructions else "No specific RTL fixes required" 