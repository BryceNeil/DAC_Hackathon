"""
Specification Analyzer Agent
============================

Analyzes YAML specifications to extract key information for RTL generation.
LangGraph-compatible tool functions.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from .state import TapeoutState, PlanStep


@tool
def analyze_yaml_specification(yaml_content: dict, problem_name: str) -> dict:
    """
    Analyze YAML specification and extract key design information.
    
    Args:
        yaml_content: Parsed YAML specification
        problem_name: Name of the problem to analyze
        
    Returns:
        Dictionary containing analyzed specification data including ports, complexity, features, etc.
    """
    analyzer = SpecAnalyzer()
    return analyzer.analyze_specification(yaml_content, problem_name)


@tool
def create_rtl_generation_prompt(analysis: dict) -> str:
    """
    Create an optimized prompt for RTL generation based on specification analysis.
    
    Args:
        analysis: Analyzed specification data
        
    Returns:
        Formatted prompt string for LLM RTL generation
    """
    analyzer = SpecAnalyzer()
    return analyzer.create_rtl_prompt(analysis)


class SpecAnalyzer:
    """Agent responsible for analyzing YAML specifications"""
    
    def __init__(self):
        self.name = "SpecAnalyzer"
    
    def analyze_specification(self, spec: Dict[str, Any], problem_name: str) -> Dict[str, Any]:
        """
        Analyze the YAML specification and extract key information
        
        Args:
            spec: Parsed YAML specification
            problem_name: Name of the problem
            
        Returns:
            Dict containing analyzed specification data
        """
        problem_spec = spec.get(problem_name, {})
        
        analysis = {
            'problem_name': problem_name,
            'module_signature': problem_spec.get('module_signature', ''),
            'description': problem_spec.get('description', ''),
            'clock_period': problem_spec.get('clock_period', '1.0ns'),
            'inputs': self._extract_ports(problem_spec.get('module_signature', ''), 'input'),
            'outputs': self._extract_ports(problem_spec.get('module_signature', ''), 'output'),
            'complexity': self._assess_complexity(problem_spec),
            'required_features': self._extract_features(problem_spec),
            'timing_requirements': self._extract_timing(problem_spec),
        }
        
        return analysis
    
    def _extract_ports(self, module_signature: str, port_type: str) -> List[str]:
        """Extract input or output ports from module signature"""
        ports = []
        lines = module_signature.split('\n')
        
        for line in lines:
            line = line.strip()
            if port_type in line and (',' in line or ');' in line):
                # Basic port extraction - can be enhanced
                if 'input' in line:
                    parts = line.replace('input', '').replace(',', '').replace(';', '').strip()
                    if parts:
                        ports.append(parts.split()[-1])  # Get port name
                elif 'output' in line:
                    parts = line.replace('output', '').replace(',', '').replace(';', '').strip()
                    if parts:
                        ports.append(parts.split()[-1])  # Get port name
        
        return ports
    
    def _assess_complexity(self, problem_spec: Dict) -> str:
        """Assess the complexity of the design"""
        description = problem_spec.get('description', '').lower()
        
        # Simple heuristics for complexity assessment
        if any(keyword in description for keyword in ['state machine', 'fifo', 'memory', 'cache']):
            return 'high'
        elif any(keyword in description for keyword in ['counter', 'multiplexer', 'decoder']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_features(self, problem_spec: Dict) -> List[str]:
        """Extract required design features"""
        features = []
        description = problem_spec.get('description', '').lower()
        
        # Feature detection based on description
        feature_keywords = {
            'state_machine': ['state machine', 'fsm', 'sequential'],
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'counter'],
            'memory': ['memory', 'ram', 'fifo', 'buffer'],
            'control': ['control', 'enable', 'reset'],
            'data_path': ['data', 'path', 'pipeline'],
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in description for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _extract_timing(self, problem_spec: Dict) -> Dict[str, Any]:
        """Extract timing requirements"""
        timing = {
            'clock_period': problem_spec.get('clock_period', '1.0ns'),
            'setup_time': problem_spec.get('setup_time', '0.1ns'),
            'hold_time': problem_spec.get('hold_time', '0.1ns'),
        }
        
        return timing
    
    def create_rtl_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create an optimized prompt for RTL generation"""
        prompt = f"""Generate SystemVerilog RTL for the following specification:

Problem: {analysis['problem_name']}
Description: {analysis['description']}

Module Signature:
{analysis['module_signature']}

Requirements:
- Clock period: {analysis['clock_period']}
- Complexity: {analysis['complexity']}
- Required features: {', '.join(analysis['required_features'])}

Input ports: {', '.join(analysis['inputs'])}
Output ports: {', '.join(analysis['outputs'])}

Please generate complete, synthesizable SystemVerilog code that:
1. Implements all required functionality
2. Follows best coding practices
3. Includes proper reset handling
4. Is optimized for the specified timing requirements
5. Includes appropriate comments

Generate only the module implementation, starting with the module declaration.
"""
        return prompt 