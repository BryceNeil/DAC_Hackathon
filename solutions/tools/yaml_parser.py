"""
YAML Parser for ASU Tapeout Agent
=================================

Handles parsing and validation of YAML problem specifications.
"""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class YAMLParser:
    """Parses and validates YAML problem specifications"""
    
    def __init__(self):
        """Initialize YAML parser"""
        self.required_fields = [
            'description',
            'clock_period', 
            'module_signature'
        ]
        
    def parse_file(self, yaml_path: str) -> Dict[str, Any]:
        """Parse YAML file and validate structure
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Parsed YAML content
            
        Raises:
            ValueError: If YAML is invalid
        """
        try:
            path = Path(yaml_path)
            if not path.exists():
                raise ValueError(f"YAML file not found: {yaml_path}")
                
            with open(path, 'r') as f:
                content = yaml.safe_load(f)
                
            if not isinstance(content, dict):
                raise ValueError("YAML must contain a dictionary")
                
            # Validate structure
            self.validate_structure(content)
            
            return content
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing YAML: {e}")
    
    def validate_structure(self, spec: Dict[str, Any]):
        """Validate YAML specification structure
        
        Args:
            spec: Parsed YAML content
            
        Raises:
            ValueError: If structure is invalid
        """
        if not spec:
            raise ValueError("Empty specification")
            
        # Should have exactly one problem
        if len(spec) != 1:
            raise ValueError(f"Expected 1 problem, found {len(spec)}")
            
        problem_name = list(spec.keys())[0]
        problem_spec = spec[problem_name]
        
        if not isinstance(problem_spec, dict):
            raise ValueError(f"Problem {problem_name} must be a dictionary")
            
        # Check required fields
        missing_fields = []
        for field in self.required_fields:
            if field not in problem_spec:
                missing_fields.append(field)
                
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    def extract_problem_info(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from problem specification
        
        Args:
            spec: Parsed YAML content
            
        Returns:
            Dictionary with extracted information
        """
        problem_name = list(spec.keys())[0]
        problem_spec = spec[problem_name]
        
        info = {
            'name': problem_name,
            'description': problem_spec.get('description', ''),
            'clock_period': problem_spec.get('clock_period', '1.0ns'),
            'module_signature': problem_spec.get('module_signature', ''),
            'is_sequential': self._is_sequential(problem_spec),
            'has_reset': 'reset' in problem_spec.get('module_signature', ''),
            'io_ports': self._extract_io_ports(problem_spec.get('module_signature', '')),
            'problem_type': self._classify_problem(problem_spec)
        }
        
        # Add any problem-specific fields
        for key, value in problem_spec.items():
            if key not in ['description', 'clock_period', 'module_signature']:
                info[key] = value
                
        return info
    
    def _is_sequential(self, spec: Dict[str, Any]) -> bool:
        """Determine if design is sequential or combinational
        
        Args:
            spec: Problem specification
            
        Returns:
            True if sequential, False if combinational
        """
        module_sig = spec.get('module_signature', '').lower()
        description = spec.get('description', '').lower()
        
        # Check for clock signal
        if 'clk' in module_sig or 'clock' in module_sig:
            return True
            
        # Check for sequential keywords
        sequential_keywords = [
            'state', 'fsm', 'counter', 'register', 'flip-flop',
            'memory', 'ram', 'fifo', 'sequential', 'pipeline'
        ]
        
        for keyword in sequential_keywords:
            if keyword in description:
                return True
                
        return False
    
    def _extract_io_ports(self, module_signature: str) -> Dict[str, List[str]]:
        """Extract input/output ports from module signature
        
        Args:
            module_signature: Module signature string
            
        Returns:
            Dictionary with 'inputs' and 'outputs' lists
        """
        ports = {'inputs': [], 'outputs': []}
        
        # Simple parsing - can be enhanced
        if not module_signature:
            return ports
            
        # Remove module name and parentheses
        sig = module_signature.strip()
        if '(' in sig and ')' in sig:
            ports_str = sig[sig.index('(')+1:sig.rindex(')')]
            
            # Split by comma and parse each port
            for port in ports_str.split(','):
                port = port.strip()
                if port.startswith('input'):
                    port_name = port.replace('input', '').strip().split()[0]
                    ports['inputs'].append(port_name)
                elif port.startswith('output'):
                    port_name = port.replace('output', '').strip().split()[0]
                    if 'reg' in port:
                        port_name = port_name.replace('reg', '').strip()
                    ports['outputs'].append(port_name)
                    
        return ports
    
    def _classify_problem(self, spec: Dict[str, Any]) -> str:
        """Classify the problem type based on specification
        
        Args:
            spec: Problem specification
            
        Returns:
            Problem type classification
        """
        description = spec.get('description', '').lower()
        module_sig = spec.get('module_signature', '').lower()
        
        # Check for specific problem types
        if 'state machine' in description or 'fsm' in description:
            return 'state_machine'
        elif 'detector' in description or 'sequence' in description:
            return 'sequence_detector'
        elif 'counter' in description:
            return 'counter'
        elif 'multiplier' in description or 'multiply' in description:
            return 'arithmetic_multiplier'
        elif 'adder' in description or 'add' in description:
            return 'arithmetic_adder'
        elif 'dot product' in description:
            return 'dot_product'
        elif 'exponential' in description or 'exp' in description:
            return 'exponential_function'
        elif 'fir' in description or 'filter' in description:
            return 'fir_filter'
        elif 'floating' in description or 'fp16' in description:
            return 'floating_point'
        elif self._is_sequential(spec):
            return 'sequential_logic'
        else:
            return 'combinational_logic' 