"""
Constraint Generator Agent with LangChain Tool Integration
==========================================================

Generates SDC timing constraints for the physical design flow using
LangChain tools for intelligent constraint generation and validation.
"""

from typing import Dict, Any, List, Optional
import os
import re
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tools.eda_langchain_tools import (
    generate_sdc_constraints,
    analyze_timing_report
)
from tools.file_manager import FileManager
from agents.state import TapeoutState


class ConstraintGenerator:
    """Agent responsible for generating SDC timing constraints with LangChain tools"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514"):
        self.name = "ConstraintGenerator"
        self.file_manager = FileManager()
        
        # Define available LangChain tools
        self.tools = [generate_sdc_constraints, analyze_timing_report]
        
        # Create ToolNode for tool execution
        self.tool_node = ToolNode(self.tools)
        
        # Select LLM based on model name
        if "claude" in llm_model.lower():
            model = ChatAnthropic(model=llm_model)
        else:
            model = ChatOpenAI(model=llm_model)
        
        # Create ReAct agent for constraint generation
        self.agent = create_react_agent(
            model=model,
            tools=self.tools,
            prompt="""You are an expert SDC constraint generation agent for digital designs.
            
            Your responsibilities:
            1. Generate comprehensive SDC timing constraints using generate_sdc_constraints
            2. Analyze any existing timing reports to optimize constraints
            3. Create constraints that ensure timing closure in physical design
            4. Consider clock domains, I/O delays, and design complexity
            
            Always use the tools to generate valid SDC constraints. The constraints should:
            - Define all clock domains with appropriate periods
            - Set reasonable input/output delays (typically 20% of clock period)
            - Include clock uncertainty and transition times
            - Add necessary false paths and multicycle paths
            - Consider the target PDK requirements"""
        )
        
        # Standard constraint templates
        self.pdk_libraries = {
            'sky130': 'sky130_fd_sc_hd',
            'asap7': 'asap7sc7p5t',
            'nangate45': 'NangateOpenCellLibrary',
            'gf180': 'gf180mcu_fd_sc_mcu7t5v0'
        }
    
    def generate_constraints(self, analysis: Dict[str, Any], rtl_file: str, pdk: str = 'sky130') -> str:
        """
        Generate comprehensive SDC constraints
        
        Args:
            analysis: Analyzed specification data
            rtl_file: Path to RTL file for port extraction  
            pdk: Target PDK name
            
        Returns:
            Generated SDC content
        """
        # Extract port information from RTL
        port_info = self._extract_port_info(rtl_file)
        
        # Generate constraint sections
        sdc_sections = []
        
        # Header
        sdc_sections.append(self._generate_header(analysis['problem_name']))
        
        # Clock constraints
        sdc_sections.append(self._generate_clock_constraints(analysis, port_info))
        
        # I/O constraints
        sdc_sections.append(self._generate_io_constraints(analysis, port_info, pdk))
        
        # Environmental constraints
        sdc_sections.append(self._generate_environment_constraints(pdk))
        
        # Design-specific constraints
        sdc_sections.append(self._generate_design_constraints(analysis, port_info))
        
        # Optimization constraints
        sdc_sections.append(self._generate_optimization_constraints(analysis))
        
        return "\n\n".join(sdc_sections)
    
    def _extract_port_info(self, rtl_file: str) -> Dict[str, List[str]]:
        """Extract port information from RTL file"""
        
        port_info = {
            'clocks': [],
            'inputs': [],
            'outputs': [],
            'resets': []
        }
        
        try:
            with open(rtl_file, 'r') as f:
                content = f.read()
            
            # Extract ports from module declaration
            lines = content.split('\n')
            in_module = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('module '):
                    in_module = True
                elif 'endmodule' in line:
                    break
                elif in_module and ('input' in line or 'output' in line):
                    # Parse port declarations
                    port_name = self._parse_port_declaration(line)
                    if port_name:
                        if 'input' in line:
                            port_info['inputs'].append(port_name)
                            # Classify special signals
                            if 'clk' in port_name.lower():
                                port_info['clocks'].append(port_name)
                            elif 'rst' in port_name.lower() or 'reset' in port_name.lower():
                                port_info['resets'].append(port_name)
                        elif 'output' in line:
                            port_info['outputs'].append(port_name)
        
        except Exception as e:
            print(f"Error extracting port info: {e}")
        
        return port_info
    
    def _parse_port_declaration(self, line: str) -> Optional[str]:
        """Parse a single port declaration line"""
        
        # Remove direction keywords and formatting
        cleaned = line.replace('input', '').replace('output', '').replace(',', '').replace(';', '').strip()
        
        # Handle different declaration styles
        if cleaned:
            parts = cleaned.split()
            if parts:
                # Get the port name (last part usually)
                return parts[-1]
        
        return None
    
    def _generate_header(self, problem_name: str) -> str:
        """Generate SDC file header"""
        
        return f"""#========================================
# SDC constraints for {problem_name}
# Generated by ASU Tapeout Agent
#========================================

# Remove any previous constraints
reset_design"""
    
    def _generate_clock_constraints(self, analysis: Dict[str, Any], port_info: Dict[str, List[str]]) -> str:
        """Generate clock-related constraints"""
        
        clock_period = analysis.get('clock_period', '1.0ns')
        period_value = float(clock_period.replace('ns', ''))
        
        constraints = ["#========================================",
                      "# Clock Constraints", 
                      "#========================================"]
        
        # Primary clock constraints
        clock_ports = port_info['clocks']
        if not clock_ports:
            # Default clock if none detected
            clock_ports = ['clk']
        
        for clock_port in clock_ports:
            constraints.append(f"# Primary clock: {clock_port}")
            constraints.append(f"create_clock -name {clock_port} -period {period_value} [get_ports {clock_port}]")
            
            # Clock uncertainty and transition time
            constraints.append(f"set_clock_uncertainty 0.1 [get_clocks {clock_port}]")
            constraints.append(f"set_clock_transition 0.05 [get_clocks {clock_port}]")
        
        # Generated clocks (if any)
        if len(clock_ports) > 1:
            constraints.append("\n# Generated clocks")
            constraints.append("# Add generated clock constraints if needed")
        
        return "\n".join(constraints)
    
    def _generate_io_constraints(self, analysis: Dict[str, Any], port_info: Dict[str, List[str]], pdk: str) -> str:
        """Generate input/output timing constraints"""
        
        clock_period = analysis.get('clock_period', '1.0ns')
        period_value = float(clock_period.replace('ns', ''))
        
        # Calculate reasonable I/O delays (typically 10-30% of clock period)
        input_delay = round(period_value * 0.2, 2)
        output_delay = round(period_value * 0.2, 2)
        
        constraints = ["#========================================",
                      "# I/O Timing Constraints",
                      "#========================================"]
        
        # Input delays
        regular_inputs = [inp for inp in port_info['inputs'] 
                         if inp not in port_info['clocks'] and inp not in port_info['resets']]
        
        if regular_inputs:
            constraints.append("# Input delays")
            constraints.append(f"set_input_delay -clock [get_clocks clk] {input_delay} [get_ports {{{' '.join(regular_inputs)}}}]")
        
        # Output delays  
        if port_info['outputs']:
            constraints.append("# Output delays")
            constraints.append(f"set_output_delay -clock [get_clocks clk] {output_delay} [get_ports {{{' '.join(port_info['outputs'])}}}]")
        
        # Driving cell constraints
        lib_name = self.pdk_libraries.get(pdk, 'sky130_fd_sc_hd')
        
        if regular_inputs:
            constraints.append("# Input driving cells")
            constraints.append(f"set_driving_cell -lib_cell {lib_name}__inv_2 [get_ports {{{' '.join(regular_inputs)}}}]")
        
        # Load constraints for outputs
        if port_info['outputs']:
            constraints.append("# Output loads")
            constraints.append(f"set_load 0.05 [get_ports {{{' '.join(port_info['outputs'])}}}]")
        
        return "\n".join(constraints)
    
    def _generate_environment_constraints(self, pdk: str) -> str:
        """Generate environmental and PVT constraints"""
        
        constraints = ["#========================================",
                      "# Environmental Constraints",
                      "#========================================"]
        
        # Operating conditions
        constraints.extend([
            "# Operating conditions",
            "set_operating_conditions -analysis_type on_chip_variation",
            "",
            "# Wire load models",
            "set_wire_load_mode top"
        ])
        
        # Temperature and voltage (PDK-specific)
        if pdk == 'sky130':
            constraints.extend([
                "# Sky130 specific conditions",
                "# Temperature: -40C to 85C",
                "# Voltage: 1.8V ±10%"
            ])
        elif pdk == 'asap7':
            constraints.extend([
                "# ASAP7 specific conditions", 
                "# Temperature: 25C",
                "# Voltage: 0.7V"
            ])
        
        return "\n".join(constraints)
    
    def _generate_design_constraints(self, analysis: Dict[str, Any], port_info: Dict[str, List[str]]) -> str:
        """Generate design-specific constraints"""
        
        constraints = ["#========================================",
                      "# Design-Specific Constraints", 
                      "#========================================"]
        
        # Reset constraints
        if port_info['resets']:
            constraints.append("# Reset constraints")
            for reset_port in port_info['resets']:
                constraints.extend([
                    f"set_false_path -from [get_ports {reset_port}]",
                    f"set_input_delay 0 [get_ports {reset_port}]"
                ])
        
        # Complexity-based constraints
        complexity = analysis.get('complexity', 'medium')
        
        if complexity == 'high':
            constraints.extend([
                "",
                "# High complexity design constraints",
                "set_max_transition 0.1 [current_design]",
                "set_max_capacitance 0.2 [current_design]",
                "set_max_fanout 16 [current_design]"
            ])
        elif complexity == 'low':
            constraints.extend([
                "",
                "# Low complexity design constraints", 
                "set_max_transition 0.2 [current_design]",
                "set_max_capacitance 0.5 [current_design]"
            ])
        
        # Feature-specific constraints
        features = analysis.get('required_features', [])
        
        if 'memory' in features:
            constraints.extend([
                "",
                "# Memory interface constraints",
                "# set_multicycle_path -setup 2 -from [get_pins mem_*/*] -to [get_pins reg_*/*]"
            ])
        
        if 'state_machine' in features:
            constraints.extend([
                "", 
                "# State machine constraints",
                "# Add state-specific timing constraints if needed"
            ])
        
        return "\n".join(constraints)
    
    def _generate_optimization_constraints(self, analysis: Dict[str, Any]) -> str:
        """Generate optimization and area constraints"""
        
        constraints = ["#========================================",
                      "# Optimization Constraints",
                      "#========================================"]
        
        # Area constraints based on complexity
        complexity = analysis.get('complexity', 'medium')
        
        if complexity == 'high':
            constraints.extend([
                "# High complexity - prioritize timing over area",
                "set_max_area 0",  # No area constraint - focus on timing
                "set_critical_range 0.1 [current_design]"
            ])
        elif complexity == 'low':
            constraints.extend([
                "# Low complexity - balance timing and area", 
                "set_max_area 1000",  # Moderate area constraint
                "set_critical_range 0.2 [current_design]"
            ])
        else:  # medium
            constraints.extend([
                "# Medium complexity - balanced optimization",
                "set_max_area 2000",
                "set_critical_range 0.15 [current_design]"
            ])
        
        # Power optimization
        constraints.extend([
            "",
            "# Power optimization",
            "set_dynamic_optimization true",
            "set_leakage_optimization true"
        ])
        
        # Final optimization directives
        constraints.extend([
            "",
            "# Optimization directives",
            "set_fix_hold [get_clocks]",
            "set_dont_touch_network [get_clocks]",
            "set_ideal_network [get_ports clk]"
        ])
        
        return "\n".join(constraints)
    
    async def generate_sdc(self, state: TapeoutState) -> Dict[str, Any]:
        """Generate SDC constraints using LangChain tools
        
        Args:
            state: The current tapeout state
            
        Returns:
            Dict with SDC constraints and updated state
        """
        spec = state.get("problem_spec", {})
        problem_name = state.get("problem_name", list(spec.keys())[0] if spec else "design")
        rtl_code = state.get("rtl_code", "")
        verification_results = state.get("verification_results", {})
        
        # Extract design information from RTL
        port_info = {'clocks': [], 'inputs': [], 'outputs': []}
        if rtl_code:
            rtl_file = f"/tmp/{problem_name}_for_sdc.v"
            with open(rtl_file, 'w') as f:
                f.write(rtl_code)
            port_info = self._extract_port_info(rtl_file)
        
        # Get timing requirements from spec
        clock_period = 1.0  # Default 1ns
        if spec:
            problem_spec = spec.get(problem_name, {})
            clock_period_str = problem_spec.get('clock_period', '1.0ns')
            clock_period = float(clock_period_str.replace('ns', ''))
        
        # Identify clock signals
        clock_names = port_info.get('clocks', [])
        if not clock_names:
            clock_names = ['clk']  # Default
        
        # Filter out clocks and resets from inputs
        regular_inputs = [p for p in port_info.get('inputs', []) 
                         if p not in clock_names and 'rst' not in p.lower() and 'reset' not in p.lower()]
        
        # Create task for agent
        task = f"""Generate comprehensive SDC timing constraints for {problem_name}:
        
        Design Information:
        - Clock signals: {', '.join(clock_names)}
        - Clock period: {clock_period} ns
        - Input ports: {', '.join(regular_inputs)}
        - Output ports: {', '.join(port_info.get('outputs', []))}
        
        Use the generate_sdc_constraints tool to create a complete SDC file with:
        1. Clock definitions with the specified period
        2. Input and output delays (use 20% of clock period)
        3. Clock uncertainty (use 5% of clock period)
        4. All necessary timing constraints
        
        The main clock name is: {clock_names[0]}
        Clock period: {clock_period}
        Input ports list: {regular_inputs}
        Output ports list: {port_info.get('outputs', [])}
        
        Generate the SDC constraints now."""
        
        try:
            # Run agent with tools
            result = await self.agent.ainvoke({"messages": [("user", task)]})
            
            # Extract SDC content from agent's tool calls
            sdc_content = None
            sdc_file_path = None
            
            for message in result.get("messages", []):
                # Check if this is a tool result message
                if hasattr(message, "name") and message.name == "generate_sdc_constraints":
                    try:
                        import json
                        tool_result = json.loads(message.content)
                        if tool_result.get("success"):
                            sdc_content = tool_result.get("sdc_content", "")
                            sdc_file_path = tool_result.get("sdc_file", "")
                    except:
                        pass
            
            # Fallback to manual generation if tool wasn't used properly
            if not sdc_content:
                sdc_content = self.generate_constraints(
                    {'problem_name': problem_name, 'clock_period': f'{clock_period}ns', 'complexity': 'medium'},
                    rtl_file if rtl_code else "",
                    'sky130'
                )
            
            return {
                "sdc_constraints": sdc_content,
                "sdc_file_path": sdc_file_path,
                "past_steps": [("constraint_generation", "Generated SDC constraints using LangChain tools")]
            }
            
        except Exception as e:
            # Fallback to traditional generation
            sdc_content = self.generate_constraints(
                {'problem_name': problem_name, 'clock_period': f'{clock_period}ns', 'complexity': 'medium'},
                rtl_file if rtl_code else "",
                'sky130'
            )
            
            return {
                "sdc_constraints": sdc_content,
                "past_steps": [("constraint_generation", f"Generated SDC constraints (fallback due to: {str(e)})")]
            } 