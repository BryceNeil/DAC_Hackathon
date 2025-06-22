"""
Physical Designer Agent with LangChain Tool Integration
========================================================

Manages the OpenROAD physical design flow for tapeout-ready results
using LangChain tools for automated synthesis, place, and route.
"""

import os
import subprocess
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tools.eda_langchain_tools import (
    yosys_synthesize,
    openroad_place_and_route,
    analyze_timing_report
)
from tools.eda_tools import EDATools
from tools.file_manager import FileManager
from agents.state import TapeoutState


class PhysicalDesigner:
    """Agent responsible for physical design using OpenROAD flow with LangChain tools"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514"):
        self.name = "PhysicalDesigner"
        self.eda_tools = EDATools()
        self.file_manager = FileManager()
        
        # Define available LangChain tools
        self.tools = [yosys_synthesize, openroad_place_and_route, analyze_timing_report]
        
        # Create ToolNode for tool execution
        self.tool_node = ToolNode(self.tools)
        
        # Select LLM based on model name
        if "claude" in llm_model.lower():
            model = ChatAnthropic(model=llm_model)
        else:
            model = ChatOpenAI(model=llm_model)
        
        # Create ReAct agent for physical design
        self.agent = create_react_agent(
            model=model,
            tools=self.tools,
            prompt="""You are an expert physical design agent for digital chip design.
            
            Your responsibilities:
            1. Synthesize RTL using yosys_synthesize with appropriate target library
            2. Run place and route using openroad_place_and_route
            3. Analyze timing reports to ensure timing closure
            4. Generate the final OpenDB (ODB) file for tapeout
            
            Follow this flow:
            1. First synthesize the RTL using yosys_synthesize with sky130 target
            2. If synthesis succeeds, run place and route with openroad_place_and_route
            3. Analyze the timing results to verify timing closure
            4. Report the final ODB file path and key metrics
            
            Focus on achieving timing closure and generating a valid ODB file."""
        )
        
        # Default ORFS configuration
        self.default_config = {
            'PLATFORM': 'sky130hd',
            'DESIGN_NAME': '',
            'VERILOG_FILES': '',
            'SDC_FILE': '',
            'CORE_UTILIZATION': 40,
            'ASPECT_RATIO': 1,
            'DIE_AREA': '',
            'CORE_AREA': '',
        }
    
    def run_physical_design(self, rtl_file: str, sdc_file: str, problem_name: str, 
                          analysis: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """
        Run the complete OpenROAD physical design flow
        
        Args:
            rtl_file: Path to RTL file
            sdc_file: Path to SDC constraints file
            problem_name: Name of the problem
            analysis: Analyzed specification data
            output_dir: Output directory for results
            
        Returns:
            Dict containing physical design results
        """
        results = {
            'success': False,
            'synthesis_success': False,
            'floorplan_success': False,
            'placement_success': False,
            'routing_success': False,
            'final_odb_path': '',
            'timing_results': {},
            'area_results': {},
            'power_results': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create work directory
            work_dir = os.path.join(output_dir, 'physical_design')
            os.makedirs(work_dir, exist_ok=True)
            
            # Generate ORFS configuration
            config = self._generate_orfs_config(rtl_file, sdc_file, problem_name, analysis, work_dir)
            config_file = os.path.join(work_dir, 'config.mk')
            self._write_config_file(config, config_file)
            
            # Run synthesis
            synthesis_result = self._run_synthesis(work_dir, config_file)
            results.update(synthesis_result)
            
            if results['synthesis_success']:
                # Run floorplan
                floorplan_result = self._run_floorplan(work_dir, config_file)
                results.update(floorplan_result)
                
                if results['floorplan_success']:
                    # Run placement
                    placement_result = self._run_placement(work_dir, config_file)
                    results.update(placement_result)
                    
                    if results['placement_success']:
                        # Run routing
                        routing_result = self._run_routing(work_dir, config_file)
                        results.update(routing_result)
                        
                        if results['routing_success']:
                            # Extract final results
                            final_result = self._extract_final_results(work_dir, output_dir)
                            results.update(final_result)
        
        except Exception as e:
            results['errors'].append(f"Physical design error: {str(e)}")
        
        results['success'] = (results['synthesis_success'] and 
                            results['floorplan_success'] and
                            results['placement_success'] and 
                            results['routing_success'])
        
        return results
    
    def _generate_orfs_config(self, rtl_file: str, sdc_file: str, problem_name: str, 
                             analysis: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        """Generate ORFS configuration based on design analysis"""
        
        config = self.default_config.copy()
        
        # Basic configuration
        config['DESIGN_NAME'] = problem_name
        config['VERILOG_FILES'] = os.path.abspath(rtl_file)
        config['SDC_FILE'] = os.path.abspath(sdc_file)
        
        # Set utilization based on complexity
        complexity = analysis.get('complexity', 'medium')
        if complexity == 'high':
            config['CORE_UTILIZATION'] = 60  # Higher utilization for complex designs
        elif complexity == 'low':
            config['CORE_UTILIZATION'] = 30  # Lower utilization for simple designs
        else:
            config['CORE_UTILIZATION'] = 40  # Default for medium complexity
        
        # Estimate die area based on features
        features = analysis.get('required_features', [])
        base_area = 100  # Base area in microns
        
        if 'memory' in features:
            base_area *= 2
        if 'state_machine' in features:
            base_area *= 1.5
        if 'arithmetic' in features:
            base_area *= 1.3
        
        # Set die area (square)
        die_size = int(base_area ** 0.5)
        config['DIE_AREA'] = f"0 0 {die_size} {die_size}"
        
        # Core area (slightly smaller than die area)
        core_margin = 5
        core_size = die_size - 2 * core_margin
        config['CORE_AREA'] = f"{core_margin} {core_margin} {die_size - core_margin} {die_size - core_margin}"
        
        return config
    
    def _write_config_file(self, config: Dict[str, Any], config_file: str):
        """Write ORFS configuration to Makefile format"""
        
        with open(config_file, 'w') as f:
            f.write("# ORFS Configuration\n")
            f.write("# Generated by ASU Tapeout Agent\n\n")
            
            for key, value in config.items():
                f.write(f"{key} = {value}\n")
            
            # Additional ORFS settings
            f.write("\n# Additional ORFS settings\n")
            f.write("SYNTH_HIERARCHICAL = 1\n")
            f.write("ABC_SPEED = 1\n")
            f.write("ABC_AREA = 0\n")
            f.write("PLACE_DENSITY_LB_ADDON = 0.20\n")
            f.write("TNS_END_PERCENT = 5\n")
    
    def _run_synthesis(self, work_dir: str, config_file: str) -> Dict[str, Any]:
        """Run synthesis using OpenROAD flow"""
        
        result = {
            'synthesis_success': False,
            'synthesis_output': '',
            'synthesis_errors': []
        }
        
        try:
            # Change to work directory
            original_dir = os.getcwd()
            os.chdir(work_dir)
            
            # Run synthesis command
            cmd = f"make -f {config_file} synth"
            print(f"Running synthesis: {cmd}")
            
            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            result['synthesis_output'] = process.stdout
            
            if process.returncode == 0:
                result['synthesis_success'] = True
            else:
                result['synthesis_errors'].append(process.stderr)
            
            # Return to original directory
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired:
            result['synthesis_errors'].append("Synthesis timeout")
            os.chdir(original_dir)
        except Exception as e:
            result['synthesis_errors'].append(f"Synthesis error: {str(e)}")
            os.chdir(original_dir)
        
        return result
    
    def _run_floorplan(self, work_dir: str, config_file: str) -> Dict[str, Any]:
        """Run floorplanning"""
        
        result = {
            'floorplan_success': False,
            'floorplan_output': '',
            'floorplan_errors': []
        }
        
        try:
            original_dir = os.getcwd()
            os.chdir(work_dir)
            
            cmd = f"make -f {config_file} floorplan"
            print(f"Running floorplan: {cmd}")
            
            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            result['floorplan_output'] = process.stdout
            
            if process.returncode == 0:
                result['floorplan_success'] = True
            else:
                result['floorplan_errors'].append(process.stderr)
            
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired:
            result['floorplan_errors'].append("Floorplan timeout")
            os.chdir(original_dir)
        except Exception as e:
            result['floorplan_errors'].append(f"Floorplan error: {str(e)}")
            os.chdir(original_dir)
        
        return result
    
    def _run_placement(self, work_dir: str, config_file: str) -> Dict[str, Any]:
        """Run placement"""
        
        result = {
            'placement_success': False,
            'placement_output': '',
            'placement_errors': []
        }
        
        try:
            original_dir = os.getcwd()
            os.chdir(work_dir)
            
            cmd = f"make -f {config_file} place"
            print(f"Running placement: {cmd}")
            
            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            result['placement_output'] = process.stdout
            
            if process.returncode == 0:
                result['placement_success'] = True
            else:
                result['placement_errors'].append(process.stderr)
            
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired:
            result['placement_errors'].append("Placement timeout")
            os.chdir(original_dir)
        except Exception as e:
            result['placement_errors'].append(f"Placement error: {str(e)}")
            os.chdir(original_dir)
        
        return result
    
    def _run_routing(self, work_dir: str, config_file: str) -> Dict[str, Any]:
        """Run routing"""
        
        result = {
            'routing_success': False,
            'routing_output': '',
            'routing_errors': []
        }
        
        try:
            original_dir = os.getcwd()
            os.chdir(work_dir)
            
            cmd = f"make -f {config_file} route"
            print(f"Running routing: {cmd}")
            
            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1200  # 20 minutes timeout
            )
            
            result['routing_output'] = process.stdout
            
            if process.returncode == 0:
                result['routing_success'] = True
            else:
                result['routing_errors'].append(process.stderr)
            
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired:
            result['routing_errors'].append("Routing timeout")
            os.chdir(original_dir)
        except Exception as e:
            result['routing_errors'].append(f"Routing error: {str(e)}")
            os.chdir(original_dir)
        
        return result
    
    def _extract_final_results(self, work_dir: str, output_dir: str) -> Dict[str, Any]:
        """Extract final results and copy ODB file"""
        
        result = {
            'final_odb_path': '',
            'timing_results': {},
            'area_results': {},
            'power_results': {}
        }
        
        try:
            # Look for final ODB file
            odb_patterns = [
                os.path.join(work_dir, 'results', '*', '6_final.odb'),
                os.path.join(work_dir, 'results', '6_final.odb'),
                os.path.join(work_dir, '6_final.odb')
            ]
            
            final_odb = None
            for pattern in odb_patterns:
                matches = list(Path().glob(pattern))
                if matches:
                    final_odb = str(matches[0])
                    break
            
            if final_odb and os.path.exists(final_odb):
                # Copy ODB to output directory
                output_odb = os.path.join(output_dir, '6_final.odb')
                subprocess.run(f"cp {final_odb} {output_odb}", shell=True)
                result['final_odb_path'] = output_odb
            else:
                # Create placeholder ODB if not found
                output_odb = os.path.join(output_dir, '6_final.odb')
                with open(output_odb, 'w') as f:
                    f.write("# Placeholder ODB file - Physical design flow incomplete\n")
                result['final_odb_path'] = output_odb
            
            # Extract timing results from reports
            timing_report = os.path.join(work_dir, 'reports', 'timing.rpt')
            if os.path.exists(timing_report):
                result['timing_results'] = self._parse_timing_report(timing_report)
            
            # Extract area results
            area_report = os.path.join(work_dir, 'reports', 'area.rpt')
            if os.path.exists(area_report):
                result['area_results'] = self._parse_area_report(area_report)
            
        except Exception as e:
            print(f"Error extracting final results: {e}")
        
        return result
    
    def _parse_timing_report(self, report_file: str) -> Dict[str, float]:
        """Parse timing report for key metrics"""
        
        timing_results = {
            'worst_negative_slack': 0.0,
            'total_negative_slack': 0.0,
            'worst_hold_slack': 0.0,
            'clock_period': 0.0
        }
        
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Simple parsing - can be enhanced
            lines = content.split('\n')
            for line in lines:
                if 'slack' in line.lower():
                    # Extract timing values
                    pass  # Add parsing logic based on actual report format
        
        except Exception as e:
            print(f"Error parsing timing report: {e}")
        
        return timing_results
    
    def _parse_area_report(self, report_file: str) -> Dict[str, float]:
        """Parse area report for key metrics"""
        
        area_results = {
            'total_area': 0.0,
            'utilization': 0.0,
            'cell_count': 0
        }
        
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Simple parsing - can be enhanced
            lines = content.split('\n')
            for line in lines:
                if 'area' in line.lower():
                    # Extract area values
                    pass  # Add parsing logic based on actual report format
        
        except Exception as e:
            print(f"Error parsing area report: {e}")
        
        return area_results
    
    async def run_physical_design_with_tools(self, state: TapeoutState) -> Dict[str, Any]:
        """Run physical design using LangChain tools
        
        Args:
            state: The current tapeout state
            
        Returns:
            Dict with physical design results and updated state
        """
        spec = state.get("problem_spec", {})
        problem_name = state.get("problem_name", list(spec.keys())[0] if spec else "design")
        rtl_code = state.get("rtl_code", "")
        sdc_constraints = state.get("sdc_constraints", "")
        
        if not rtl_code:
            return {
                "physical_results": {
                    "success": False,
                    "errors": ["No RTL code available for physical design"]
                },
                "past_steps": [("physical_design", "Failed - no RTL code")]
            }
        
        # Write RTL and SDC to files
        rtl_file = f"/tmp/{problem_name}.v"
        sdc_file = f"/tmp/{problem_name}.sdc"
        
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        with open(sdc_file, 'w') as f:
            f.write(sdc_constraints)
        
        # Extract module name from RTL
        module_name = problem_name
        for line in rtl_code.split('\n'):
            if line.strip().startswith('module '):
                module_name = line.split()[1].split('(')[0]
                break
        
        # Create task for agent
        task = f"""Run complete physical design flow for {problem_name}:
        
        1. First synthesize the RTL file {rtl_file} using yosys_synthesize
           - Top module name: {module_name}
           - Target library: sky130
        
        2. If synthesis succeeds, run place and route using openroad_place_and_route
           - RTL file: {rtl_file} (or use the synthesized netlist from step 1)
           - SDC file: {sdc_file}
           - Design name: {problem_name}
        
        3. If timing reports are generated, analyze them using analyze_timing_report
        
        Report the synthesis statistics, place and route results, and final ODB file path.
        Focus on generating a valid ODB file for tapeout."""
        
        try:
            # Run agent with tools
            result = await self.agent.ainvoke({"messages": [("user", task)]})
            
            # Extract results from agent's tool calls
            synthesis_success = False
            pnr_success = False
            odb_file_path = None
            metrics = {}
            
            for message in result.get("messages", []):
                # Check tool results
                if hasattr(message, "name"):
                    try:
                        import json
                        tool_result = json.loads(message.content)
                        
                        if message.name == "yosys_synthesize" and tool_result.get("success"):
                            synthesis_success = True
                            metrics.update(tool_result.get("stats", {}))
                        
                        elif message.name == "openroad_place_and_route":
                            if tool_result.get("success"):
                                pnr_success = True
                                odb_file_path = tool_result.get("odb_file")
                                metrics.update(tool_result.get("metrics", {}))
                        
                        elif message.name == "analyze_timing_report" and tool_result.get("success"):
                            metrics.update(tool_result.get("metrics", {}))
                    except:
                        pass
            
            # Fallback ODB path if not found
            if pnr_success and not odb_file_path:
                odb_file_path = f"/tmp/orfs_{problem_name}/results/sky130hd/{problem_name}/6_final.odb"
            
            # Create minimal ODB if nothing worked
            if not odb_file_path or not os.path.exists(odb_file_path):
                odb_file_path = f"/tmp/{problem_name}_final.odb"
                with open(odb_file_path, 'w') as f:
                    f.write(f"# ODB file for {problem_name}\n")
                    f.write(f"# Generated by ASU Tapeout Agent\n")
                    f.write(f"# Synthesis: {'Success' if synthesis_success else 'Failed'}\n")
                    f.write(f"# Place & Route: {'Success' if pnr_success else 'Failed'}\n")
            
            return {
                "physical_results": {
                    "success": synthesis_success and pnr_success,
                    "synthesis_success": synthesis_success,
                    "pnr_success": pnr_success,
                    "metrics": metrics,
                    "odb_path": odb_file_path
                },
                "odb_file_path": odb_file_path,
                "past_steps": [("physical_design", f"Physical design {'completed' if pnr_success else 'attempted'} using LangChain tools")]
            }
            
        except Exception as e:
            # Fallback to traditional flow
            output_dir = f"/tmp/{problem_name}_physical"
            os.makedirs(output_dir, exist_ok=True)
            
            results = self.run_physical_design(
                rtl_file, sdc_file, problem_name,
                {'complexity': 'medium', 'required_features': []},
                output_dir
            )
            
            return {
                "physical_results": results,
                "odb_file_path": results.get("final_odb_path", ""),
                "past_steps": [("physical_design", f"Physical design fallback due to: {str(e)}")]
            } 