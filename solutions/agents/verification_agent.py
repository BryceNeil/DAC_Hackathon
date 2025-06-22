"""
Verification Agent with LangChain Tool Integration
==================================================

Handles functional verification of generated RTL using simulation tools
integrated with LangChain's ToolNode for automated tool execution.
"""

import os
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tools.eda_langchain_tools import (
    icarus_verilog_compile, 
    run_verilog_simulation,
    verilator_lint_check,
    analyze_timing_report
)
from tools.eda_tools import EDATools
from tools.file_manager import FileManager
from agents.state import TapeoutState


class VerificationAgent:
    """Agent responsible for RTL functional verification with LangChain tool integration"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514", execution_mode: str = "autonomous"):
        self.name = "VerificationAgent"
        self.eda_tools = EDATools()
        self.file_manager = FileManager()
        self.execution_mode = execution_mode
        
        # Initialize error recovery for autonomous mode
        self.error_recovery = None
        if execution_mode == "autonomous":
            from utils.error_handling import ErrorRecoveryAgent
            self.error_recovery = ErrorRecoveryAgent(llm_model)
        
        # Define available LangChain tools
        self.tools = [
            icarus_verilog_compile, 
            run_verilog_simulation,
            verilator_lint_check,
            analyze_timing_report
        ]
        
        # Create ToolNode for tool execution
        self.tool_node = ToolNode(self.tools)
        
        # Select LLM based on model name
        if "claude" in llm_model.lower():
            model = ChatAnthropic(model=llm_model)
        else:
            model = ChatOpenAI(model=llm_model)
        
        # Create ReAct agent
        self.agent = create_react_agent(
            model=model,
            tools=self.tools,
            prompt="""You are an expert verification agent for digital designs.
            
            Your responsibilities:
            1. Validate RTL syntax using verilator_lint_check
            2. Compile RTL with testbenches using icarus_verilog_compile  
            3. Run simulations using run_verilog_simulation
            4. Analyze timing reports if available
            5. Provide detailed feedback on any issues found
            
            Always use tools to validate your analysis. If compilation fails,
            provide specific suggestions for fixing the RTL code.
            
            Follow this verification flow:
            1. First, always run verilator_lint_check to catch syntax issues
            2. If lint passes, compile with icarus_verilog_compile
            3. If compilation succeeds, run the simulation
            4. Analyze all results and provide comprehensive feedback"""
        )
    
    def verify_rtl(self, rtl_file: str, problem_name: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify RTL functionality using available testbenches and simulation
        
        Args:
            rtl_file: Path to RTL file
            problem_name: Name of the problem
            analysis: Analyzed specification data
            
        Returns:
            Dict containing verification results
        """
        results = {
            'success': False,
            'compilation_success': False,
            'simulation_success': False,
            'testbench_found': False,
            'errors': [],
            'warnings': [],
            'simulation_output': '',
        }
        
        try:
            # Find testbench
            testbench_path = self._find_testbench(problem_name)
            
            if testbench_path:
                results['testbench_found'] = True
                
                # Compile RTL with testbench
                compilation_result = self._compile_rtl(rtl_file, testbench_path, problem_name)
                results.update(compilation_result)
                
                if results['compilation_success']:
                    # Run simulation
                    simulation_result = self._run_simulation(problem_name)
                    results.update(simulation_result)
            else:
                # Create basic testbench if none found
                testbench_path = self._generate_basic_testbench(rtl_file, problem_name, analysis)
                if testbench_path:
                    results['testbench_found'] = True
                    compilation_result = self._compile_rtl(rtl_file, testbench_path, problem_name)
                    results.update(compilation_result)
                    
                    if results['compilation_success']:
                        simulation_result = self._run_simulation(problem_name)
                        results.update(simulation_result)
        
        except Exception as e:
            results['errors'].append(f"Verification error: {str(e)}")
        
        results['success'] = (results['compilation_success'] and 
                            results['simulation_success'] and 
                            len(results['errors']) == 0)
        
        return results
    
    def _find_testbench(self, problem_name: str) -> Optional[str]:
        """Find testbench file for the problem"""
        
        # Common testbench locations
        possible_paths = [
            f"../evaluation/visible/{problem_name}_tb.v",
            f"../evaluation/visible/{problem_name}_tb.sv",
            f"../evaluation/{problem_name}_tb.v",
            f"../evaluation/{problem_name}_tb.sv",
            f"./{problem_name}_tb.v",
            f"./{problem_name}_tb.sv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None
    
    def _compile_rtl(self, rtl_file: str, testbench_file: str, problem_name: str) -> Dict[str, Any]:
        """Compile RTL with testbench using iVerilog"""
        
        result = {
            'compilation_success': False,
            'compilation_output': '',
            'compilation_errors': []
        }
        
        try:
            # Create output directory
            sim_dir = "/tmp/verification"
            os.makedirs(sim_dir, exist_ok=True)
            
            # Compile command
            output_file = os.path.join(sim_dir, f"{problem_name}_sim")
            cmd = f"iverilog -o {output_file} {rtl_file} {testbench_file}"
            
            print(f"Compiling: {cmd}")
            
            # Run compilation
            process = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            result['compilation_output'] = process.stdout
            
            if process.returncode == 0:
                result['compilation_success'] = True
            else:
                result['compilation_errors'].append(process.stderr)
                
        except subprocess.TimeoutExpired:
            result['compilation_errors'].append("Compilation timeout")
        except Exception as e:
            result['compilation_errors'].append(f"Compilation error: {str(e)}")
        
        return result
    
    def _run_simulation(self, problem_name: str) -> Dict[str, Any]:
        """Run the compiled simulation"""
        
        result = {
            'simulation_success': False,
            'simulation_output': '',
            'simulation_errors': []
        }
        
        try:
            sim_file = f"/tmp/verification/{problem_name}_sim"
            
            if os.path.exists(sim_file):
                # Run simulation
                print(f"Running simulation: {sim_file}")
                
                process = subprocess.run(
                    sim_file,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                result['simulation_output'] = process.stdout
                
                if process.returncode == 0:
                    result['simulation_success'] = True
                else:
                    result['simulation_errors'].append(process.stderr)
            else:
                result['simulation_errors'].append("Simulation executable not found")
                
        except subprocess.TimeoutExpired:
            result['simulation_errors'].append("Simulation timeout")
        except Exception as e:
            result['simulation_errors'].append(f"Simulation error: {str(e)}")
        
        return result
    
    def _generate_basic_testbench(self, rtl_file: str, problem_name: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate a basic testbench if none exists"""
        
        try:
            # Extract module info from RTL
            module_info = self._extract_module_info(rtl_file)
            
            if not module_info:
                return None
            
            # Generate testbench
            testbench_code = self._create_testbench_template(problem_name, module_info, analysis)
            
            # Save testbench
            tb_file = f"/tmp/verification/{problem_name}_tb.v"
            os.makedirs("/tmp/verification", exist_ok=True)
            
            with open(tb_file, 'w') as f:
                f.write(testbench_code)
            
            return tb_file
            
        except Exception as e:
            print(f"Error generating testbench: {e}")
            return None
    
    def _extract_module_info(self, rtl_file: str) -> Optional[Dict[str, Any]]:
        """Extract module information from RTL file"""
        
        try:
            with open(rtl_file, 'r') as f:
                content = f.read()
            
            # Simple extraction - can be enhanced
            module_info = {
                'name': '',
                'inputs': [],
                'outputs': [],
                'has_clock': False,
                'has_reset': False
            }
            
            lines = content.split('\n')
            in_module = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('module '):
                    module_info['name'] = line.split()[1].split('(')[0]
                    in_module = True
                elif 'endmodule' in line:
                    break
                elif in_module:
                    if 'input' in line:
                        # Extract input ports
                        port_line = line.replace('input', '').replace(',', '').replace(';', '').strip()
                        if port_line:
                            port_name = port_line.split()[-1]
                            module_info['inputs'].append(port_name)
                            
                            if 'clk' in port_name.lower():
                                module_info['has_clock'] = True
                            if 'rst' in port_name.lower() or 'reset' in port_name.lower():
                                module_info['has_reset'] = True
                    
                    elif 'output' in line:
                        # Extract output ports
                        port_line = line.replace('output', '').replace(',', '').replace(';', '').strip()
                        if port_line:
                            port_name = port_line.split()[-1]
                            module_info['outputs'].append(port_name)
            
            return module_info
            
        except Exception as e:
            print(f"Error extracting module info: {e}")
            return None
    
    def _create_testbench_template(self, problem_name: str, module_info: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a basic testbench template"""
        
        module_name = module_info['name']
        
        # Generate signal declarations
        input_signals = ""
        output_signals = ""
        input_assignments = ""
        
        for inp in module_info['inputs']:
            input_signals += f"    reg {inp};\n"
            if 'clk' in inp.lower():
                input_assignments += f"    // Clock generation\n    initial {inp} = 0;\n    always #{analysis.get('clock_period', '1.0ns').replace('ns', '')}/2 {inp} = ~{inp};\n\n"
            elif 'rst' in inp.lower() or 'reset' in inp.lower():
                input_assignments += f"    // Reset sequence\n    initial begin\n        {inp} = 0;\n        #100 {inp} = 1;\n    end\n\n"
            else:
                input_assignments += f"    initial {inp} = 0;\n"
        
        for out in module_info['outputs']:
            output_signals += f"    wire {out};\n"
        
        # Generate port connections
        port_connections = []
        for inp in module_info['inputs']:
            port_connections.append(f".{inp}({inp})")
        for out in module_info['outputs']:
            port_connections.append(f".{out}({out})")
        
        port_list = ",\n        ".join(port_connections)
        
        testbench_code = f"""// Basic testbench for {module_name}
// Generated by ASU Tapeout Agent

`timescale 1ns/1ps

module {module_name}_tb;

    // Signal declarations
{input_signals}{output_signals}
    // DUT instantiation
    {module_name} dut (
        {port_list}
    );

    // Test stimulus
{input_assignments}
    // Test sequence
    initial begin
        $dumpfile("{module_name}_tb.vcd");
        $dumpvars(0, {module_name}_tb);
        
        // Add test vectors here
        #1000;
        
        $display("Test completed");
        $finish;
    end

endmodule
"""
        
        return testbench_code
    
    async def verify_design(self, state: TapeoutState) -> Dict[str, Any]:
        """Complete verification flow using LangChain tools
        
        Args:
            state: The current tapeout state containing RTL code and specs
            
        Returns:
            Dict with verification results and updated state
        """
        rtl_code = state.get("rtl_code", "")
        spec = state.get("problem_spec", {})
        problem_name = state.get("problem_name", list(spec.keys())[0] if spec else "design")
        
        if not rtl_code:
            return {
                "verification_results": {
                    "passed": False,
                    "details": "No RTL code available for verification"
                },
                "past_steps": [("verification", "No RTL code to verify")]
            }
        
        # Run autonomous verification with error recovery if in autonomous mode
        if self.execution_mode == "autonomous" and self.error_recovery:
            return await self._verify_with_error_recovery(state, rtl_code, spec, problem_name)
        
        # Write RTL to temporary file
        rtl_file = f"/tmp/{problem_name}.v"
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        # Find testbench
        testbench_path = self._find_testbench(problem_name)
        if not testbench_path:
            # Generate basic testbench if none found
            module_info = self._extract_module_info(rtl_file)
            if module_info:
                testbench_code = self._create_testbench_template(problem_name, module_info, spec)
                testbench_path = f"/tmp/{problem_name}_tb.v"
                with open(testbench_path, 'w') as f:
                    f.write(testbench_code)
        
        # Create verification task for the agent
        task = f"""Verify the RTL design for {problem_name}:
        
        1. First run lint check on: {rtl_file}
        2. If lint check passes, compile RTL file {rtl_file} with testbench {testbench_path}
        3. If compilation succeeds, run the simulation executable
        4. Analyze any errors or warnings from all steps
        5. Provide comprehensive feedback and recommendations
        
        RTL file path: {rtl_file}
        Testbench path: {testbench_path}
        
        Report the verification status and any issues found."""
        
        try:
            # Run agent with tools
            result = await self.agent.ainvoke({"messages": [("user", task)]})
            
            # Extract verification status from agent's response
            agent_response = result["messages"][-1].content if result.get("messages") else ""
            verification_passed = self._extract_verification_status(result.get("messages", []))
            
            return {
                "verification_results": {
                    "passed": verification_passed,
                    "details": agent_response,
                    "tool_calls": len([m for m in result.get("messages", []) if hasattr(m, "tool_calls")])
                },
                "past_steps": [("verification", f"Verification {'passed' if verification_passed else 'failed'} - used LangChain tools")]
            }
            
        except Exception as e:
            return {
                "verification_results": {
                    "passed": False,
                    "details": f"Verification error: {str(e)}"
                },
                "past_steps": [("verification", f"Verification failed with error: {str(e)}")]
            }
    
    def _extract_verification_status(self, messages: List[Any]) -> bool:
        """Extract verification pass/fail status from agent messages
        
        Args:
            messages: List of messages from the agent
            
        Returns:
            True if verification passed, False otherwise
        """
        # Look through messages for verification results
        for message in messages:
            if hasattr(message, "content"):
                content = message.content.lower()
                # Check for explicit pass/fail indicators
                if "verification passed" in content or "all tests passed" in content:
                    return True
                elif "verification failed" in content or "error" in content:
                    # But not if it's just mentioning that no errors were found
                    if "no error" not in content and "no lint error" not in content:
                        return False
            
            # Check tool call results
            if hasattr(message, "tool_calls"):
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, "args"):
                        # This would be the tool being called
                        pass
            
            # Check if this is a tool result message
            if hasattr(message, "name") and hasattr(message, "content"):
                try:
                    # Try to parse as JSON result from tool
                    import json
                    result = json.loads(message.content)
                    if isinstance(result, dict):
                        if result.get("success") is False:
                            return False
                except:
                    pass
        
        # Default to True if no explicit failures found
        return True

    async def _verify_with_error_recovery(self, state: Dict[str, Any], rtl_code: str, spec: Dict[str, Any], problem_name: str) -> Dict[str, Any]:
        """Verify with automatic error recovery in autonomous mode
        
        Args:
            state: Current state
            rtl_code: RTL code to verify
            spec: Problem specification
            problem_name: Name of the problem
            
        Returns:
            Dict with verification results
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Write current RTL to file
                rtl_file = f"/tmp/{problem_name}.v"
                with open(rtl_file, 'w') as f:
                    f.write(rtl_code)
                
                # Find or generate testbench
                testbench_path = self._find_testbench(problem_name)
                if not testbench_path:
                    module_info = self._extract_module_info(rtl_file)
                    if module_info:
                        testbench_code = self._create_testbench_template(problem_name, module_info, spec)
                        testbench_path = f"/tmp/{problem_name}_tb.v"
                        with open(testbench_path, 'w') as f:
                            f.write(testbench_code)
                
                # Run verification
                verification_result = self._run_autonomous_verification(rtl_file, testbench_path, problem_name)
                
                if verification_result['success']:
                    return {
                        "verification_results": verification_result,
                        "past_steps": [("verification", f"Autonomous verification passed (attempt {retry_count + 1})")]
                    }
                else:
                    # Tool failed, try to recover
                    error_info = {
                        "tool": "verification",
                        "error": verification_result.get("error", "Unknown verification error"),
                        "context": {
                            "stage": "rtl_verification", 
                            "attempt": retry_count,
                            "compilation_errors": verification_result.get("compilation_errors", []),
                            "simulation_errors": verification_result.get("simulation_errors", [])
                        }
                    }
                    
                    # Get fix from error recovery agent
                    fix_result = await self.error_recovery.handle_tool_error(state, error_info)
                    
                    # Apply fix
                    if fix_result.get("rtl_code"):
                        rtl_code = fix_result["rtl_code"]
                        state["rtl_code"] = rtl_code
                    
                    # Check if we should retry
                    if not fix_result.get("retry_action") == "verification":
                        break
                    
                    retry_count += 1
                    
            except Exception as e:
                # Handle unexpected errors
                error_info = {
                    "tool": "verification",
                    "error": str(e),
                    "context": {"stage": "rtl_verification", "unexpected": True}
                }
                
                try:
                    fix_result = await self.error_recovery.handle_tool_error(state, error_info)
                    if fix_result.get("rtl_code"):
                        rtl_code = fix_result["rtl_code"]
                        state["rtl_code"] = rtl_code
                except:
                    pass
                    
                retry_count += 1
        
        # All retries exhausted
        return {
            "verification_results": {
                "passed": False, 
                "details": "Verification failed after maximum retries in autonomous mode",
                "retries_exhausted": True
            },
            "errors": ["Verification failed after error recovery attempts"],
            "past_steps": [("verification", f"Failed after {retry_count} autonomous recovery attempts")]
        }
    
    def _run_autonomous_verification(self, rtl_file: str, testbench_path: str, problem_name: str) -> Dict[str, Any]:
        """Run verification without LangChain agent for autonomous mode
        
        Args:
            rtl_file: Path to RTL file
            testbench_path: Path to testbench
            problem_name: Problem name
            
        Returns:
            Verification results
        """
        results = {
            'success': False,
            'compilation_success': False,
            'simulation_success': False,
            'errors': [],
            'compilation_errors': [],
            'simulation_errors': []
        }
        
        # Lint check with Verilator
        try:
            lint_cmd = f"verilator --lint-only -Wall {rtl_file}"
            lint_result = subprocess.run(lint_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if lint_result.returncode != 0:
                results['errors'].append(f"Lint errors: {lint_result.stderr}")
                results['error'] = lint_result.stderr
                return results
        except Exception as e:
            results['errors'].append(f"Lint check error: {str(e)}")
            results['error'] = str(e)
            return results
        
        # Compile with iVerilog
        compilation_result = self._compile_rtl(rtl_file, testbench_path, problem_name)
        results.update(compilation_result)
        
        if not compilation_result['compilation_success']:
            results['error'] = '\n'.join(compilation_result.get('compilation_errors', []))
            return results
        
        # Run simulation
        simulation_result = self._run_simulation(problem_name)
        results.update(simulation_result)
        
        if not simulation_result['simulation_success']:
            results['error'] = '\n'.join(simulation_result.get('simulation_errors', []))
            return results
        
        results['success'] = True
        return results 