"""
RTL Generator Agent with LangGraph ReAct Pattern
================================================

This agent uses LangGraph's ReAct pattern with tools to generate
high-quality SystemVerilog RTL from YAML specifications.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import re
import subprocess
import tempfile
from pathlib import Path

from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from .state import TapeoutState
from tools.file_manager import FileManager


# Define tools for RTL generation
@tool
def validate_rtl_syntax(rtl_code: str) -> str:
    """Validate RTL syntax using Verilator
    
    Args:
        rtl_code: SystemVerilog RTL code to validate
        
    Returns:
        Validation result message
    """
    try:
        # Create temporary file for RTL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
            f.write(rtl_code)
            rtl_file = f.name
        
        # Run Verilator for linting
        cmd = ['verilator', '--lint-only', '-Wall', rtl_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up
        os.unlink(rtl_file)
        
        if result.returncode == 0:
            return "RTL syntax is valid. No errors found."
        else:
            errors = result.stderr if result.stderr else result.stdout
            return f"RTL syntax errors found:\n{errors}"
            
    except Exception as e:
        return f"Could not validate RTL: {str(e)}"


@tool
def generate_testbench(spec: dict, module_name: str) -> str:
    """Generate basic testbench from specification
    
    Args:
        spec: Design specification dictionary
        module_name: Name of the module to test
        
    Returns:
        Generated testbench code
    """
    try:
        # Extract module info from spec
        module_spec = spec.get(module_name, {})
        inputs = module_spec.get('module_signature', {}).get('inputs', {})
        outputs = module_spec.get('module_signature', {}).get('outputs', {})
        
        # Generate testbench
        tb_code = f"""// Testbench for {module_name}
`timescale 1ns/1ps

module tb_{module_name};

    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Module signals
"""
        
        # Add input declarations
        for name, width in inputs.items():
            if name not in ['clk', 'rst_n']:
                tb_code += f"    logic [{width-1}:0] {name};\n"
        
        # Add output declarations
        for name, width in outputs.items():
            tb_code += f"    logic [{width-1}:0] {name};\n"
        
        tb_code += f"""
    // Instantiate DUT
    {module_name} dut (
        .clk(clk),
        .rst_n(rst_n),
"""
        
        # Add port connections
        all_ports = list(inputs.items()) + list(outputs.items())
        for i, (name, _) in enumerate(all_ports):
            if name not in ['clk', 'rst_n']:
                tb_code += f"        .{name}({name})"
                if i < len(all_ports) - 1:
                    tb_code += ","
                tb_code += "\n"
        
        tb_code += """    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock
    
    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        #20;
        rst_n = 1;
        
        // Add test vectors here
        #100;
        
        $display("Testbench completed");
        $finish;
    end
    
    // Monitor
    initial begin
        $monitor("Time=%0t rst_n=%b", $time, rst_n);
    end

endmodule
"""
        
        return tb_code
        
    except Exception as e:
        return f"Error generating testbench: {str(e)}"


@tool
def compile_rtl(rtl_code: str, testbench_code: str) -> dict:
    """Compile RTL with Icarus Verilog
    
    Args:
        rtl_code: RTL code to compile
        testbench_code: Testbench code
        
    Returns:
        Compilation results
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
            f.write(rtl_code)
            rtl_file = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
            f.write(testbench_code)
            tb_file = f.name
        
        # Compile with Icarus Verilog
        cmd = ['iverilog', '-g2012', '-o', 'sim.vvp', tb_file, rtl_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up
        os.unlink(rtl_file)
        os.unlink(tb_file)
        if os.path.exists('sim.vvp'):
            os.unlink('sim.vvp')
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "errors": str(e)
        }


@tool 
def extract_module_signature(spec: dict) -> str:
    """Extract module signature from specification
    
    Args:
        spec: Design specification
        
    Returns:
        Module signature string
    """
    try:
        # Get the first (and usually only) module
        module_name = list(spec.keys())[0]
        module_spec = spec[module_name]
        
        # Extract signature
        signature = module_spec.get('module_signature', {})
        inputs = signature.get('inputs', {})
        outputs = signature.get('outputs', {})
        
        # Build signature string
        sig_str = f"module {module_name} (\n"
        
        # Add inputs
        for name, width in inputs.items():
            sig_str += f"    input  logic [{width-1}:0] {name},\n"
            
        # Add outputs
        output_names = list(outputs.keys())
        for i, (name, width) in enumerate(outputs.items()):
            sig_str += f"    output logic [{width-1}:0] {name}"
            if i < len(output_names) - 1:
                sig_str += ","
            sig_str += "\n"
            
        sig_str += ");"
        
        return sig_str
        
    except Exception as e:
        return f"Error extracting signature: {str(e)}"


# RTL Generation prompt template
rtl_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert RTL generation agent specialized in creating 
    high-quality, synthesizable SystemVerilog code from specifications.
    
    Your responsibilities:
    1. Analyze the specification thoroughly
    2. Generate RTL that exactly matches the module signature
    3. Implement all described functionality correctly
    4. Use the provided tools to validate your work
    5. Iterate and fix any issues found
    
    Guidelines for RTL generation:
    - Use SystemVerilog-2012 syntax
    - Follow proper coding conventions (proper indentation, meaningful names)
    - Add comments explaining complex logic
    - Ensure all outputs are driven
    - Avoid latches in combinational logic
    - Use non-blocking assignments in sequential blocks
    - Use blocking assignments in combinational blocks
    
    Always validate your RTL using the available tools before finalizing."""),
    
    ("user", "{task}")
])


class RTLGenerationAgent:
    """ReAct-based agent for RTL generation"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514"):
        """Initialize the RTL generation agent
        
        Args:
            llm_model: The LLM model to use
        """
        self.tools = [
            validate_rtl_syntax,
            generate_testbench,
            compile_rtl,
            extract_module_signature
        ]
        
        # Select LLM based on model name
        if "claude" in llm_model.lower():
            model = ChatAnthropic(model=llm_model, temperature=0)
        else:
            model = ChatOpenAI(model=llm_model, temperature=0)
        
        # Create ReAct agent
        self.agent = create_react_agent(
            model=model,
            tools=self.tools,
            prompt=rtl_generation_prompt
        )
        
        self.file_manager = FileManager()
    
    async def generate_rtl(self, state: TapeoutState) -> Dict[str, Any]:
        """Generate RTL using ReAct pattern
        
        Args:
            state: Current tapeout state
            
        Returns:
            State update with generated RTL
        """
        spec = state["problem_spec"]
        problem_name = state.get("problem_name") or list(spec.keys())[0]
        
        # Create detailed task for the agent
        task = f"""Generate SystemVerilog RTL for module: {problem_name}
        
Specification:
{spec}

Requirements:
1. First use extract_module_signature to understand the exact interface
2. Generate RTL that implements the described functionality
3. Validate the RTL syntax using validate_rtl_syntax
4. Generate a testbench using generate_testbench
5. Compile both RTL and testbench using compile_rtl
6. If there are any errors, fix them and re-validate

Make sure the RTL is complete, synthesizable, and matches the specification exactly."""
        
        # Run ReAct agent
        result = await self.agent.ainvoke({"task": task})
        
        # Extract RTL from agent's work
        rtl_code = self._extract_rtl_from_messages(result["messages"])
        
        if rtl_code:
            # Save RTL file
            rtl_path = f"rtl/{problem_name}.sv"
            self.file_manager.save_file(rtl_path, rtl_code)
            
            return {
                "rtl_code": rtl_code,
                "past_steps": [(
                    "rtl_generation",
                    f"Generated and validated RTL for {problem_name}"
                )]
            }
        else:
            return {
                "errors": ["Failed to generate valid RTL"],
                "past_steps": [("rtl_generation", "RTL generation failed")]
            }
    
    def _extract_rtl_from_messages(self, messages: List[Any]) -> Optional[str]:
        """Extract RTL code from agent messages
        
        Args:
            messages: List of messages from the agent
            
        Returns:
            Extracted RTL code or None
        """
        rtl_code = None
        
        # Look through messages for RTL code
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                content = msg.content
                
                # Look for SystemVerilog code blocks
                sv_matches = re.findall(
                    r'```(?:systemverilog|verilog|sv)\n(.*?)```',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                
                if sv_matches:
                    # Take the last/most complete version
                    rtl_code = sv_matches[-1].strip()
                    break
                
                # Also check for module definitions without code blocks
                if 'module ' in content and 'endmodule' in content:
                    module_match = re.search(
                        r'(module\s+\w+.*?endmodule)',
                        content,
                        re.DOTALL
                    )
                    if module_match:
                        rtl_code = module_match.group(1)
                        break
        
        return rtl_code
    
    def _post_process_rtl(self, rtl_code: str, analysis: Dict[str, Any]) -> str:
        """Post-process generated RTL for consistency"""
        
        # Remove any markdown formatting
        rtl_code = re.sub(r'```systemverilog\n?|```verilog\n?|```\n?', '', rtl_code)
        
        # Ensure proper module name
        module_name = analysis['problem_name']
        rtl_code = re.sub(r'module\s+\w+', f'module {module_name}', rtl_code)
        
        # Add standard header if missing
        if not rtl_code.startswith('//'):
            header = f"""// Generated RTL for {module_name}
// Description: {analysis['description']}
// Generated by ASU Tapeout Agent

"""
            rtl_code = header + rtl_code
        
        return rtl_code
    
    def _validate_rtl_syntax(self, rtl_code: str) -> bool:
        """Basic RTL syntax validation"""
        
        # Check for required elements
        required_elements = ['module ', 'endmodule']
        for element in required_elements:
            if element not in rtl_code:
                return False
        
        # Check for balanced begin/end
        begin_count = rtl_code.count('begin')
        end_count = rtl_code.count('end') - rtl_code.count('endmodule')
        
        if begin_count != end_count:
            return False
        
        # Check for basic module structure
        lines = rtl_code.split('\n')
        has_module = any('module ' in line for line in lines)
        has_endmodule = any('endmodule' in line for line in lines)
        
        return has_module and has_endmodule
    
    def _generate_template_rtl(self, analysis: Dict[str, Any]) -> str:
        """Generate RTL using templates as fallback"""
        
        module_name = analysis['problem_name']
        description = analysis['description']
        module_signature = analysis['module_signature']
        
        # Load appropriate template based on features
        template = self._select_template(analysis['required_features'])
        
        # Basic template RTL
        rtl_code = f"""// Generated RTL for {module_name}
// Description: {description}
// Generated by ASU Tapeout Agent (Template Mode)

{module_signature}
    
    {template}
    
endmodule
"""
        
        return rtl_code
    
    def _select_template(self, required_features: list) -> str:
        """Select appropriate template based on required features"""
        
        if 'state_machine' in required_features:
            return self._get_state_machine_template()
        elif 'arithmetic' in required_features:
            return self._get_arithmetic_template()
        elif 'memory' in required_features:
            return self._get_memory_template()
        else:
            return self._get_basic_template()
    
    def _get_state_machine_template(self) -> str:
        """State machine template"""
        return """// State machine implementation
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        ACTIVE = 2'b01,
        DONE = 2'b10
    } state_t;
    
    state_t current_state, next_state;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                // Add state transition logic
            end
            ACTIVE: begin
                // Add state transition logic
            end
            DONE: begin
                // Add state transition logic
            end
            default: next_state = IDLE;
        endcase
    end
    
    // Output logic
    always_comb begin
        // Add output assignments
    end"""
    
    def _get_arithmetic_template(self) -> str:
        """Arithmetic operation template"""
        return """// Arithmetic operations
    logic [31:0] result_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_reg <= 32'b0;
        end else begin
            // Add arithmetic operations
        end
    end
    
    // Output assignment
    // assign output_port = result_reg;"""
    
    def _get_memory_template(self) -> str:
        """Memory interface template"""
        return """// Memory interface
    logic [31:0] memory [0:255];
    logic [7:0] addr_reg;
    logic [31:0] data_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_reg <= 8'b0;
            data_reg <= 32'b0;
        end else begin
            // Add memory operations
        end
    end"""
    
    def _get_basic_template(self) -> str:
        """Basic combinational template"""
        return """// Basic logic implementation
    always_comb begin
        // Add combinational logic here
    end
    
    // Register implementation if needed
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset logic
        end else begin
            // Sequential logic
        end
    end""" 