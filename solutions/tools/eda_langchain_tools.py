"""
EDA LangChain Tools Integration
===============================

This module provides EDA tools wrapped as LangChain tools using the @tool decorator.
These tools can be used by agents with ToolNode for automated tool execution.
"""

from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
import subprocess
import json
import os
import tempfile
from pathlib import Path

@tool
def icarus_verilog_compile(rtl_file: str, testbench_file: str) -> Dict[str, Any]:
    """Compile Verilog RTL using Icarus Verilog (iverilog)
    
    Args:
        rtl_file: Path to the RTL Verilog file
        testbench_file: Path to the testbench file
    
    Returns:
        Dict with compilation results, errors, and success status
    """
    try:
        output_file = "/tmp/simulation.out"
        cmd = ["iverilog", "-o", output_file, rtl_file, testbench_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "executable": output_file if result.returncode == 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": "Compilation timeout after 30 seconds"}
    except Exception as e:
        return {"success": False, "errors": str(e)}

@tool
def run_verilog_simulation(executable_path: str) -> Dict[str, Any]:
    """Run compiled Verilog simulation
    
    Args:
        executable_path: Path to the compiled simulation executable
        
    Returns:
        Dict with simulation results and output
    """
    try:
        if not os.path.exists(executable_path):
            return {"success": False, "errors": f"Executable not found: {executable_path}"}
            
        result = subprocess.run([executable_path], capture_output=True, text=True, timeout=60)
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "waveform_generated": "simulation.vcd" in result.stdout or os.path.exists("simulation.vcd")
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": "Simulation timeout after 60 seconds"}
    except Exception as e:
        return {"success": False, "errors": str(e)}

@tool  
def verilator_lint_check(rtl_file: str) -> Dict[str, Any]:
    """Lint check RTL using Verilator
    
    Args:
        rtl_file: Path to the RTL file to check
        
    Returns:
        Dict with lint results and warnings
    """
    try:
        cmd = ["verilator", "--lint-only", "-Wall", rtl_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Parse warnings from Verilator output
        warnings = []
        errors = []
        
        for line in result.stderr.split('\n'):
            if '%Warning' in line:
                warnings.append(line)
            elif '%Error' in line:
                errors.append(line)
        
        return {
            "success": result.returncode == 0 and len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "clean": len(warnings) == 0 and len(errors) == 0,
            "full_output": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": ["Lint check timeout after 30 seconds"]}
    except FileNotFoundError:
        return {"success": False, "errors": ["Verilator not found. Please install it first."]}
    except Exception as e:
        return {"success": False, "errors": [str(e)]}

@tool
def yosys_synthesize(rtl_file: str, top_module: str, target_library: str = "generic") -> Dict[str, Any]:
    """Synthesize RTL using Yosys
    
    Args:
        rtl_file: Path to RTL file
        top_module: Name of top module
        target_library: Target library (generic, sky130, etc.)
        
    Returns:
        Dict with synthesis results and statistics
    """
    try:
        # Create Yosys script based on target library
        if target_library == "sky130":
            script = f"""
            read_verilog {rtl_file}
            hierarchy -check -top {top_module}
            proc; opt; memory; opt
            techmap; opt
            dfflibmap -liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
            abc -liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
            clean
            stat -liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
            write_verilog /tmp/synthesized_{top_module}.v
            """
        else:
            script = f"""
            read_verilog {rtl_file}
            hierarchy -check -top {top_module}
            proc; opt; memory; opt
            techmap; opt
            stat
            write_verilog /tmp/synthesized_{top_module}.v
            """
        
        script_file = "/tmp/synth_script.ys"
        with open(script_file, 'w') as f:
            f.write(script)
            
        cmd = ["yosys", "-s", script_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parse statistics from output
        stats = {}
        for line in result.stdout.split('\n'):
            if 'Number of cells:' in line:
                stats['cell_count'] = int(line.split()[-1])
            elif 'Number of wires:' in line:
                stats['wire_count'] = int(line.split()[-1])
            elif '$_DFF_' in line or 'DFF' in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    stats['ff_count'] = stats.get('ff_count', 0) + int(parts[1])
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "stats": stats,
            "synthesized_file": f"/tmp/synthesized_{top_module}.v" if result.returncode == 0 else None,
            "synthesizable": "successfully" in result.stdout.lower() or result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": "Synthesis timeout after 60 seconds"}
    except Exception as e:
        return {"success": False, "errors": str(e)}

@tool
def openroad_place_and_route(rtl_file: str, sdc_file: str, design_name: str) -> Dict[str, Any]:
    """Run OpenROAD place and route flow
    
    Args:
        rtl_file: Path to synthesized RTL
        sdc_file: Path to SDC constraints
        design_name: Name of the design
        
    Returns:
        Dict with P&R results and metrics
    """
    try:
        # Check if OpenROAD-flow-scripts is available
        orfs_dir = "/OpenROAD-flow-scripts"
        if not os.path.exists(orfs_dir):
            return {
                "success": False,
                "errors": "OpenROAD-flow-scripts not found. Please set up ORFS environment."
            }
        
        # Create working directory
        work_dir = f"/tmp/orfs_{design_name}"
        os.makedirs(work_dir, exist_ok=True)
        
        # Copy files to working directory
        import shutil
        shutil.copy(rtl_file, f"{work_dir}/{design_name}.v")
        shutil.copy(sdc_file, f"{work_dir}/{design_name}.sdc")
        
        # Run ORFS make command
        cmd = [
            "make", "-C", f"{orfs_dir}/flow",
            f"DESIGN_NAME={design_name}",
            f"VERILOG_FILES={work_dir}/{design_name}.v",
            f"SDC_FILE={work_dir}/{design_name}.sdc",
            "PLATFORM=sky130hd",
            f"WORK_DIR={work_dir}",
            "DESIGN_CONFIG=config.mk"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=work_dir)
        
        # Parse results for key metrics
        metrics = {}
        if result.returncode == 0:
            # Look for results in log files
            log_files = [
                f"{work_dir}/logs/sky130hd/{design_name}/6_report.log",
                f"{work_dir}/reports/sky130hd/{design_name}/6_final_report.txt"
            ]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        content = f.read()
                        # Parse timing metrics
                        if "wns" in content.lower():
                            for line in content.split('\n'):
                                if "wns" in line.lower():
                                    parts = line.split()
                                    for i, part in enumerate(parts):
                                        if "wns" in part.lower() and i+1 < len(parts):
                                            try:
                                                metrics['wns'] = float(parts[i+1])
                                            except:
                                                pass
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,  # Limit output size
            "errors": result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr,
            "metrics": metrics,
            "odb_file": f"{work_dir}/results/sky130hd/{design_name}/6_final.odb" if result.returncode == 0 else None,
            "gds_file": f"{work_dir}/results/sky130hd/{design_name}/6_final.gds" if result.returncode == 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": "Place and route timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "errors": str(e)}

@tool
def generate_sdc_constraints(clock_name: str, clock_period: float, input_ports: List[str], output_ports: List[str]) -> Dict[str, Any]:
    """Generate basic SDC timing constraints
    
    Args:
        clock_name: Name of the clock signal
        clock_period: Clock period in nanoseconds
        input_ports: List of input port names
        output_ports: List of output port names
        
    Returns:
        Dict with generated SDC content and file path
    """
    try:
        # Generate SDC content
        sdc_content = f"""# SDC Constraints generated by ASU Tapeout Agent
# Clock constraint
create_clock -name {clock_name} -period {clock_period} [get_ports {clock_name}]

# Input delays (20% of clock period)
set_input_delay -clock {clock_name} -max {clock_period * 0.2} [get_ports {{{' '.join(input_ports)}}}]
set_input_delay -clock {clock_name} -min 0 [get_ports {{{' '.join(input_ports)}}}]

# Output delays (20% of clock period)  
set_output_delay -clock {clock_name} -max {clock_period * 0.2} [get_ports {{{' '.join(output_ports)}}}]
set_output_delay -clock {clock_name} -min 0 [get_ports {{{' '.join(output_ports)}}}]

# Clock uncertainty (5% of clock period)
set_clock_uncertainty {clock_period * 0.05} [get_clocks {clock_name}]

# Don't touch the clock network
set_dont_touch_network [get_clocks {clock_name}]
"""
        
        # Save to file
        sdc_file = f"/tmp/constraints_{clock_name}.sdc"
        with open(sdc_file, 'w') as f:
            f.write(sdc_content)
        
        return {
            "success": True,
            "sdc_content": sdc_content,
            "sdc_file": sdc_file
        }
    except Exception as e:
        return {"success": False, "errors": str(e)}

@tool
def analyze_timing_report(report_file: str) -> Dict[str, Any]:
    """Analyze timing report from synthesis or P&R
    
    Args:
        report_file: Path to timing report file
        
    Returns:
        Dict with parsed timing metrics
    """
    try:
        if not os.path.exists(report_file):
            return {"success": False, "errors": f"Report file not found: {report_file}"}
        
        with open(report_file, 'r') as f:
            content = f.read()
        
        metrics = {
            "wns": None,  # Worst Negative Slack
            "tns": None,  # Total Negative Slack
            "critical_paths": [],
            "violated_paths": 0
        }
        
        # Parse timing metrics (patterns may vary by tool)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for WNS
            if "worst negative slack" in line_lower or "wns" in line_lower:
                parts = line.split()
                for j, part in enumerate(parts):
                    try:
                        val = float(part)
                        metrics["wns"] = val
                        break
                    except:
                        continue
            
            # Look for TNS
            if "total negative slack" in line_lower or "tns" in line_lower:
                parts = line.split()
                for j, part in enumerate(parts):
                    try:
                        val = float(part)
                        metrics["tns"] = val
                        break
                    except:
                        continue
            
            # Count violated paths
            if "violated" in line_lower and "path" in line_lower:
                metrics["violated_paths"] += 1
        
        # Determine if timing is met
        timing_met = metrics["wns"] is not None and metrics["wns"] >= 0
        
        return {
            "success": True,
            "metrics": metrics,
            "timing_met": timing_met,
            "summary": f"WNS: {metrics['wns']}, TNS: {metrics['tns']}, Violated Paths: {metrics['violated_paths']}"
        }
    except Exception as e:
        return {"success": False, "errors": str(e)}

# Export all tools
__all__ = [
    'icarus_verilog_compile',
    'run_verilog_simulation',
    'verilator_lint_check',
    'yosys_synthesize',
    'openroad_place_and_route',
    'generate_sdc_constraints',
    'analyze_timing_report'
] 