"""
EDA Tools Wrapper for LangGraph
===============================

LangGraph-compatible tool wrappers for EDA operations.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
import subprocess
import os
import tempfile
from pathlib import Path


@tool
def run_yosys_synthesis(rtl_code: str, module_name: str, target_lib: str = "sky130") -> dict:
    """
    Run Yosys synthesis on RTL code.
    
    Args:
        rtl_code: SystemVerilog/Verilog RTL code
        module_name: Top module name
        target_lib: Target technology library
        
    Returns:
        Dictionary with synthesis results including gates count, area, timing
    """
    try:
        # Create temporary file for RTL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(rtl_code)
            rtl_file = f.name
        
        # Create Yosys script
        yosys_script = f"""
read_verilog {rtl_file}
hierarchy -top {module_name}
proc; opt; memory; opt; fsm; opt
techmap; opt
"""
        
        if target_lib == "sky130":
            yosys_script += """
dfflibmap -liberty /pdk/sky130A/sky130_fd_sc_hd.lib
abc -liberty /pdk/sky130A/sky130_fd_sc_hd.lib
"""
        
        yosys_script += f"""
stat
write_verilog {rtl_file}.synth.v
"""
        
        # Write Yosys script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as f:
            f.write(yosys_script)
            script_file = f.name
        
        # Run Yosys
        result = subprocess.run(
            ['yosys', '-s', script_file],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse results
        if result.returncode == 0:
            return _parse_yosys_output(result.stdout)
        else:
            return {
                'success': False,
                'error': result.stderr,
                'gates': 0,
                'area': 0.0
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'gates': 0,
            'area': 0.0
        }
    finally:
        # Cleanup temp files
        try:
            os.unlink(rtl_file)
            os.unlink(script_file)
        except:
            pass


@tool  
def run_iverilog_simulation(rtl_code: str, testbench_code: str, module_name: str) -> dict:
    """
    Run Icarus Verilog simulation.
    
    Args:
        rtl_code: RTL code to simulate
        testbench_code: Testbench code
        module_name: Module name
        
    Returns:
        Dictionary with simulation results
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(rtl_code)
            rtl_file = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(testbench_code)
            tb_file = f.name
            
        sim_file = f"/tmp/{module_name}_sim"
        
        # Compile with iverilog
        compile_result = subprocess.run(
            ['iverilog', '-o', sim_file, rtl_file, tb_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if compile_result.returncode != 0:
            return {
                'success': False,
                'compile_error': compile_result.stderr,
                'simulation_output': ''
            }
        
        # Run simulation
        sim_result = subprocess.run(
            [sim_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return {
            'success': sim_result.returncode == 0,
            'compile_error': '',
            'simulation_output': sim_result.stdout,
            'simulation_error': sim_result.stderr
        }
        
    except Exception as e:
        return {
            'success': False,
            'compile_error': str(e),
            'simulation_output': ''
        }
    finally:
        # Cleanup
        try:
            os.unlink(rtl_file)
            os.unlink(tb_file)
            os.unlink(sim_file)
        except:
            pass


@tool
def run_openroad_flow(rtl_file: str, sdc_file: str, output_dir: str, pdk: str = "sky130hd") -> dict:
    """
    Run OpenROAD flow for physical implementation.
    
    Args:
        rtl_file: Path to synthesized RTL
        sdc_file: Path to SDC constraints
        output_dir: Output directory for results
        pdk: PDK to use
        
    Returns:
        Dictionary with physical implementation results
    """
    try:
        # Create OpenROAD flow configuration
        config = _create_openroad_config(rtl_file, sdc_file, output_dir, pdk)
        
        # Write config file
        config_file = os.path.join(output_dir, "config.mk")
        with open(config_file, 'w') as f:
            f.write(config)
        
        # Run OpenROAD flow
        result = subprocess.run(
            ['make', '-f', '/openroad-flow/Makefile', f'WORK_HOME={output_dir}'],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        # Parse results
        if result.returncode == 0:
            return _parse_openroad_results(output_dir)
        else:
            return {
                'success': False,
                'error': result.stderr,
                'odb_file': None,
                'metrics': {}
            }
            
    except Exception as e:
        return {
            'success': False,  
            'error': str(e),
            'odb_file': None,
            'metrics': {}
        }


@tool
def validate_timing_constraints(sdc_content: str, rtl_code: str) -> dict:
    """
    Validate SDC timing constraints against RTL.
    
    Args:
        sdc_content: SDC constraint content
        rtl_code: RTL code
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Basic SDC validation
        issues = []
        
        # Check for required constraints
        if 'create_clock' not in sdc_content:
            issues.append("Missing create_clock constraint")
            
        if 'set_input_delay' not in sdc_content:
            issues.append("Missing input delay constraints")
            
        if 'set_output_delay' not in sdc_content:
            issues.append("Missing output delay constraints")
        
        # Extract clock names from RTL
        rtl_clocks = _extract_clock_signals(rtl_code)
        
        # Validate clock constraints match RTL
        for clock in rtl_clocks:
            if clock not in sdc_content:
                issues.append(f"Clock signal '{clock}' not constrained in SDC")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'clock_signals': rtl_clocks
        }
        
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Validation error: {str(e)}"],
            'clock_signals': []
        }


def _parse_yosys_output(output: str) -> dict:
    """Parse Yosys synthesis output"""
    lines = output.split('\n')
    gates = 0
    area = 0.0
    
    for line in lines:
        if 'Number of cells:' in line:
            gates = int(line.split()[-1])
        elif 'Chip area' in line:
            try:
                area = float(line.split()[-1])
            except:
                area = 0.0
    
    return {
        'success': True,
        'gates': gates,
        'area': area,
        'output': output
    }


def _create_openroad_config(rtl_file: str, sdc_file: str, output_dir: str, pdk: str) -> str:
    """Create OpenROAD flow configuration"""
    return f"""
export DESIGN_NAME = $(basename {rtl_file} .v)
export PLATFORM    = {pdk}
export VERILOG_FILES = {rtl_file}
export SDC_FILE = {sdc_file}
export CORE_UTILIZATION = 40
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN = 2
export PLACE_DENSITY = 0.65
"""


def _parse_openroad_results(output_dir: str) -> dict:
    """Parse OpenROAD flow results"""
    results_dir = os.path.join(output_dir, "results", "final")
    
    # Look for final ODB
    odb_file = os.path.join(results_dir, "6_final.odb")
    
    # Parse timing report if available
    timing_report = os.path.join(results_dir, "timing.rpt")
    metrics = {}
    
    if os.path.exists(timing_report):
        metrics = _parse_timing_report(timing_report)
    
    return {
        'success': os.path.exists(odb_file),
        'odb_file': odb_file if os.path.exists(odb_file) else None,
        'metrics': metrics
    }


def _parse_timing_report(report_file: str) -> dict:
    """Parse timing report for metrics"""
    try:
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Extract basic timing metrics
        wns = 0.0
        tns = 0.0
        
        # Simple parsing - can be enhanced
        lines = content.split('\n')
        for line in lines:
            if 'WNS' in line:
                try:
                    wns = float(line.split()[-1])
                except:
                    pass
            elif 'TNS' in line:
                try:
                    tns = float(line.split()[-1])
                except:
                    pass
        
        return {
            'wns': wns,
            'tns': tns
        }
    except:
        return {}


def _extract_clock_signals(rtl_code: str) -> List[str]:
    """Extract clock signal names from RTL"""
    import re
    
    # Simple regex to find clock signals
    clock_patterns = [
        r'input\s+.*?(\w*clk\w*)',
        r'input\s+.*?(\w*clock\w*)',
        r'(\w+)\s*<=.*@\(posedge\s+(\w+)\)'
    ]
    
    clocks = set()
    for pattern in clock_patterns:
        matches = re.finditer(pattern, rtl_code, re.IGNORECASE)
        for match in matches:
            clocks.add(match.group(1))
    
    return list(clocks)


class EDATools:
    """
    EDA Tools wrapper class for providing access to EDA operations.
    This class provides a unified interface to various EDA tools and operations.
    """
    
    def __init__(self):
        """Initialize EDA Tools wrapper"""
        pass
    
    def run_yosys_synthesis(self, rtl_code: str, module_name: str, target_lib: str = "sky130") -> dict:
        """Wrapper for yosys synthesis"""
        return run_yosys_synthesis.invoke({"rtl_code": rtl_code, "module_name": module_name, "target_lib": target_lib})
    
    def run_iverilog_simulation(self, rtl_code: str, testbench_code: str, module_name: str) -> dict:
        """Wrapper for iverilog simulation"""
        return run_iverilog_simulation.invoke({"rtl_code": rtl_code, "testbench_code": testbench_code, "module_name": module_name})
    
    def run_openroad_flow(self, rtl_file: str, sdc_file: str, output_dir: str, pdk: str = "sky130hd") -> dict:
        """Wrapper for OpenROAD flow"""
        return run_openroad_flow.invoke({"rtl_file": rtl_file, "sdc_file": sdc_file, "output_dir": output_dir, "pdk": pdk})
    
    def validate_timing_constraints(self, sdc_content: str, rtl_code: str) -> dict:
        """Wrapper for timing constraints validation"""
        return validate_timing_constraints.invoke({"sdc_content": sdc_content, "rtl_code": rtl_code}) 