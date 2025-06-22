"""
Configuration management for ASU Tapeout Agent
=============================================

Manages API keys, tool paths, PDK selection, and problem classification.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import yaml

class ProblemType(Enum):
    """Classification of problem types based on design characteristics"""
    STATE_MACHINE = "state_machine"
    ARITHMETIC = "arithmetic"
    PIPELINED = "pipelined"
    FIXED_POINT = "fixed_point"
    FLOATING_POINT = "floating_point"
    DSP = "dsp"
    MEMORY = "memory"
    COMBINATIONAL = "combinational"
    UNKNOWN = "unknown"


class PDKType(Enum):
    """Available Process Design Kits"""
    SKY130HD = "sky130hd"
    SKY130HS = "sky130hs"
    ASAP7 = "asap7"
    NANGATE45 = "nangate45"
    GF180 = "gf180mcu"
    IHP_SG13G2 = "ihp-sg13g2"


class ToolPaths(BaseModel):
    """Paths to EDA tools"""
    iverilog: str = Field(default="iverilog")
    verilator: str = Field(default="verilator")
    yosys: str = Field(default="yosys")
    openroad: str = Field(default="openroad")
    sta: str = Field(default="sta")
    klayout: str = Field(default="klayout")
    gtkwave: str = Field(default="gtkwave")
    orfs_flow_dir: str = Field(default="/OpenROAD-flow-scripts/flow")


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, ollama")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=4000, description="Maximum tokens for response")
    api_key: Optional[str] = Field(default=None, description="API key")


class AgentConfig(BaseModel):
    """Main configuration for the agent system"""
    # Tool paths
    tools: ToolPaths = Field(default_factory=ToolPaths)
    
    # LLM configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # PDK selection
    default_pdk: PDKType = Field(default=PDKType.SKY130HD)
    
    # Directories
    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    temp_dir: Path = Field(default=Path("/tmp/asu_tapeout"))
    
    # Execution settings
    parallel_jobs: int = Field(default=4, description="Number of parallel jobs")
    timeout_synthesis: int = Field(default=300, description="Synthesis timeout in seconds")
    timeout_pnr: int = Field(default=600, description="Place and route timeout in seconds")
    timeout_simulation: int = Field(default=60, description="Simulation timeout in seconds")
    
    # Debugging
    debug_mode: bool = Field(default=False)
    keep_temp_files: bool = Field(default=False)
    verbose: bool = Field(default=True)


class ConfigManager:
    """Manages configuration loading and problem classification"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self._setup_directories()
    
    def _load_config(self, config_file: Optional[str]) -> AgentConfig:
        """Load configuration from file or environment"""
        config = AgentConfig()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                config = AgentConfig(**config_data)
        
        # Override with environment variables
        if api_key := os.getenv("OPENAI_API_KEY"):
            config.llm.provider = "openai"
            config.llm.api_key = api_key
        elif api_key := os.getenv("ANTHROPIC_API_KEY"):
            config.llm.provider = "anthropic"
            config.llm.api_key = api_key
            config.llm.model = "claude-3-opus-20240229"
        
        # Check for ORFS environment
        if orfs_dir := os.getenv("ORFS_ROOT"):
            config.tools.orfs_flow_dir = f"{orfs_dir}/flow"
        
        return config
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def classify_problem(self, spec: dict) -> ProblemType:
        """Classify problem type based on specification"""
        problem_name = list(spec.keys())[0]
        problem_data = spec[problem_name]
        
        description = problem_data.get('description', '').lower()
        module_sig = problem_data.get('module_signature', '').lower()
        
        # Check for specific patterns
        if any(term in description for term in ['state machine', 'fsm', 'detector', 'sequence']):
            return ProblemType.STATE_MACHINE
        elif any(term in description for term in ['multiply', 'add', 'dot product', 'arithmetic']):
            if 'pipeline' in description or 'stage' in description:
                return ProblemType.PIPELINED
            else:
                return ProblemType.ARITHMETIC
        elif any(term in description for term in ['fixed point', 'q format']):
            return ProblemType.FIXED_POINT
        elif any(term in description for term in ['floating point', 'fp16', 'ieee']):
            return ProblemType.FLOATING_POINT
        elif any(term in description for term in ['fir', 'filter', 'dsp', 'convolution']):
            return ProblemType.DSP
        elif 'mem' in module_sig or 'ram' in module_sig:
            return ProblemType.MEMORY
        elif 'clk' not in module_sig:
            return ProblemType.COMBINATIONAL
        else:
            return ProblemType.UNKNOWN
    
    def get_pdk_for_problem(self, problem_type: ProblemType) -> PDKType:
        """Select appropriate PDK based on problem type"""
        # Use Sky130HD as default for most designs
        # Could use HS for high-speed requirements
        if problem_type == ProblemType.FLOATING_POINT:
            # Might want faster PDK for FP operations
            return PDKType.SKY130HS
        else:
            return self.config.default_pdk
    
    def get_tool_path(self, tool: str) -> str:
        """Get path to a specific tool"""
        return getattr(self.config.tools, tool, tool)
    
    def get_timeout(self, operation: str) -> int:
        """Get timeout for a specific operation"""
        timeout_map = {
            'synthesis': self.config.timeout_synthesis,
            'pnr': self.config.timeout_pnr,
            'simulation': self.config.timeout_simulation
        }
        return timeout_map.get(operation, 300)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def init_config(config_file: Optional[str] = None) -> ConfigManager:
    """Initialize configuration with specific file"""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager 