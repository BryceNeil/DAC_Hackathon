"""
Logging system for ASU Tapeout Agent
====================================

Provides structured logging with color support and different log levels.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog

class AgentLogger:
    """Custom logger for the agent system"""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, verbose: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers.clear()
        
        # Console handler with color
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Color formatter for console
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def step(self, step_name: str, status: str = "STARTING"):
        """Log a workflow step"""
        self.info(f"{'='*60}")
        self.info(f"STEP: {step_name} - {status}")
        self.info(f"{'='*60}")
    
    def tool_call(self, tool: str, command: str):
        """Log a tool invocation"""
        self.debug(f"Tool: {tool}")
        self.debug(f"Command: {command}")
    
    def tool_result(self, tool: str, success: bool, output: Optional[str] = None):
        """Log tool result"""
        if success:
            self.info(f"✓ {tool} completed successfully")
        else:
            self.error(f"✗ {tool} failed")
        if output and self.logger.level == logging.DEBUG:
            self.debug(f"Output: {output[:500]}...")  # Truncate long outputs
    
    def agent_action(self, agent: str, action: str):
        """Log agent action"""
        self.info(f"[{agent}] {action}")
    
    def metrics(self, metrics: dict):
        """Log design metrics"""
        self.info("Design Metrics:")
        for key, value in metrics.items():
            self.info(f"  {key}: {value}")


# Global logger instances
_loggers = {}

def get_logger(name: str, log_file: Optional[Path] = None, verbose: bool = True) -> AgentLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = AgentLogger(name, log_file, verbose)
    return _loggers[name]

def setup_logging(log_dir: Path, verbose: bool = True):
    """Set up logging for the entire system"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"asu_tapeout_{timestamp}.log"
    
    # Create main logger
    main_logger = get_logger("ASUTapeout", log_file, verbose)
    main_logger.info(f"Logging initialized - Log file: {log_file}")
    
    return main_logger 