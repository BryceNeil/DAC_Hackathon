#!/usr/bin/env python3
"""
Progress Indicator Utility
==========================

Provides visual feedback during long-running operations with animated dots,
spinners, and status messages.
"""

import sys
import time
import threading
from typing import Optional, Callable


class ProgressIndicator:
    """Thread-safe progress indicator with various display modes"""
    
    def __init__(self):
        self.active = False
        self.thread = None
        self.message = ""
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current_spinner = 0
        
    def start_dots(self, message: str = "Processing", max_dots: int = 3):
        """Start animated dots progress indicator
        
        Args:
            message: Base message to display
            max_dots: Maximum number of dots to show
        """
        self.active = True
        self.message = message
        
        def animate():
            dots = 0
            while self.active:
                sys.stdout.write(f'\r{self.message}{"." * dots}   ')
                sys.stdout.flush()
                dots = (dots + 1) % (max_dots + 1)
                time.sleep(0.5)
            sys.stdout.write('\r' + ' ' * (len(self.message) + max_dots + 3) + '\r')
            sys.stdout.flush()
        
        self.thread = threading.Thread(target=animate)
        self.thread.daemon = True
        self.thread.start()
    
    def start_spinner(self, message: str = "Working"):
        """Start spinner progress indicator
        
        Args:
            message: Message to display with spinner
        """
        self.active = True
        self.message = message
        
        def animate():
            while self.active:
                char = self.spinner_chars[self.current_spinner]
                sys.stdout.write(f'\r{char} {self.message}  ')
                sys.stdout.flush()
                self.current_spinner = (self.current_spinner + 1) % len(self.spinner_chars)
                time.sleep(0.1)
            sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')
            sys.stdout.flush()
        
        self.thread = threading.Thread(target=animate)
        self.thread.daemon = True
        self.thread.start()
    
    def start_thinking(self):
        """Start a thinking indicator with rotating brain emoji"""
        self.active = True
        
        def animate():
            thinking_emojis = ['🤔', '💭', '🧠', '💡']
            idx = 0
            while self.active:
                emoji = thinking_emojis[idx]
                sys.stdout.write(f'\r{emoji} Thinking and reasoning...  ')
                sys.stdout.flush()
                idx = (idx + 1) % len(thinking_emojis)
                time.sleep(0.5)
            sys.stdout.write('\r' + ' ' * 30 + '\r')
            sys.stdout.flush()
        
        self.thread = threading.Thread(target=animate)
        self.thread.daemon = True
        self.thread.start()
    
    def start_planning(self):
        """Start a planning indicator"""
        self.active = True
        
        def animate():
            planning_steps = ['📋 Planning', '📝 Analyzing', '🗺️ Strategizing', '📊 Evaluating']
            idx = 0
            while self.active:
                step = planning_steps[idx]
                sys.stdout.write(f'\r{step}...  ')
                sys.stdout.flush()
                idx = (idx + 1) % len(planning_steps)
                time.sleep(0.6)
            sys.stdout.write('\r' + ' ' * 30 + '\r')
            sys.stdout.flush()
        
        self.thread = threading.Thread(target=animate)
        self.thread.daemon = True
        self.thread.start()
    
    def update_message(self, new_message: str):
        """Update the current message without stopping animation"""
        self.message = new_message
    
    def stop(self):
        """Stop the progress indicator"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def show_status(self, emoji: str, message: str, duration: float = 0):
        """Show a status message with emoji
        
        Args:
            emoji: Emoji to display
            message: Status message
            duration: How long to display (0 = instant)
        """
        sys.stdout.write(f'\r{emoji} {message}')
        sys.stdout.flush()
        if duration > 0:
            time.sleep(duration)
        print()  # New line after status


class NodeProgress:
    """Context manager for node-specific progress indication"""
    
    def __init__(self, node_name: str, thinking: bool = False):
        self.node_name = node_name
        self.thinking = thinking
        self.indicator = ProgressIndicator()
        
    def __enter__(self):
        if self.thinking:
            self.indicator.show_status("🧠", f"Node '{self.node_name}' is thinking...", 0)
            self.indicator.start_thinking()
        else:
            self.indicator.show_status("⚙️", f"Node '{self.node_name}' is processing...", 0)
            self.indicator.start_spinner(f"Running {self.node_name}")
        return self.indicator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indicator.stop()
        if exc_type is None:
            self.indicator.show_status("✅", f"Node '{self.node_name}' completed", 0)


def show_step_progress(step_name: str, action: Callable, thinking: bool = False) -> any:
    """Execute an action with progress indication
    
    Args:
        step_name: Name of the step
        action: Function to execute
        thinking: Whether this is a thinking/reasoning step
        
    Returns:
        Result of the action
    """
    with NodeProgress(step_name, thinking) as progress:
        result = action()
    return result


# Singleton instance for easy access
progress = ProgressIndicator() 