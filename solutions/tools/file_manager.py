"""
File Manager for ASU Tapeout Agent
==================================

Handles all file I/O operations with proper error handling and path management.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml


class FileManager:
    """Manages file operations for the tapeout agent"""
    
    def __init__(self, base_dir: str = "."):
        """Initialize file manager with base directory"""
        self.base_dir = Path(base_dir)
        self.temp_dir = self.base_dir / "temp"
        self.output_dir = self.base_dir / "output"
        
    def setup_directories(self):
        """Create necessary directories"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_file(self, filepath: str, content: str) -> bool:
        """Save content to file
        
        Args:
            filepath: Path to save file
            content: Content to write
            
        Returns:
            Success status
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return True
        except Exception as e:
            print(f"Error saving file {filepath}: {e}")
            return False
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Read content from file
        
        Args:
            filepath: Path to read from
            
        Returns:
            File content or None if error
        """
        try:
            path = Path(filepath)
            if path.exists():
                return path.read_text()
            else:
                print(f"File not found: {filepath}")
                return None
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None
    
    def copy_file(self, src: str, dest: str) -> bool:
        """Copy file from source to destination
        
        Args:
            src: Source file path
            dest: Destination file path
            
        Returns:
            Success status
        """
        try:
            src_path = Path(src)
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            return True
        except Exception as e:
            print(f"Error copying file from {src} to {dest}: {e}")
            return False
    
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern
        
        Args:
            directory: Directory to list
            pattern: Glob pattern to match
            
        Returns:
            List of file paths
        """
        try:
            dir_path = Path(directory)
            if dir_path.exists():
                return [str(f) for f in dir_path.glob(pattern)]
            else:
                return []
        except Exception as e:
            print(f"Error listing files in {directory}: {e}")
            return []
    
    def ensure_directory(self, directory: str) -> bool:
        """Ensure directory exists
        
        Args:
            directory: Directory path
            
        Returns:
            Success status
        """
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    def get_temp_filepath(self, filename: str) -> str:
        """Get path for temporary file
        
        Args:
            filename: Name of temporary file
            
        Returns:
            Full path to temp file
        """
        self.ensure_directory(str(self.temp_dir))
        return str(self.temp_dir / filename)
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error cleaning temp files: {e}")
    
    def save_json(self, filepath: str, data: Dict[str, Any]) -> bool:
        """Save dictionary as JSON file
        
        Args:
            filepath: Path to save JSON
            data: Dictionary to save
            
        Returns:
            Success status
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False
    
    def load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load JSON file as dictionary
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary or None if error
        """
        try:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                print(f"JSON file not found: {filepath}")
                return None
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}")
            return None
    
    def save_yaml(self, filepath: str, data: Dict[str, Any]) -> bool:
        """Save dictionary as YAML file
        
        Args:
            filepath: Path to save YAML
            data: Dictionary to save
            
        Returns:
            Success status
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving YAML to {filepath}: {e}")
            return False
    
    def load_yaml(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load YAML file as dictionary
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Dictionary or None if error
        """
        try:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"YAML file not found: {filepath}")
                return None
        except Exception as e:
            print(f"Error loading YAML from {filepath}: {e}")
            return None 