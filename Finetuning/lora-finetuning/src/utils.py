"""
Utilities module for LoRA Fine-tuning Pipeline
=============================================

This module provides utility functions, custom exceptions, and helper classes
for the LoRA fine-tuning pipeline.
"""

import os
import time
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import functools


class FineTuningError(Exception):
    """Custom exception for fine-tuning related errors."""
    pass


class DatasetError(Exception):
    """Custom exception for dataset related errors."""
    pass


class ModelConfigError(Exception):
    """Custom exception for model configuration errors."""
    pass


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator that retries a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise e
                    
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    logging.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def validate_dataset(dataset_path: str) -> bool:
    """
    Validate dataset file exists and has required format.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        True if dataset is valid
        
    Raises:
        DatasetError: If dataset is invalid
    """
    if not os.path.exists(dataset_path):
        raise DatasetError(f"Dataset file not found: {dataset_path}")
    
    if not dataset_path.endswith('.csv'):
        raise DatasetError(f"Dataset must be a CSV file: {dataset_path}")
    
    # Additional validation could be added here
    # e.g., checking required columns, data format, etc.
    
    return True


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation level
    """
    try:
        ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        raise IOError(f"Failed to save JSON file {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file with error handling.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data dictionary
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {filepath}: {e}")


class ProgressTracker:
    """Track and report progress of long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = None, message: str = None):
        """Update progress tracker."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        log_message = f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)"
        if message:
            log_message += f" - {message}"
        log_message += f" [Elapsed: {elapsed}]"
        
        logging.info(log_message)
    
    def complete(self, message: str = "Completed"):
        """Mark progress as complete."""
        self.current_step = self.total_steps
        elapsed = datetime.now() - self.start_time
        logging.info(f"{self.description}: {message} [Total time: {elapsed}]")


class ModelValidator:
    """Validate model configurations and parameters."""
    
    REQUIRED_FIELDS = [
        'name', 'base_model', 'client_name', 'dataset_path', 
        'repo_name', 'adapter_name', 'prompt_template'
    ]
    
    VALID_BASE_MODELS = [
        'mistral-7b', 'mistral-7b-instruct-v0-3', 'llama-3-8b-instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct', 'google/gemma-2b',
        'microsoft/Phi-3-mini-4k-instruct', 'HuggingFaceH4/zephyr-7b-beta'
    ]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ModelConfigError: If configuration is invalid
        """
        # Check required fields
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in config]
        if missing_fields:
            raise ModelConfigError(f"Missing required fields: {missing_fields}")
        
        # Validate base model
        if config['base_model'] not in cls.VALID_BASE_MODELS:
            raise ModelConfigError(f"Invalid base model: {config['base_model']}")
        
        # Validate numeric parameters
        if 'epochs' in config and (not isinstance(config['epochs'], int) or config['epochs'] <= 0):
            raise ModelConfigError("Epochs must be a positive integer")
        
        if 'rank' in config and (not isinstance(config['rank'], int) or config['rank'] <= 0):
            raise ModelConfigError("Rank must be a positive integer")
        
        if 'learning_rate' in config and (not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0):
            raise ModelConfigError("Learning rate must be a positive number")
        
        return True


class MetricsCollector:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str):
        """End timing an operation and record duration."""
        if operation in self.start_times:
            duration = datetime.now() - self.start_times[operation]
            self.metrics[f"{operation}_duration"] = duration.total_seconds()
            del self.start_times[operation]
    
    def record_metric(self, name: str, value: Any):
        """Record a custom metric."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def generate_summary(self) -> str:
        """Generate a summary of collected metrics."""
        if not self.metrics:
            return "No metrics collected."
        
        summary = ["Performance Metrics Summary:", "=" * 30]
        
        for name, value in self.metrics.items():
            if name.endswith('_duration'):
                summary.append(f"{name}: {value:.2f} seconds")
            else:
                summary.append(f"{name}: {value}")
        
        return "\n".join(summary)


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def check_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements for fine-tuning.
    
    Returns:
        Dictionary with system information
    """
    import psutil
    import platform
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_bytes(psutil.virtual_memory().total),
        "memory_available": format_bytes(psutil.virtual_memory().available),
        "disk_free": format_bytes(psutil.disk_usage('.').free)
    }


class ConfigManager:
    """Manage configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback to defaults."""
        if os.path.exists(self.config_path):
            return load_config(self.config_path)
        else:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "PIPELINE": {
                "max_retries": 3,
                "retry_delay": 10,
                "batch_size": 1,
                "parallel_processing": False
            },
            "LOGGING": {
                "level": "INFO",
                "log_file": "lora_finetuning.log"
            },
            "OUTPUT": {
                "results_directory": "./results",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "include_metadata": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filepath: Optional[str] = None) -> None:
        """Save configuration to file."""
        filepath = filepath or self.config_path
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
