"""
Logging utilities for the QLora fine-tuning package.

This module provides standardized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger('qlora_finetuning')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f'qlora_finetuning.{name}')


class TrainingLogger:
    """Specialized logger for training events."""
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)
    
    def log_training_start(self, model_name: str, config: dict) -> None:
        """Log training start event."""
        self.logger.info(f"Starting training for model: {model_name}")
        self.logger.info(f"Configuration: {config}")
    
    def log_training_end(self, model_name: str, success: bool) -> None:
        """Log training end event."""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Training for {model_name} {status}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_checkpoint_saved(self, step: int, path: str) -> None:
        """Log checkpoint save event."""
        self.logger.info(f"Checkpoint saved at step {step}: {path}")
    
    def log_model_pushed(self, model_name: str, hub_name: str) -> None:
        """Log model push to hub."""
        self.logger.info(f"Model {model_name} pushed to hub: {hub_name}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context."""
        error_msg = f"Error in {context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg, exc_info=True)
