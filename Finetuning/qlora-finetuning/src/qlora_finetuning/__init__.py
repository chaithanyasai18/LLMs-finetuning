"""
QLora Fine-tuning Package
========================

A Python package for fine-tuning Large Language Models using QLora technique.

This package provides:
- Model configurations for various LLMs (Gemma, Mistral, Phi)
- Training pipeline with standardized configurations
- Memory-efficient fine-tuning using QLora
- Experiment tracking and model management
- Professional logging and error handling

"""

from .core.trainer import QLorAFineTuner
from .core.config import ModelConfig, TrainingConfig
from .models.registry import ModelRegistry

__all__ = [
    "QLorAFineTuner",
    "ModelConfig", 
    "TrainingConfig",
    "ModelRegistry",
]
