"""
Model registry for supported LLM configurations.

This module maintains the registry of supported models and their configurations.
"""

from typing import Dict
from ..core.config import ModelConfig


class ModelRegistry:
    """Registry for supported model configurations."""
    
    _models: Dict[str, ModelConfig] = {
        "gemma-2-2b": ModelConfig(
            base_model="google/gemma-2-2b",
            dataset_name="chaithanyasai/MET-Data-Gemma-5K",
            new_model_name="Gemma-2-2B-MET-qlora",
            wandb_project="Google-Gemma2-2B-MET-qlora",
            wandb_run_name="Gemma2-2B-MET-qlora-run",
        ),
        "gemma-2-2b-instruct": ModelConfig(
            base_model="google/gemma-2-2b-it",
            dataset_name="chaithanyasai/MET-Data-Gemma-5K",
            new_model_name="Gemma-2-2B-Instruct-MET-qlora",
            wandb_project="Google-Gemma2-2B-Instruct-MET-qlora",
            wandb_run_name="Gemma2-2B-Instruct-MET-qlora-run",
        ),
        "mistral-7b-instruct": ModelConfig(
            base_model="mistralai/Mistral-7B-Instruct-v0.3",
            dataset_name="chaithanyasai/MET-Data-Mistral-5K",
            new_model_name="Mistral-7B-Instruct-MET-qlora",
            wandb_project="Mistral-7B-Instruct-MET-qlora",
            wandb_run_name="Mistral-7B-Instruct-MET-qlora-run",
        ),
        "phi-3-mini": ModelConfig(
            base_model="microsoft/Phi-3-mini-4k-instruct",
            dataset_name="chaithanyasai/MET-Data-Phi-5K",
            new_model_name="Microsoft-Phi-3B-MET-qlora",
            wandb_project="Microsoft-Phi-3B-MET-qlora",
            wandb_run_name="Microsoft-Phi-3B-MET-qlora-run",
        ),
        "phi-2": ModelConfig(
            base_model="microsoft/phi-2",
            dataset_name="chaithanyasai/MET-Data-Phi-5K",
            new_model_name="Microsoft-Phi-2-MET-qlora",
            wandb_project="Microsoft-Phi-2-MET-qlora",
            wandb_run_name="Microsoft-Phi-2-MET-qlora-run",
        ),
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")
        return cls._models[model_name]
    
    @classmethod
    def list_models(cls) -> list[str]:
        """Get list of supported model names."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, config: ModelConfig) -> None:
        """Register a new model configuration."""
        cls._models[name] = config
    
    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in cls._models
