"""
Configuration classes for QLora fine-tuning.

This module contains data classes that define model and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration class for model-specific parameters."""
    
    base_model: str
    dataset_name: str
    new_model_name: str
    wandb_project: str
    wandb_run_name: str
    target_modules: List[str] = field(default_factory=lambda: ['k_proj', 'q_proj', 'v_proj', 'o_proj'])
    max_seq_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lora_r <= 0:
            raise ValueError("LoRA rank (r) must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("LoRA dropout must be between 0 and 1")
        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    output_dir: str = "./results"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 1
    eval_steps: int = 50
    save_steps: int = 50
    test_size: float = 0.1
    fp16: bool = False
    bf16: bool = True
    use_flash_attention: bool = True
    push_to_hub: bool = True
    private_hub: bool = True
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.per_device_train_batch_size <= 0:
            raise ValueError("Train batch size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("Eval batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("Test size must be between 0 and 1")
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate the effective batch size."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Create configuration from JSON file."""
        import json
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
