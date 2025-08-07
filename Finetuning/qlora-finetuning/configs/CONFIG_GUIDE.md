# Configuration Guide

This document explains the configuration parameters and provides standardized configurations for reproducible results.

## Configuration Parameters

### Core Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | "./results" | Directory to save model outputs |
| `per_device_train_batch_size` | int | 1 | Training batch size per GPU |
| `per_device_eval_batch_size` | int | 1 | Evaluation batch size per GPU |
| `gradient_accumulation_steps` | int | 8 | Steps to accumulate gradients |
| `num_train_epochs` | int | 3 | Number of training epochs |
| `learning_rate` | float | 2e-4 | Learning rate for optimizer |
| `warmup_steps` | int | 10 | Linear warmup steps |

### Evaluation & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging_steps` | int | 1 | Log metrics every N steps |
| `eval_steps` | int | 50 | Evaluate every N steps |
| `save_steps` | int | 50 | Save checkpoint every N steps |
| `test_size` | float | 0.1 | Fraction of data for testing |

### Performance Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp16` | bool | false | Use 16-bit floating point |
| `bf16` | bool | true | Use bfloat16 (recommended for modern GPUs) |
| `use_flash_attention` | bool | true | Enable Flash Attention 2 |

### Model Sharing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `push_to_hub` | bool | true | Push trained model to HF Hub |
| `private_hub` | bool | true | Make repository private |

## Standardized Configurations

All model configurations use the same baseline parameters for reproducibility:

### Default Configuration
```json
{
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "eval_steps": 50,
    "save_steps": 50,
    "bf16": true,
    "use_flash_attention": true
}
```

**Effective Batch Size**: 1 × 8 = 8 samples per update
**Total Training Steps**: Approximately (dataset_size × epochs) / effective_batch_size

### Memory Requirements

| Configuration | GPU Memory | Effective Batch Size | Training Time (approx) |
|---------------|------------|---------------------|------------------------|
| Default | 12-16GB | 8 | Baseline |
| Low Memory | 8-12GB | 16 | +20% slower |
| High Performance | 20-24GB | 8 | -30% faster |

## Configuration Files

### Available Configurations

1. **`default_config.json`** - Balanced performance and memory usage
2. **`low_memory_config.json`** - Optimized for limited GPU memory
3. **`high_performance_config.json`** - Optimized for speed with high-end GPUs
4. **`gemma_config.json`** - Model-specific optimizations for Gemma
5. **`mistral_config.json`** - Model-specific optimizations for Mistral
6. **`phi_config.json`** - Model-specific optimizations for Phi

### Usage Examples

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qlora_finetuning import QLorAFineTuner, TrainingConfig

# Use default configuration
config = TrainingConfig()
trainer = QLorAFineTuner("phi-2", config)

# Load from JSON configuration
config = TrainingConfig.from_json("configs/phi_config.json")
trainer = QLorAFineTuner("mistral-7b-instruct", config)

# Use low memory configuration for any model
config = TrainingConfig.from_json("configs/low_memory_config.json")
trainer = QLorAFineTuner("gemma-2-2b-instruct", config)
```

## Customization Guidelines

### Adjusting for Your Hardware

#### Limited GPU Memory (8-12GB)
```json
{
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "fp16": true,
    "bf16": false,
    "use_flash_attention": false
}
```

#### High-End GPU (24GB+)
```json
{
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "bf16": true,
    "use_flash_attention": true
}
```

### Learning Rate Guidelines

| Model Size | Recommended LR | Notes |
|------------|---------------|-------|
| 2B parameters | 2e-4 | Standard rate |
| 7B parameters | 1e-4 to 2e-4 | Lower rate for stability |
| 13B+ parameters | 5e-5 to 1e-4 | Very conservative |

### Epoch Guidelines

| Dataset Size | Recommended Epochs | Notes |
|--------------|-------------------|-------|
| < 1K samples | 5-10 | More epochs needed |
| 1K-5K samples | 3-5 | Standard training |
| 5K-10K samples | 2-3 | Avoid overfitting |
| 10K+ samples | 1-2 | Large dataset |

## Experimental Configurations

### Quick Test Configuration
```json
{
    "num_train_epochs": 1,
    "eval_steps": 10,
    "save_steps": 10,
    "push_to_hub": false
}
```

### Research Configuration
```json
{
    "num_train_epochs": 5,
    "eval_steps": 25,
    "save_steps": 25,
    "learning_rate": 1e-4,
    "warmup_steps": 50
}
```

## Configuration Impact

### Batch Size vs Memory

```
Batch Size 1: ~8GB VRAM
Batch Size 2: ~14GB VRAM  
Batch Size 4: ~24GB VRAM
```

### Gradient Accumulation vs Speed

```
Steps 4:  Fast training, higher memory
Steps 8:  Balanced (recommended)
Steps 16: Slower training, lower memory
```

## Best Practices

1. **Start with default configuration** for initial experiments
2. **Use consistent settings** across model comparisons
3. **Monitor GPU memory usage** during training
4. **Adjust batch size and accumulation** based on hardware
5. **Keep effective batch size constant** when comparing models
6. **Use bf16 on modern GPUs** (RTX 30/40 series, A100, H100)
7. **Enable Flash Attention** when available for 30-50% speedup

## Troubleshooting

### Out of Memory Errors
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Use `fp16` instead of `bf16`
4. Disable `use_flash_attention`

### Slow Training
1. Increase `per_device_train_batch_size`
2. Decrease `gradient_accumulation_steps`
3. Enable `use_flash_attention`
4. Use `bf16` if supported

### Poor Convergence
1. Reduce `learning_rate`
2. Increase `warmup_steps`
3. Increase `num_train_epochs`
4. Check dataset quality

## License

This configuration guide is part of the QLora Fine-tuning Framework, licensed under the Apache License 2.0.
