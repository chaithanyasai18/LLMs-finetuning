# QLora LLM Fine-tuning

A comprehensive Python framework for fine-tuning large language models using Quantized Low-Rank Adaptation (QLora) technique. This framework provides efficient fine-tuning of multiple model architectures with standardized configurations and reproducible results.

## Features

- **Clean Package Structure**: Organized source code in src/ directory
- **Multiple Model Support**: Google Gemma 2, Mistral 7B, Microsoft Phi-3, and more
- **Efficient QLora Training**: 4-bit quantization with Low-Rank Adaptation
- **Standardized Configurations**: Consistent training parameters across all models
- **Comprehensive Logging**: Detailed training logs and system monitoring
- **Weights & Biases Integration**: Built-in experiment tracking and visualization
- **Memory Optimization**: Gradient checkpointing and efficient data loading
- **Reproducible Results**: Seed control and deterministic training

## QLora Explained

**QLora (Quantized Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that combines quantization and Low-Rank Adaptation (LoRA) to enable efficient training of large language models on consumer-grade hardware.

### Technical Overview

#### Low-Rank Adaptation (LoRA)

LoRA decomposes weight updates into low-rank matrices, dramatically reducing trainable parameters:

```
Original: W ∈ R^(d×k) → Full fine-tuning updates entire weight matrix
LoRA: W + ΔW = W + BA → Where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

- **W**: Original pre-trained weights (frozen)
- **B, A**: Trainable low-rank matrices
- **r**: Rank (typically 4-64, much smaller than d,k)

#### Quantization in QLora

QLora uses **4-bit NormalFloat (NF4)** quantization:

1. **4-bit Quantization**: Reduces memory by 4x compared to 16-bit
2. **NF4 Data Type**: Optimized for normally distributed weights
3. **Double Quantization**: Quantizes the quantization constants themselves
4. **Paged Optimizers**: Uses NVIDIA unified memory for large optimizer states

### QLora Benefits

| Aspect                          | Traditional Fine-tuning | QLora             | Improvement                   |
| ------------------------------- | ----------------------- | ----------------- | ----------------------------- |
| **Memory Usage**          | 65GB (7B model)         | 9GB (7B model)    | **85% reduction**       |
| **Training Speed**        | Baseline                | 0.8-1.2x          | **Similar or better**   |
| **Quality**               | 100%                    | 97-99%            | **Minimal degradation** |
| **Hardware Requirements** | A100 80GB               | RTX 3090/4090/L40S    | **Consumer accessible** |
| **Trainable Parameters**  | 7B parameters           | 16-32M parameters | **99% reduction**       |

## Supported Models

- **Google Gemma 2**: 2B and 2B-Instruct variants
- **Mistral**: 7B Instruct model
- **Microsoft Phi**: Phi-2 and Phi-3 Mini models
- **Meta Llama**: 3.1 models (configurable)
- **Zephyr**: 7B models

Each model includes optimized configurations for QLora fine-tuning with standardized parameters.

## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory (for 2B models), 16GB+ (for 7B models)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/qlora-finetuning.git
cd qlora-finetuning

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Project Structure

```
qlora-finetuning/
├── src/qlora_finetuning/          # Main package source code
│   ├── __init__.py                 # Package initialization
│   ├── __main__.py                 # Main entry point
│   ├── train.py                    # Training script
│   ├── core/                       # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration classes
│   │   └── trainer.py             # Main trainer class
│   ├── models/                     # Model registry and configs
│   │   ├── __init__.py
│   │   └── registry.py            # Model registry
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── auth.py                # Authentication utilities
│       ├── logging.py             # Logging configuration
│       └── system.py              # System utilities
├── configs/                        # Model configurations
│   ├── CONFIG_GUIDE.md            # Configuration documentation
│   ├── gemma_config.json
│   ├── mistral_config.json
│   └── phi_config.json
├── data/                          # Training datasets
│   ├── Google-Gemma/
│   ├── Meta-Llama3.1/
│   ├── Microsoft-Phi/
│   ├── Mistral/
│   └── Zephyr/
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

## Usage

### Quick Start - How to Run

There are multiple ways to run the training:

#### Method 1: Using the Python Module (Recommended)
```bash
# Run as Python module
cd qlora-finetuning
python -m src.qlora_finetuning

# Or run the main script directly
python src/qlora_finetuning/__main__.py
```

#### Method 2: Using the Training Script
```bash
# Run the training script directly
python src/qlora_finetuning/train.py
```

#### Method 3: Import and Use in Python/Jupyter
```python
# Create a new Python script or Jupyter notebook
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qlora_finetuning import QLorAFineTuner, TrainingConfig

# Your training code here...
```

### Complete Training Example

Here's a complete example to train a model from start to finish:

```python
# save this as train_example.py
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qlora_finetuning import QLorAFineTuner, TrainingConfig, ModelRegistry

def main():
    print("Starting QLora Fine-tuning...")
    
    # 1. List available models
    models = ModelRegistry.list_models()
    print(f"Available models: {models}")
    
    # 2. Choose your model and dataset
    model_name = "phi-2"  # Change this to your preferred model
    dataset_path = "data/Microsoft-Phi/MET-Data-Phi-5K.csv"  # Update path to your dataset
    
    # 3. Create training configuration
    config = TrainingConfig(
        output_dir=f"./results/{model_name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        push_to_hub=False,  # Set to True if you want to upload to HuggingFace
    )
    
    # 4. Initialize trainer
    trainer = QLorAFineTuner(model_name, config)
    
    # 5. Set up authentication (make sure .env is configured)
    print("Setting up authentication...")
    trainer.setup_authentication()
    
    # 6. Check system requirements
    print("Checking system requirements...")
    trainer.check_system_requirements()
    
    # 7. Load model and tokenizer
    print("Loading model and tokenizer...")
    trainer.load_model_and_tokenizer()
    
    # 8. Configure LoRA
    print("Configuring LoRA...")
    trainer.configure_peft()
    
    # 9. Load and prepare dataset
    print("Loading dataset...")
    trainer.load_and_prepare_dataset(dataset_path)
    
    # 10. Set up trainer
    print("Setting up trainer...")
    trainer.setup_trainer()
    
    # 11. Start training
    print("Starting training...")
    trainer.train()
    
    # 12. Test the model (optional)
    print("Testing model...")
    test_prompt = "What is cybersecurity?"
    result = trainer.test_model(test_prompt)
    print(f"Test result: {result}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

Then run it:
```bash
python train_example.py
```

### Basic Usage

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qlora_finetuning import QLorAFineTuner, TrainingConfig, ModelRegistry

# List available models
models = ModelRegistry.list_models()
print(f"Available models: {models}")

# Create training configuration
config = TrainingConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4
)

# Initialize trainer
trainer = QLorAFineTuner("phi-2", config)

# Set up authentication
trainer.setup_authentication()

# Train the model
trainer.load_model_and_tokenizer()
trainer.configure_peft()
trainer.load_and_prepare_dataset()
trainer.setup_trainer()
trainer.train()
```

### Using Configuration Files

```python
# Load from JSON config
config = TrainingConfig.from_json("configs/phi_config.json")
trainer = QLorAFineTuner("phi-2", config)
```

### Advanced Configuration

```python
# Custom training configuration
config = TrainingConfig(
    output_dir="./custom_results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=100,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    push_to_hub=True,
    private_hub=False
)

# Custom model configuration
model_config = ModelConfig(
    model_name="microsoft/phi-2",
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.1
)
```

## Configuration

### Environment Variables

Create a `.env` file with your API tokens:

```bash
# Hugging Face Authentication
HUGGINGFACE_TOKEN=hf_your_token_here

# Weights & Biases Authentication
WANDB_TOKEN=your_wandb_token_here

# CUDA Configuration (Optional)
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

Each model has a JSON configuration file in the `configs/` directory:

```json
{
  "model_config": {
    "model_name": "microsoft/phi-2",
    "use_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
  },
  "training_config": {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "learning_rate": 2e-4
  }
}
```

## Data Format

The framework expects CSV files with the following columns:

```csv
prompt,response
"How do I configure a firewall?","To configure a firewall, follow these steps..."
"What is cloud computing?","Cloud computing is a model for..."
```

### Data Processing

The package automatically formats prompts using the Alpaca template:

```
### Instruction:
{prompt}

### Response:
{response}
```

## Training Process

1. **Model Loading**: Load quantized model with 4-bit precision
2. **LoRA Configuration**: Apply Low-Rank Adaptation layers
3. **Dataset Preparation**: Format and tokenize training data
4. **Training**: Execute supervised fine-tuning with QLora
5. **Evaluation**: Monitor loss and performance metrics
6. **Saving**: Save LoRA adapters and optionally merge with base model

## Memory Requirements

### Minimum Requirements

- **2B models**: 8GB GPU memory
- **7B models**: 16GB GPU memory
- **System RAM**: 16GB recommended

### Recommended Setup

- **GPU**: RTX 3090/4090, A100, or equivalent
- **Memory**: 24GB+ GPU memory for comfortable training
- **Storage**: 50GB+ free space for models and checkpoints

## Monitoring

### Weights & Biases Integration

The framework includes built-in W&B integration for experiment tracking:

- Training loss and metrics
- Model configuration
- System information
- GPU utilization
- Training progress

### System Monitoring

```python
from qlora_finetuning.utils.system import get_system_info, get_memory_usage

# Get system information
info = get_system_info()
print(f"GPU available: {info['gpu_available']}")

# Monitor memory usage
memory = get_memory_usage()
print(f"Memory usage: {memory['percent']}%")
```

## Model Registry

The framework includes a model registry for easy model management:

```python
from qlora_finetuning import ModelRegistry

# List all available models
models = ModelRegistry.list_models()

# Get model configuration
config = ModelRegistry.get_model_config("gemma-2-2b-instruct")

# Register custom model
custom_config = ModelConfig(model_name="custom/model")
ModelRegistry.register_model("custom-model", custom_config)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src/ is in your Python path
2. **GPU Memory**: Reduce batch size or use gradient accumulation
3. **Authentication**: Check your .env file and API tokens
4. **Model Loading**: Verify model names and availability

### Performance Tips

1. **Use Flash Attention**: Uncomment flash-attn in requirements.txt
2. **Gradient Checkpointing**: Enable for memory savings
3. **Mixed Precision**: Use bf16 on modern GPUs
4. **Batch Size**: Increase with more GPU memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Update documentation as needed
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in configuration
2. **Authentication errors**: Check your .env file and HuggingFace token
3. **Dataset loading issues**: Verify CSV format and file paths
4. **Import errors**: Ensure all requirements are installed

### Getting Help

For detailed configuration options, see `configs/CONFIG_GUIDE.md`.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
