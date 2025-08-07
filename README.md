# Fine-Tuning Open-Source Large Language Models for Cybersecurity and IT Support: A Comparative Study on LLM Fine-Tuning Techniques

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.36%2B-yellow.svg)
![Datasets](https://img.shields.io/badge/Datasets-2.14%2B-green.svg)
![PEFT](https://img.shields.io/badge/PEFT-0.7%2B-red.svg)
![TRL](https://img.shields.io/badge/TRL-0.7%2B-purple.svg)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-0.41%2B-blue.svg)
![Accelerate](https://img.shields.io/badge/Accelerate-0.25%2B-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-lightblue.svg)
![Predibase](https://img.shields.io/badge/Predibase-2024.7%2B-green.svg)
![Wandb](https://img.shields.io/badge/Wandb-0.16%2B-yellow.svg)
![LiteLLM](https://img.shields.io/badge/LiteLLM-1.7%2B-purple.svg)

A comprehensive project for Large Language Model (LLM) fine-tuning specialized in cybersecurity, cloud computing, and IT support applications. This project combines synthetic data generation, Supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) techniques Like LoRA and QLoRA to create domain-specific conversational AI assistants.

This research case study is supported by Birmingham City University and METCLOUD, as part of the Knowledge TransferPartnership (KTP) (Project No: 10053887) funded by Innovate UK Research and Innovation (UKRI) Department, UK.

<div align="center">
  <table>
    <tr>
      <td align="center" style="padding: 0 30px;">
        <img src="logos/bcu.png" alt="Birmingham City University" height="60"/>
      </td>
      <td align="center" style="padding: 0 30px;">
        <img src="logos/metcloud.png" alt="METCLOUD" height="80"/>
      </td>
      <td align="center" style="padding: 0 30px;">
        <img src="logos/innovateuk.png" alt="Innovate UK" height="60"/>
      </td>
    </tr>
  </table>
</div>

## Project Architecture

```
Code/
├── Dataset/             
│   └── dataset.xlsx               # Main dataset file
├── Synthetic-Data-Generator/      # Synthetic dataset creation pipeline
├── Finetuning/                    # Model fine-tuning pipelines
│   ├── lora-finetuning/           # LoRA fine-tuning
│   ├── qlora-finetuning/          # QLoRA fine-tuning
│   └── sft-finetuning/            # Supervised fine-tuning config
└── README.md                      # Documentation guidelines
```

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Component Documentation](#component-documentation)
  - [Synthetic Data Generator](#synthetic-data-generator)
  - [LoRA Fine-tuning](#lora-fine-tuning)
  - [QLoRA Fine-tuning](#qlora-fine-tuning)
  - [Dataset Management](#dataset-management)
- [Supported Models](#supported-models)
- [Installation Guide](#installation-guide)
- [Usage Examples](#usage-examples)
- [Architecture Decisions](#architecture-decisions)
- [Testing &amp; Evaluation](#testing--evaluation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Research Context](#research-context)

## Overview

This project addresses the critical need for domain-specific Large Language Models in cybersecurity and IT support through a complete pipeline that includes:

1. **Synthetic Data Generation**: Creates diverse, high-quality training datasets using topic trees and parallel generation
2. **Parameter-Efficient Fine-tuning**: Implements both LoRA and QLoRA techniques for memory-efficient model adaptation
3. **Multi-Model Support**: Standardized configurations for 7+ popular LLM architectures
4. **Reproducible Research**: Consistent hyperparameters and evaluation protocols across all models

### Key Benefits

- **Domain Specialization**: Focused on cybersecurity, cloud computing, and IT support
- **Cost Efficiency**: Parameter-efficient techniques reduce training costs by 85%
- **Accessibility**: Consumer-grade hardware compatibility (RTX 3090/4090)
- **Scalability**: Parallel processing and batch optimization
- **Research-Ready**: Comprehensive logging and experiment tracking

## Prerequisites

### Hardware Requirements

| Component         | Minimum             | Recommended                     |
| ----------------- | ------------------- | ------------------------------- |
| **GPU**     | 8GB VRAM (RTX 3070) | 16GB+ VRAM (RTX 4090/L40S/A100) |
| **RAM**     | 16GB                | 32GB+                           |
| **Storage** | 50GB free           | 100GB+ SSD                      |
| **CPU**     | 8 cores             | 16+ cores                       |

### Software Requirements

- **Python**: 3.11+ (3.12 recommended)
- **CUDA**: 11.8+ or 12.1+ (for GPU acceleration)

### API Keys & Access

- **Predibase API Token** (for LoRA fine-tuning)
- **Hugging Face Token** (for model access)
- **Weights & Biases API Key** (optional, for experiment tracking)
- **LLM Provider API Keys** (OpenAI, Anthropic, Gemini etc. for synthetic data generation)

## Synthetic Data Generator

**Location**: `Synthetic-Data-Generator/`

The synthetic data generator creates diverse, high-quality question-answer pairs for training conversational AI models. It uses a sophisticated topic tree approach to ensure comprehensive coverage of cybersecurity and IT domains.

#### Architecture

```
Synthetic-Data-Generator/
├── pluto/                     # Core generation engine
│   ├── data_engine.py        # Main data generation logic
│   ├── dataset.py            # Dataset management
│   ├── prompts.py            # Prompt templates
│   ├── topic_tree.py         # Hierarchical topic generation
│   └── utils.py              # Utility functions
├── app.py                    # Streamlit web interface
├── jsonl_to_csv.py          # Format conversion utilities
└── requirements.txt          # Dependencies
```

#### Key Features

- **Topic Tree Generation**: Creates hierarchical topic structures for comprehensive coverage
- **Parallel Processing**: Efficient batch generation using multiple API calls
- **Multi-Provider Support**: Works with OpenAI, Anthropic, and other LLM providers
- **Quality Control**: Built-in validation and filtering mechanisms
- **Format Flexibility**: Outputs in JSONL and CSV formats

#### Configuration

Edit the topic tree and prompts in:

- `pluto/topic_tree.py`: Define domain-specific topics
- `pluto/prompts.py`: Customize system prompts and templates
- `Synthetic-Data-Topics.xlsx`: Structured topic definitions

## LoRA Fine-tuning

**Location**: `Finetuning/lora-finetuning/`

Production-ready LoRA fine-tuning pipeline using the Predibase platform for cloud-based, scalable model adaptation.

#### Architecture

```
lora-finetuning/
├── src/
│   ├── config.yaml           # Centralized model configurations
│   ├── lora_finetuning_pipeline.py  # Main training pipeline
│   └── utils.py              # Helper functions
├── data/                     # Model-specific datasets
│   ├── Google-Gemma/
│   ├── Meta-Llama3.1/
│   ├── Microsoft-Phi/
│   ├── Mistral/
│   └── Zephyr/
├── Dataprep_scripts/         # Data preprocessing notebooks
└── requirements.txt
```

#### Key Features

- **Cloud-Based Training**: Leverages Predibase infrastructure
- **Multi-Model Support**: 7 pre-configured model architectures
- **Standardized Hyperparameters**: Consistent training across all models
- **Automated Pipeline**: End-to-end training with minimal setup
- **Experiment Tracking**: Built-in Weights & Biases integration

## QLoRA Fine-tuning

**Location**: `Finetuning/qlora-finetuning/`

Local QLoRA fine-tuning framework for memory-efficient training on consumer hardware using 4-bit quantization.

#### Architecture

```
qlora-finetuning/
├── src/qlora_finetuning/     # Main package
│   ├── __main__.py           # Entry point
│   ├── train.py              # Training script
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   └── trainer.py        # Training logic
│   ├── models/
│   │   └── registry.py       # Model registry
│   └── utils/
│       ├── auth.py           # Authentication
│       ├── logging.py        # Logging setup
│       └── system.py         # System monitoring
├── configs/                  # Pre-built configurations
│   ├── gemma_config.json
│   ├── mistral_config.json
│   ├── phi_config.json
│   └── high_performance_config.json
└── data/                     # Training datasets
```

#### Key Features

- **Memory Efficiency**: 85% memory reduction vs. full fine-tuning
- **4-bit Quantization**: NF4 quantization with double quantization
- **Consumer Hardware**: Runs on RTX 3090/4090 level GPUs
- **Package Structure**: Clean, modular codebase
- **Comprehensive Logging**: Detailed training metrics and system monitoring

#### Memory Requirements

| Model Size | Traditional | QLoRA     | Hardware  |
| ---------- | ----------- | --------- | --------- |
| 2B params  | 16GB VRAM   | 4GB VRAM  | RTX 3070+ |
| 7B params  | 65GB VRAM   | 9GB VRAM  | RTX 3090+ |
| 13B params | 120GB VRAM  | 16GB VRAM | RTX 4090+ |

## Dataset Management

**Location**: `Dataset/`

Centralized dataset storage and management for training and evaluation.

#### Structure

- `dataset.xlsx`: Primary dataset containing cybersecurity Q&A pairs
- Model-specific subdirectories in each fine-tuning folder
- Standardized CSV format across all components

#### Data Processing Notebooks

Located in `Finetuning/lora-finetuning/Dataprep_scripts/`:

- `Dataset_Prep_Google_Gemma.ipynb`
- `Dataset_Prep_Llama3.1.ipynb`
- `Dataset_Prep_Microsoft-Phi.ipynb`
- `Dataset_Prep_Mistral.ipynb`
- `Dataset_Prep_Zephyr.ipynb`

## Installation Guide

### Detailed Setup Instructions

#### 1. System Prerequisites

```powershell
# Verify Python version
python --version  # Should be 3.11+

# Check CUDA availability (for GPU acceleration)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 2. Repository Setup

```powershell
# Clone and navigate
git clone https://github.com/your-org/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline

# Verify structure
dir  # Should show Dataset/, Finetuning/, Synthetic-Data-Generator/
```

#### 3. Component-Specific Installation

**For Synthetic Data Generation:**

```powershell
cd Synthetic-Data-Generator
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**For LoRA Fine-tuning:**

```powershell
cd Finetuning\lora-finetuning
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**For QLoRA Fine-tuning:**

```powershell
cd Finetuning\qlora-finetuning
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: Install Flash Attention for performance
pip install flash-attn --no-build-isolation
```

#### 4. Environment Configuration

Create and populate `.env` files:

```powershell
# Example .env for Synthetic Data Generator
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
LITELLM_LOG=INFO

# Example .env for LoRA Fine-tuning
PREDIBASE_API_TOKEN=your-predibase-token
WANDB_API_KEY=your-wandb-key
HUGGINGFACE_TOKEN=hf_your-token

# Example .env for QLoRA Fine-tuning
HUGGINGFACE_TOKEN=hf_your-token
WANDB_API_KEY=your-wandb-key
TOKENIZERS_PARALLELISM=false
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-model-support`
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Submit pull request** with detailed description

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**

```powershell
# Reduce batch size in config
# For QLoRA:
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8
}

# Monitor GPU usage
nvidia-smi -l 1
```

**API Rate Limiting:**

```python
# Add delays in synthetic data generation
import time
time.sleep(1)  # 1 second between API calls
```

**Model Loading Errors:**

```powershell
# Clear cache and reinstall
pip cache purge
pip uninstall transformers torch
pip install transformers torch --upgrade
```

**Import Errors:**

```powershell
# Verify virtual environment activation
echo $env:VIRTUAL_ENV  # Should show venv path

# Reinstall in development mode
pip install -e .
```

### Performance Optimization

**For Limited VRAM:**

```json
{
  "load_in_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_use_double_quant": true,
  "gradient_checkpointing": true
}
```

**For Slow Training:**

```json
{
  "dataloader_num_workers": 4,
  "fp16": true,
  "gradient_accumulation_steps": 8
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

We extend our sincere thanks to the following researchers and industry partner who contributed their expertise and effort to this project:

**Birmingham City University - Faculty of Computing, Engineering and Built Environment:**

- **Chaithanya Vamshi Sai**, **Nouh Sabri Elmitwally**, **Iain Rice**, **Haitham Mahmoud**

**METCLOUD (Managed Enterprise Technologies Limited)**

- **Ian Vickers**, **Xavier Schmoor**
