# LoRA Fine-tuning Pipeline for Large Language Models (LLM)

## What is LoRA Fine-tuning?

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that reduces the number of trainable parameters for downstream tasks. Instead of fine-tuning all parameters of a pre-trained model, LoRA freezes the original weights and introduces trainable low-rank decomposition matrices into each layer of the transformer architecture.

**Key Benefits of LoRA:**

- **Memory Efficiency**: Reduces memory requirements by up to 3x ## Configuration Details
- **Storage Efficiency**: Adapter weights are typically only 1-2% of the original model size
- **Training Speed**: Faster training due to fewer parameters to update
- **Model Preservation**: Original model weights remain unchanged, allowing easy switching between tasks
- **Cost Effectiveness**: Lower computational costs make fine-tuning accessible for smaller organizations

### Model Configuration Structure

The pipeline uses a standardized configuration approach defined in `src/config.yaml`. All models share the same hyperparameters for consistency:

```yaml
STANDARD_CONFIG:
  epochs: 5                    # Sufficient for convergence without overfitting
  rank: 16                     # Optimal balance between expressiveness and efficiency
  learning_rate: 0.00005       # Conservative rate for stability across all model sizes
  max_new_tokens: 512          # Sufficient for detailed cybersecurity responses
  temperature: 0.1             # Low temperature for consistent, focused outputs
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention layer adaptation
```

Each model is configured with standardized parameters for reproducibility:@dataclass

```python
class ModelConfig:
    name: str                    # Display name
    base_model: str             # Predibase model identifier
    dataset_path: str           # Path to training data
    repository_name: str        # Consistent naming pattern
    adapter_name: str           # Consistent adapter naming
    hyperparameters: dict       # Standardized hyperparameters
    prompt_template: str        # Model-specific prompt format
```

**How LoRA Works:**
LoRA decomposes weight updates into two smaller matrices (A and B) where the rank (r) is much smaller than the original weight dimensions. The adaptation is applied as: W = W₀ + BA, where W₀ are the frozen pre-trained weights and BA represents the low-rank adaptation.

## Project Overview

This project provides a production-ready pipeline for fine-tuning multiple Large Language Models using LoRA technique via the Predibase platform. The pipeline supports 7 different model architectures with standardized configurations for reproducible results.

### Key Features

- **Standardized Configuration**: Consistent hyperparameters and naming conventions across all models
- **Model-Specific Prompt Templates**: Optimized conversation formats for each model family
- **Reproducible Results**: Standardized configurations ensure consistent outcomes
- **Multi-Model Support**: Support for 7 popular LLM architectures
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Flexible API**: Both programmatic and command-line interfaces available
- **Automated Data Preparation**: Template formatting for each model family

## Project Structure

```
LoRa-finetuning/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env                               # Environment variables (API keys)
│
├── data/                              # Training datasets
│   ├── meta_data/                     # Original source data
│   ├── Google-Gemma/                  # Prepared data for Google Gemma
│   ├── Meta-Llama3.1/                 # Prepared data for Llama 3.1
│   ├── Microsoft-Phi/                 # Prepared data for Microsoft Phi
│   ├── Mistral/                       # Prepared data for Mistral
│   └── Zephyr/                        # Prepared data for Zephyr
│
├── Dataprep/                          # Data preparation notebooks
│   ├── Dataset_Prep_Google_Gemma.ipynb
│   ├── Dataset_Prep_Llama3.1.ipynb
│   ├── Dataset_Prep_Microsoft-Phi.ipynb
│   ├── Dataset_Prep_Mistral.ipynb
│   └── Dataset_Prep_Zephyr.ipynb
│
└── src/                               # Main implementation
    ├── lora_finetuning_pipeline.py    # Main pipeline implementation
    ├── config.yaml                    # Configuration file
    └── utils.py                       # Utility functions
```

## Step 1: Prerequisites

### 1. Prerequisites & Setup

- Save your API token securely in .env file

2. **Python Environment**

   - Python 3.11 or higher
   - Python venv or conda environment

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd LoRa-finetuning
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt
```

**Core Dependencies:**

- `predibase>=2024.7.2` - LoRA fine-tuning platform
- `python-dotenv>=1.0.0` - Environment variable management
- `pandas>=2.0.0` - Data manipulation
- `langchain-community>=0.2.0` - LLM integration utilities
- `PyYAML>=6.0` - Configuration file parsing
- `rich>=13.5.0` - Enhanced console output
- `tqdm>=4.66.0` - Progress monitoring

**Optional Dependencies:**

- `wandb>=0.15.0` - Experiment tracking (requires WANDB_API_KEY)
- `pytest>=7.4.0` - Testing framework
- `black>=23.7.0` - Code formatting

### Step 3: Set up Environment

**Required Accounts:**

1. **Predibase Account** - Create at [Predibase](https://predibase.com/) and generate API token
2. **Hugging Face Account** (optional) - For model access tokens
3. **Weights & Biases Account** (optional) - For experiment tracking

Create a `.env` file in the root directory:

```bash
# Required
PREDIBASE_API_TOKEN=your_predibase_api_token_here

# Optional  
WANDB_API_KEY=your_wandb_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Quick Usage

### Show Model Information

```bash
cd src
python lora_finetuning_pipeline.py --info
```

### Show Configuration Rationale

```bash
python lora_finetuning_pipeline.py --config-rationale
```

### Configure Pipeline Settings

The pipeline can be configured via `src/config.yaml`. You can modify:

- Model-specific settings (dataset paths, naming conventions)
- Pipeline settings (retries, batch size)
- Logging configuration
- Test questions for evaluation

### Train All Models

```bash
python lora_finetuning_pipeline.py --models all --output results.json
```

### Test Models (No Training)

```bash
python lora_finetuning_pipeline.py --test-only --models all
```

### Compare Models

```bash
python lora_finetuning_pipeline.py --compare --output comparison.json
```

## Supported Models

| Model Key               | Full Name                     | Base Model                            | Learning Rate |
| ----------------------- | ----------------------------- | ------------------------------------- | ------------- |
| `mistral_7b_base`     | Mistral-7B-Base-CyberQA       | mistral-7b                            | 0.00005       |
| `mistral_7b_instruct` | Mistral-7B-Instruct-CyberQA   | mistral-7b-instruct-v0-3              | 0.00005       |
| `llama3_8b_instruct`  | Llama-3-8B-Instruct-CyberQA   | llama-3-8b-instruct                   | 0.00005       |
| `llama31_8b_instruct` | Llama-3.1-8B-Instruct-CyberQA | meta-llama/Meta-Llama-3.1-8B-Instruct | 0.00005       |
| `google_gemma_2b`     | Google-Gemma-2B-CyberQA       | google/gemma-2b                       | 0.00005       |
| `microsoft_phi3_mini` | Microsoft-Phi-3-Mini-CyberQA  | microsoft/Phi-3-mini-4k-instruct      | 0.00005       |
| `zephyr_7b_beta`      | Zephyr-7B-Beta-CyberQA        | HuggingFaceH4/zephyr-7b-beta          | 0.00005       |

## Standardized Configuration

**Consistent Hyperparameters for Reproducibility:**

- **Learning Rate**: 0.00005 (chosen for stability across all model sizes)
- **Epochs**: 5 (optimal balance between training time and performance)
- **Rank**: 16 (low-rank dimension for efficient adaptation)
- **Target Modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]` (attention layer adaptation)
- **Max New Tokens**: 512 (sufficient for detailed cybersecurity responses)
- **Temperature**: 0.1 (low temperature for consistent, focused outputs)

**Why These Configuration Choices:**

**Learning Rate (0.00005)**: This conservative learning rate ensures stable training across different model architectures without overfitting. Larger models (7B-8B parameters) and smaller models (2B-3B parameters) both benefit from this careful approach, preventing catastrophic forgetting while enabling effective adaptation.

**Rank (16)**: The rank of 16 provides an optimal balance between model expressiveness and efficiency. Lower ranks (4-8) may be too restrictive for complex cybersecurity knowledge, while higher ranks (32-64) increase training time and memory usage without significant performance gains.

**Target Modules**: Focusing on attention projection layers (q_proj, v_proj, k_proj, o_proj) allows the model to learn new attention patterns specific to cybersecurity concepts while preserving the core language understanding from pre-training.

**Consistent Naming Convention:**

- **Repository**: `{ModelName}-CyberQA-v1`
- **Adapter**: `{ModelName}-CyberQA-v1/1`
- **Dataset**: `{ModelName}-CyberQA-Dataset`

## Python API Usage

```python
from lora_finetuning_pipeline import LoRAFineTuningPipeline

# Initialize pipeline
pipeline = LoRAFineTuningPipeline()

# Show all model configurations
pipeline.print_reproducibility_info()

# Train specific models
results = pipeline.run_full_pipeline(['mistral_7b_base', 'llama3_8b_instruct'])

# Test single model inference
response = pipeline.test_model_inference(
    'mistral_7b_base', 
    "What is multi-factor authentication and why is it important?"
)
print(response)

# Compare all models with custom questions
comparison = pipeline.compare_models([
    "What is zero-trust security architecture?",
    "Explain ransomware protection strategies.",
    "What are the key components of a SIEM system?"
])

# Get detailed model information
model_info = pipeline.get_model_info()
for key, info in model_info.items():
    print(f"{key}: {info['name']} - LR: {info['hyperparameters']['learning_rate']}")

# Save results
pipeline.save_results('my_fine_tuning_results.json')
```

## Data Preparation

### Step 1: Source Data Setup

Place your source data in the `data/meta_data/data.csv` directory. The CSV should contain your raw training data with appropriate columns for questions and answers.

### Step 2: Model-Specific Data Processing

Use the provided Jupyter notebooks in the `Dataprep/` folder to process your data with model-specific prompt templates:

**Available Data Preparation Notebooks:**

- `Dataset_Prep_Google_Gemma.ipynb` - Google Gemma format
- `Dataset_Prep_Llama3.1.ipynb` - Meta Llama format
- `Dataset_Prep_Microsoft-Phi.ipynb` - Microsoft Phi format
- `Dataset_Prep_Mistral.ipynb` - Mistral format
- `Dataset_Prep_Zephyr.ipynb` - Zephyr format

**Output Structure:**

```
data/
├── meta_data/data.csv          # Your source data
├── Google-Gemma/data.csv       # Gemma-formatted data
├── Meta-Llama3.1/data.csv      # Llama-formatted data  
├── Microsoft-Phi/data.csv      # Phi-formatted data
├── Mistral/data.csv            # Mistral-formatted data
└── Zephyr/data.csv             # Zephyr-formatted data
```

### Step 3: Running Data Preparation

```bash
# For Mistral model
jupyter notebook Dataprep/Dataset_Prep_Mistral.ipynb

# For Llama model
jupyter notebook Dataprep/Dataset_Prep_Llama3.1.ipynb

# For other models, use their respective preparation notebooks
```

**Data Format Requirements:**

- Your source CSV should contain columns: `Question` and `Answer`
- The preparation scripts will format the data according to each model's requirements
- Data will be split into training (70-80%) and evaluation (20-30%) sets

#### Model-Specific Instruction Prompt Templates

Each model family uses a unique conversation format and special tokens:

**1. Mistral Models Template:**

```python
prompt_template = """<s>[INST] You are a safe, ethical, helpful, knowledgeable AI assistant and customer support expert specializing in cyber security, cloud computing and IT technical Support domains. Your primary job is to deliver detailed responses to customer questions in these domains. Drawing on your extensive expertise, adhere to the following guidelines:

1. Provide concise, accurate, and helpful answers to these questions, typically ranging from 450 - 500 words, depending on the complexity of the question.
2. Enhance readability by using appropriate formatting such as bullet points, short paragraphs, or numbered lists when applicable.
3. Prioritize customer satisfaction while maintaining an empathetic, human and professional tone throughout interactions.
4. Provide troubleshooting steps in a clear, organised and logical order when applicable.
5. Avoid providing information that could be harmful, biased, misused or leading to security risks or data loss.

Question: {Question}

Answer: [/INST]
"""
```

**2. Llama 3.1 Models Template:**

```python
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a safe, ethical, helpful, knowledgeable AI assistant and customer support expert specialising in cyber security, cloud computing and IT technical Support domains. Your primary job is to deliver detailed responses to customer questions in these domains. Drawing on your extensive expertise, adhere to the following guidelines:

1. Provide concise, accurate, and helpful answers to these questions, typically ranging from 450 - 500 words, depending on the complexity of the question.
2. Enhance readability by using appropriate formatting such as bullet points, short paragraphs, or numbered lists when applicable.
3. Prioritize customer satisfaction while maintaining an empathetic, human and professional tone throughout interactions.
4. Provide troubleshooting steps in a clear, organised and logical order when applicable.
5. Avoid providing information that could be harmful, biased, misused or leading to security risks or data loss.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {Question}

Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
```

**3. Google Gemma Models Template:**

```python
prompt_template = """<bos><start_of_turn>user
You are a safe, ethical, helpful, knowledgeable AI assistant and customer support expert specializing in cyber security, cloud computing and IT technical Support domains. Your primary job is to deliver detailed responses to customer questions in these domains. Drawing on your extensive expertise, adhere to the following guidelines:

1. Provide concise, accurate, and helpful answers to these questions, typically ranging from 450 - 500 words, depending on the complexity of the question.
2. Enhance readability by using appropriate formatting such as bullet points, short paragraphs, or numbered lists when applicable.
3. Prioritize customer satisfaction while maintaining an empathetic, human and professional tone throughout interactions.
4. Provide troubleshooting steps in a clear, organised and logical order when applicable.
5. Avoid providing information that could be harmful, biased, misused or leading to security risks or data loss.

Question: {Question}

Answer: <end_of_turn>
<start_of_turn>model

"""
```

**4. Microsoft Phi Models Template:**

```python
prompt_template = """<s><|user|>
You are a safe, ethical, helpful, knowledgeable AI assistant and customer support expert specializing in cyber security, cloud computing and IT technical Support domains. Your primary job is to deliver detailed responses to customer questions in these domains. Drawing on your extensive expertise, adhere to the following guidelines:

1. Provide concise, accurate, and helpful answers to these questions, typically ranging from 450 - 500 words, depending on the complexity of the question.
2. Enhance readability by using appropriate formatting such as bullet points, short paragraphs, or numbered lists when applicable.
3. Prioritize customer satisfaction while maintaining an empathetic, human and professional tone throughout interactions.
4. Provide troubleshooting steps in a clear, organised and logical order when applicable.
5. Avoid providing information that could be harmful, biased, misused or leading to security risks or data loss.

Question: {Question}

Answer: <|end|>
<|assistant|>

"""
```

**5. Zephyr Models Template:**

```python
prompt_template = """<|system|>
You are a safe, ethical, helpful, knowledgeable AI assistant and customer support expert specialising in cyber security, cloud computing and IT technical Support domains. Your primary job is to deliver detailed responses to customer questions in these domains. Drawing on your extensive expertise, adhere to the following guidelines:

1. Provide concise, accurate, and helpful answers to these questions, typically ranging from 450 - 500 words, depending on the complexity of the question.
2. Enhance readability by using appropriate formatting such as bullet points, short paragraphs, or numbered lists when applicable.
3. Prioritize customer satisfaction while maintaining an empathetic, human and professional tone throughout interactions.
4. Provide troubleshooting steps in a clear, organised and logical order when applicable.
5. Avoid providing information that could be harmful, biased, misused or leading to security risks or data loss.

</s>
<|user|>
Question: {Question}

Answer: </s>
<|assistant|>

"""
```

#### Template Format Guidelines

- **Special Tokens**: Each model uses specific conversation delimiters and special tokens that must be preserved exactly
- **System Instructions**: The cybersecurity domain expertise and guidelines are consistent across all templates
- **Variable Substitution**: The `{Question}` placeholder gets replaced with actual questions from your dataset
- **Output Format**: Each template prepares the model to generate appropriate responses following the instruction format

### Step 2: Fine-Tuning Process

Open the main fine-tuning notebook:

```bash
jupyter notebook src/Lora_finetuning_script.ipynb
```

Follow these sections in order:

#### 2.1 Setup and Authentication

```python
# Load environment variables and authenticate
from predibase import Predibase
import os
from dotenv import load_dotenv

load_dotenv()
pb = Predibase()
```

#### 2.2 Upload Dataset

```python
# Upload your prepared dataset
dataset = pb.datasets.from_file('../data/Mistral/data.csv', name='Your-Dataset-Name')
```

#### 2.3 Create Repository and Adapter

```python
# Create a repository for your fine-tuned model
repo = pb.repos.create(
    name="Your-Model-Name", 
    description="Description of your model", 
    exists_ok=True
)

# Start fine-tuning
adapter = pb.adapters.create(
    config={
        "base_model": "mistral-7b-instruct-v0-2",
        "epochs": 5,
        "rank": 16,
        "task": "instruction_tuning",
        "learning_rate": 0.0001,
        "target_modules": ["q_proj", "v_proj", "k_proj"]
    },
    dataset=dataset,
    repo=repo,
    description="Your experiment description"
)
```

## Supported Models

### 1. Mistral Models

- `mistral-7b`
- `mistral-7b-instruct-v0-2`
- `mistral-7b-instruct-v0-3`

### 2. Meta Llama Models

- `llama-3-8b-instruct`

### 3. Other Models

- Google Gemma variants
- Microsoft Phi models
- Zephyr models

## Hyperparameter Configuration

### Key Parameters Explained

| Parameter          | Description                   | Recommended Values                 |
| ------------------ | ----------------------------- | ---------------------------------- |
| `epochs`         | Number of training iterations | 3-10                               |
| `rank`           | LoRA rank (model complexity)  | 8-32                               |
| `learning_rate`  | Training step size            | 0.00001-0.0005                     |
| `target_modules` | Modules to fine-tune          | `["q_proj", "v_proj", "k_proj"]` |

### Example Configurations

**Conservative (Prevent Overfitting):**

```python
config = {
    "epochs": 3,
    "rank": 8,
    "learning_rate": 0.00005,
    "target_modules": ["q_proj", "v_proj"]
}
```

**Balanced:**

```python
config = {
    "epochs": 5,
    "rank": 16,
    "learning_rate": 0.0001,
    "target_modules": ["q_proj", "v_proj", "k_proj"]
}
```

**Aggressive (More Expressive):**

```python
config = {
    "epochs": 10,
    "rank": 32,
    "learning_rate": 0.0002,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

## Model Inference

After fine-tuning, test your model:

```python
# Connect to your fine-tuned model
lorax_client = pb.deployments.client("base-model-name")

# Generate responses
response = lorax_client.generate(
    input_prompt, 
    adapter_id="Your-Model-Name/version", 
    max_new_tokens=512
)
print(response.generated_text)
```

```python
# Check adapter status
adapter = pb.adapters.get("Your-Model-Name/version")
print(adapter.status)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**

   - Verify your API token in the `.env` file
   - Ensure the token has proper permissions
2. **Dataset Upload Failures**

   - Check CSV format and column names
   - Ensure file size is within Predibase limits

## 🔧 Configuration Details

### Model Configuration Structure

Each model is configured with standardized parameters for reproducibility:

```python
@dataclass
class ModelConfig:
    name: str                    # Display name
    base_model: str             # Predibase model identifier
    data_path: str              # Path to training data
    repository_name: str        # Consistent naming pattern
    adapter_name: str           # Consistent adapter naming
    dataset_name: str           # Consistent dataset naming
    hyperparameters: dict       # Standardized hyperparameters
    prompt_template: str        # Model-specific prompt format
```

### Prompt Templates

**Full prompt templates are automatically applied during fine-tuning:**

- **Mistral Models**: `<s>[INST] {input} [/INST] {output}</s>`
- **Llama Models**: `<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{output}<|eot_id|>`
- **Gemma Models**: `<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>`
- **Phi Models**: `<|user|>\n{input}<|end|>\n<|assistant|>\n{output}<|end|>`
- **Zephyr Models**: `<|user|>\n{input}</s>\n<|assistant|>\n{output}</s>`

### Configuration File Usage

The pipeline supports configuration via `src/config.yaml`:

```python
# Load custom configuration (optional)
pipeline = LoRAFineTuningPipeline(config_path="src/config.yaml")

# Or use default hardcoded configurations
pipeline = LoRAFineTuningPipeline()
```

Configuration file allows you to:

- Modify dataset paths for each model
- Adjust pipeline settings (retries, batch size)
- Customize logging options
- Define test questions for evaluation
- Set output directories

## 🛠️ Advanced Usage

### Custom Model Configuration

```python
# Add custom model configuration
custom_config = ModelConfig(
    name="Custom-Model-CyberQA",
    base_model="custom/model-name",
    data_path="data/custom/data.csv",
    repository_name="Custom-Model-CyberQA-v1",
    adapter_name="Custom-Model-CyberQA-v1/1", 
    dataset_name="Custom-Model-CyberQA-Dataset",
    hyperparameters={
        "learning_rate": 0.00005,
        "epochs": 5,
        "rank": 16,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "max_new_tokens": 512,
        "temperature": 0.1
    },
    prompt_template="<custom>{input}</custom>{output}"
)

# Add to pipeline
pipeline.model_configs['custom_model'] = custom_config
```

### Batch Processing

```python
# Process multiple model groups
mistral_models = ['mistral_7b_base', 'mistral_7b_instruct']
llama_models = ['llama3_8b_instruct', 'llama31_8b_instruct']

# Train by model family
mistral_results = pipeline.run_full_pipeline(mistral_models)
llama_results = pipeline.run_full_pipeline(llama_models)
```

### Error Handling and Logging

The pipeline includes comprehensive error handling and logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Pipeline automatically logs:
# - Model training progress
# - Inference testing results  
# - Error states and recovery
# - Performance metrics
```

## Troubleshooting

### Common Issues

**1. Authentication Errors**

```bash
# Verify your tokens
python -c "import os; print('PREDIBASE_API_TOKEN' in os.environ)"
```

**2. Data Format Issues**

- Ensure CSV files have correct columns
- Check prompt template formatting
- Verify file paths in model configurations

**3. Memory Issues**

- Reduce batch size in hyperparameters
- Use smaller models for testing
- Monitor system resources

**4. Model Training Failures**

- Check Predibase service status
- Verify model name spelling
- Ensure sufficient account credits

### Getting Help

- **Predibase Documentation**: [docs.predibase.com](https://docs.predibase.com)
- **API Reference**: Check the Predibase SDK documentation
- **Community Support**: Predibase Discord/Forums

## Best Practices

### Data Quality

- **High-Quality Q&A Pairs**: Ensure your training data contains accurate, comprehensive question-answer pairs
- **Domain Relevance**: Focus on cybersecurity, cloud computing, and IT support domains for optimal performance
- **Consistent Formatting**: Maintain uniform data structure across all model datasets
- **Diverse Examples**: Include varied question types and complexity levels

### Model Selection

- **Base Model Choice**: Select appropriate base models for your specific use case
- **Resource Considerations**: Balance model size with available computational resources
- **Performance vs. Speed**: Consider inference speed requirements for production deployment
- **Multi-Model Testing**: Test multiple configurations to find optimal performance

### Hyperparameter Optimization

- **Conservative Start**: Begin with proven hyperparameter settings
- **Systematic Tuning**: Adjust one parameter at a time for controlled experimentation
- **Evaluation Monitoring**: Track validation metrics to prevent overfitting
- **Early Stopping**: Implement early stopping based on validation performance

### Production Deployment

- **Reproducibility**: Use standardized configurations for consistent results
- **Monitoring**: Implement logging and monitoring for production systems
- **Version Control**: Track model versions and configurations
- **Testing**: Thoroughly test models before production deployment

## Additional Resources

### Documentation

- [Predibase API Documentation](https://docs.predibase.com)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Support

For questions, issues, or suggestions:

- Open an issue on GitHub
- Contact the development team
- Check the troubleshooting section above

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add appropriate tests
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
