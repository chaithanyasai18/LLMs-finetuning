"""
LoRA Fine-tuning Pipeline for Cybersecurity and IT Support Domain
======================================================

A comprehensive, reproducible fine-tuning pipeline for multiple Large Language Models (LLMs) 
using LoRA (Low-Rank Adaptation) technique with Predibase SDK for cybersecurity and IT support domain.

This pipeline ensures consistency and reproducibility across all model architectures with
standardized configurations, naming conventions, and full prompt templates.

Supported Models:
- Mistral-7B (Base & Instruct v0.3)
- Meta Llama-3-8B-Instruct
- Meta Llama-3.1-8B-Instruct  
- Google Gemma-2B
- Microsoft Phi-3-Mini-4K-Instruct
- Zephyr-7B-Beta
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from predibase import Predibase, FinetuningConfig, DeploymentConfig
from langchain_community.llms import Predibase as LangChainPredibase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_finetuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model fine-tuning parameters."""
    name: str
    base_model: str
    client_name: str
    dataset_path: str
    repo_name: str
    adapter_name: str
    prompt_template: str
    epochs: int = 5
    rank: int = 16
    learning_rate: float = 0.00005
    target_modules: List[str] = None
    max_new_tokens: int = 300
    temperature: float = 0.1

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRAFineTuningPipeline:
    """
    Main pipeline class for LoRA fine-tuning of multiple LLM architectures.
    
    This class provides a comprehensive framework for:
    - Dataset management and upload
    - Model fine-tuning with LoRA
    - Inference testing and evaluation
    - Model comparison and benchmarking
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LoRA Fine-tuning Pipeline.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.setup_environment()
        self.pb = Predibase()
        self.models_config = self._initialize_model_configs()
        self.results = {}
        
        logger.info("LoRA Fine-tuning Pipeline initialized successfully")
    
    def setup_environment(self) -> None:
        """Load environment variables and validate API tokens."""
        load_dotenv()
        
        required_tokens = ["PREDIBASE_API_TOKEN"]
        optional_tokens = ["WANDB_API_KEY", "HUGGINGFACE_TOKEN"]
        
        for token in required_tokens:
            if not os.getenv(token):
                raise ValueError(f"Required environment variable {token} not found")
        
        for token in optional_tokens:
            if not os.getenv(token):
                logger.warning(f"Optional environment variable {token} not found")
        
        logger.info("Environment setup completed")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Initialize standardized model configurations for all supported architectures.
        
        This method ensures reproducibility by using consistent:
        - Naming conventions
        - Learning rates and hyperparameters
        - Full prompt templates specific to each model
        - Target modules for LoRA adaptation
        
        Configuration Rationale:
        - Learning Rate (0.00005): Conservative rate ensuring stability across all model sizes
        - Rank (16): Optimal balance between expressiveness and efficiency
        - Target Modules: Attention layers for effective domain adaptation
        - Epochs (5): Sufficient for convergence without overfitting
        
        Returns:
            Dictionary mapping model keys to standardized ModelConfig objects
        """
        
        base_path = Path("../data")
        
        # Standard hyperparameters for reproducibility across all models
        STANDARD_EPOCHS = 5
        STANDARD_RANK = 16
        STANDARD_LEARNING_RATE = 0.00005  # Conservative rate for stability across all model sizes
        STANDARD_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention layer adaptation
        STANDARD_MAX_TOKENS = 512  # Sufficient for detailed cybersecurity responses
        STANDARD_TEMPERATURE = 0.1  # Low temperature for consistent, focused outputs
        
        # Standard prompt template components
        SYSTEM_PROMPT = """You are a safe and helpful AI assistant specializing in cybersecurity, cloud computing, and IT support. Your role is to provide accurate, helpful responses to customer questions. Draw on your extensive knowledge to offer clear, concise answers tailored to each question and its context. 

Key guidelines:
1. Focus on cybersecurity, cloud computing, and IT service desk support topics
2. Prioritize customer satisfaction while maintaining accuracy
3. If unsure about an answer, reply: "I'm sorry, but I don't know the answer. I'm continuously learning and improving every day. Please try asking me another question."
4. Stay within your areas of expertise
5. Use bullet points and numbered lists when helpful
6. Maintain a professional and helpful demeanor"""
        
        configs = {
            "mistral_7b_base": ModelConfig(
                name="Mistral-7B-Base-CyberQA",
                base_model="mistral-7b",
                client_name="mistral-7b",
                dataset_path=str(base_path / "Mistral" / "data.csv"),
                repo_name="Mistral-7B-Base-CyberQA-v1",
                adapter_name="Mistral-7B-Base-CyberQA-v1/1",
                prompt_template=f"""<s>[INST] {SYSTEM_PROMPT}

Question: {{question}}

Answer: [/INST]""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "mistral_7b_instruct": ModelConfig(
                name="Mistral-7B-Instruct-CyberQA",
                base_model="mistral-7b-instruct-v0-3",
                client_name="mistral-7b-instruct-v0-3",
                dataset_path=str(base_path / "Mistral" / "data.csv"),
                repo_name="Mistral-7B-Instruct-CyberQA-v1",
                adapter_name="Mistral-7B-Instruct-CyberQA-v1/1",
                prompt_template=f"""<s>[INST] {SYSTEM_PROMPT}

Question: {{question}}

Answer: [/INST]""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "llama3_8b_instruct": ModelConfig(
                name="Llama-3-8B-Instruct-CyberQA",
                base_model="llama-3-8b-instruct",
                client_name="llama-3-8b-instruct",
                dataset_path=str(base_path / "meta_data" / "data.csv"),
                repo_name="Llama-3-8B-Instruct-CyberQA-v1",
                adapter_name="Llama-3-8B-Instruct-CyberQA-v1/1",
                prompt_template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {{question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "llama31_8b_instruct": ModelConfig(
                name="Llama-3.1-8B-Instruct-CyberQA",
                base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                client_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                dataset_path=str(base_path / "Meta-Llama3.1" / "data.csv"),
                repo_name="Llama-3.1-8B-Instruct-CyberQA-v1",
                adapter_name="Llama-3.1-8B-Instruct-CyberQA-v1/1",
                prompt_template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {{question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "google_gemma_2b": ModelConfig(
                name="Google-Gemma-2B-CyberQA",
                base_model="google/gemma-2b",
                client_name="google/gemma-2b",
                dataset_path=str(base_path / "Google-Gemma" / "data.csv"),
                repo_name="Google-Gemma-2B-CyberQA-v1",
                adapter_name="Google-Gemma-2B-CyberQA-v1/1",
                prompt_template=f"""<bos><start_of_turn>user
{SYSTEM_PROMPT}

Question: {{question}}<end_of_turn>
<start_of_turn>model
""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "microsoft_phi3_mini": ModelConfig(
                name="Microsoft-Phi-3-Mini-CyberQA",
                base_model="microsoft/Phi-3-mini-4k-instruct",
                client_name="microsoft/Phi-3-mini-4k-instruct",
                dataset_path=str(base_path / "Microsoft-Phi" / "data.csv"),
                repo_name="Microsoft-Phi-3-Mini-CyberQA-v1",
                adapter_name="Microsoft-Phi-3-Mini-CyberQA-v1/1",
                prompt_template=f"""<|system|>
{SYSTEM_PROMPT}<|end|>
<|user|>
Question: {{question}}<|end|>
<|assistant|>
""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            ),
            
            "zephyr_7b_beta": ModelConfig(
                name="Zephyr-7B-Beta-CyberQA",
                base_model="HuggingFaceH4/zephyr-7b-beta",
                client_name="HuggingFaceH4/zephyr-7b-beta",
                dataset_path=str(base_path / "Zephyr" / "data.csv"),
                repo_name="Zephyr-7B-Beta-CyberQA-v1",
                adapter_name="Zephyr-7B-Beta-CyberQA-v1/1",
                prompt_template=f"""<|system|>
{SYSTEM_PROMPT}</s>
<|user|>
Question: {{question}}</s>
<|assistant|>
""",
                epochs=STANDARD_EPOCHS,
                rank=STANDARD_RANK,
                learning_rate=STANDARD_LEARNING_RATE,
                target_modules=STANDARD_TARGET_MODULES,
                max_new_tokens=STANDARD_MAX_TOKENS,
                temperature=STANDARD_TEMPERATURE
            )
        }
        
        return configs
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all configured models for reproducibility.
        
        Returns:
            Dictionary with model information including configurations and metadata
        """
        model_info = {}
        
        for key, config in self.models_config.items():
            model_info[key] = {
                "name": config.name,
                "base_model": config.base_model,
                "dataset_path": config.dataset_path,
                "repo_name": config.repo_name,
                "adapter_name": config.adapter_name,
                "hyperparameters": {
                    "epochs": config.epochs,
                    "rank": config.rank,
                    "learning_rate": config.learning_rate,
                    "max_new_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "target_modules": config.target_modules
                },
                "prompt_template_preview": config.prompt_template[:200] + "..." if len(config.prompt_template) > 200 else config.prompt_template
            }
        
        return model_info
    
    def print_reproducibility_info(self) -> None:
        """Print standardized configuration information for reproducibility."""
        print("=" * 80)
        print("LoRA Fine-tuning Pipeline - Reproducibility Information")
        print("=" * 80)
        print(f"Pipeline Version: 1.0.0")
        print(f"Total Models Supported: {len(self.models_config)}")
        print(f"Standard Configuration:")
        print(f"  - Epochs: 5")
        print(f"  - Rank: 16") 
        print(f"  - Target Modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj']")
        print(f"  - Max New Tokens: 512")
        print(f"  - Temperature: 0.1")
        print("\nModel-Specific Learning Rates:")
        
        for key, config in self.models_config.items():
            print(f"  - {config.name}: {config.learning_rate}")
        
        print("\nAvailable Models:")
        for key, config in self.models_config.items():
            dataset_exists = os.path.exists(config.dataset_path)
            status = "✅" if dataset_exists else "❌"
            print(f"  {status} {key}: {config.name}")
        
        print("=" * 80)
    
    def upload_dataset(self, model_key: str, dataset_name: str = None) -> Any:
        """
        Upload dataset for a specific model.
        
        Args:
            model_key: Key identifying the model configuration
            dataset_name: Optional custom dataset name
            
        Returns:
            Predibase dataset object
        """
        config = self.models_config[model_key]
        
        if not os.path.exists(config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
        
        dataset_name = dataset_name or f"{config.name}-CyberQA-Dataset"
        
        try:
            dataset = self.pb.datasets.from_file(
                config.dataset_path, 
                name=dataset_name
            )
            logger.info(f"Successfully uploaded dataset for {config.name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to upload dataset for {config.name}: {str(e)}")
            raise
    
    def create_repository(self, model_key: str) -> Any:
        """
        Create repository for model adapter storage.
        
        Args:
            model_key: Key identifying the model configuration
            
        Returns:
            Predibase repository object
        """
        config = self.models_config[model_key]
        
        try:
            repo = self.pb.repos.create(
                name=config.repo_name,
                description=f"{config.name} - Cybersecurity Question and Answer Fine Tuning",
                exists_ok=True
            )
            logger.info(f"Repository created/accessed for {config.name}")
            return repo
            
        except Exception as e:
            logger.error(f"Failed to create repository for {config.name}: {str(e)}")
            raise
    
    def fine_tune_model(self, model_key: str, dataset: Any, repo: Any) -> Any:
        """
        Fine-tune a specific model using LoRA.
        
        Args:
            model_key: Key identifying the model configuration
            dataset: Predibase dataset object
            repo: Predibase repository object
            
        Returns:
            Predibase adapter object
        """
        config = self.models_config[model_key]
        
        fine_tuning_config = {
            "base_model": config.base_model,
            "epochs": config.epochs,
            "rank": config.rank,
            "task": "instruction_tuning",
            "learning_rate": config.learning_rate,
            "target_modules": config.target_modules,
        }
        
        try:
            logger.info(f"Starting fine-tuning for {config.name}")
            adapter = self.pb.adapters.create(
                config=fine_tuning_config,
                dataset=dataset,
                repo=repo,
                description=f"{config.name} fine-tuning for cybersecurity domain"
            )
            
            logger.info(f"Fine-tuning completed for {config.name}")
            return adapter
            
        except Exception as e:
            logger.error(f"Fine-tuning failed for {config.name}: {str(e)}")
            raise
    
    def test_model_inference(self, model_key: str, test_question: str = None) -> str:
        """
        Test inference for a fine-tuned model with standardized configuration.
        
        Args:
            model_key: Key identifying the model configuration
            test_question: Question to test the model with
            
        Returns:
            Model response text
        """
        config = self.models_config[model_key]
        test_question = test_question or "What is multi-factor authentication and why is it important in cybersecurity?"
        
        try:
            client = self.pb.deployments.client(config.client_name)
            prompt = config.prompt_template.format(question=test_question)
            
            logger.info(f"Testing inference for {config.name} with question: {test_question[:50]}...")
            
            response = client.generate(
                prompt,
                adapter_id=config.adapter_name,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature
            )
            
            logger.info(f"Inference test completed for {config.name}")
            return response.generated_text
            
        except Exception as e:
            logger.error(f"Inference test failed for {config.name}: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_full_pipeline(self, model_keys: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline for specified models.
        
        Args:
            model_keys: List of model keys to process. If None, processes all models.
            
        Returns:
            Dictionary containing results for each model
        """
        if model_keys is None:
            model_keys = list(self.models_config.keys())
        
        results = {}
        
        for model_key in model_keys:
            logger.info(f"Starting pipeline for {self.models_config[model_key].name}")
            
            try:
                # Step 1: Upload dataset
                dataset = self.upload_dataset(model_key)
                
                # Step 2: Create repository
                repo = self.create_repository(model_key)
                
                # Step 3: Fine-tune model
                adapter = self.fine_tune_model(model_key, dataset, repo)
                
                # Step 4: Test inference
                response = self.test_model_inference(model_key)
                
                results[model_key] = {
                    "status": "success",
                    "adapter_id": adapter.id if hasattr(adapter, 'id') else 'unknown',
                    "test_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Pipeline failed for {model_key}: {str(e)}")
                results[model_key] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        self.results = results
        return results
    
    def compare_models(self, test_questions: List[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Compare responses from all fine-tuned models using standardized questions.
        
        Args:
            test_questions: List of questions to test all models with
            
        Returns:
            Dictionary containing model comparisons with consistent format
        """
        if test_questions is None:
            # Standard test questions for cybersecurity domain evaluation
            test_questions = [
                "What is multi-factor authentication and why is it important in cybersecurity?",
                "Explain the difference between symmetric and asymmetric encryption in simple terms.",
                "What are the key components of a Security Operations Center (SOC) and their functions?",
                "How does zero-trust security architecture work and why is it becoming popular?",
                "What is the difference between vulnerability assessment and penetration testing?",
                "Explain what a firewall does and the different types available."
            ]
        
        comparison_results = {}
        
        logger.info(f"Starting model comparison with {len(test_questions)} questions")
        
        for question in test_questions:
            comparison_results[question] = {}
            logger.info(f"Testing question: {question[:50]}...")
            
            for model_key, config in self.models_config.items():
                try:
                    response = self.test_model_inference(model_key, question)
                    comparison_results[question][config.name] = response
                    
                except Exception as e:
                    logger.error(f"Error testing {config.name}: {str(e)}")
                    comparison_results[question][config.name] = f"Error: {str(e)}"
        
        logger.info("Model comparison completed")
        return comparison_results
    
    def print_configuration_rationale(self) -> None:
        """
        Print detailed explanation of configuration choices for all models.
        """
        print("\n" + "="*80)
        print("STANDARDIZED CONFIGURATION RATIONALE")
        print("="*80)
        
        print("\nHyperparameter Configuration:")
        print(f"  Learning Rate: 0.00005")
        print(f"    - Conservative rate ensuring stable training across all model sizes")
        print(f"    - Prevents catastrophic forgetting while enabling effective adaptation")
        print(f"    - Works well for both large models (7B-8B) and smaller models (2B-3B)")
        
        print(f"\n  Rank: 16")
        print(f"    - Optimal balance between model expressiveness and efficiency")
        print(f"    - Lower ranks (4-8) may be too restrictive for complex cybersecurity knowledge")
        print(f"    - Higher ranks (32-64) increase training time without significant gains")
        
        print(f"\n  Target Modules: {['q_proj', 'v_proj', 'k_proj', 'o_proj']}")
        print(f"    - Focus on attention projection layers for effective domain adaptation")
        print(f"    - Allows learning new attention patterns for cybersecurity concepts")
        print(f"    - Preserves core language understanding from pre-training")
        
        print(f"\n  Epochs: 5")
        print(f"    - Sufficient for convergence without overfitting")
        print(f"    - Balances training time with performance gains")
        
        print(f"\n  Temperature: 0.1")
        print(f"    - Low temperature for consistent, focused outputs")
        print(f"    - Reduces randomness in cybersecurity responses")
        
        print(f"\n  Max New Tokens: 512")
        print(f"    - Sufficient for detailed cybersecurity explanations")
        print(f"    - Balances comprehensiveness with efficiency")
        
        print("\n" + "="*80)
        
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about all configured models.
        
        Returns:
            Dictionary containing model configurations and hyperparameters
        """
        model_info = {}
        
        for key, config in self.models_config.items():
            model_info[key] = {
                "name": config.name,
                "base_model": config.base_model,
                "hyperparameters": {
                    "learning_rate": 0.00005,  # All models use standardized rate
                    "epochs": config.epochs,
                    "rank": config.rank,
                    "target_modules": config.target_modules,
                    "max_new_tokens": config.max_new_tokens,
                    "temperature": config.temperature
                },
                "naming": {
                    "repository": config.repo_name,
                    "adapter": config.adapter_name,
                    "dataset": f"{config.name}-Dataset"
                }
            }
        
        return model_info
    
    def save_results(self, filepath: str = None) -> None:
        """
        Save pipeline results to JSON file.
        
        Args:
            filepath: Optional custom filepath for results
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"lora_finetuning_results_{timestamp}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the fine-tuning pipeline results.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results available. Run the pipeline first."
        
        report = []
        report.append("=" * 80)
        report.append("LoRA Fine-tuning Pipeline Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        successful_models = []
        failed_models = []
        
        for model_key, result in self.results.items():
            model_name = self.models_config[model_key].name
            
            if result["status"] == "success":
                successful_models.append(model_name)
            else:
                failed_models.append((model_name, result.get("error", "Unknown error")))
        
        report.append(f"Summary:")
        report.append(f"- Total models processed: {len(self.results)}")
        report.append(f"- Successful fine-tuning: {len(successful_models)}")
        report.append(f"- Failed fine-tuning: {len(failed_models)}")
        report.append("")
        
        if successful_models:
            report.append("✅ Successfully fine-tuned models:")
            for model in successful_models:
                report.append(f"  - {model}")
            report.append("")
        
        if failed_models:
            report.append("❌ Failed models:")
            for model, error in failed_models:
                report.append(f"  - {model}: {error}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """
    Main execution function demonstrating pipeline usage for community reproducibility.
    
    This example shows how to use the pipeline for reproducible fine-tuning
    across all supported model architectures.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning Pipeline for Cybersecurity LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show all model configurations
    python lora_finetuning_pipeline.py --info
    
    # Show configuration rationale and hyperparameter choices
    python lora_finetuning_pipeline.py --config-rationale
    
    # Train specific models
    python lora_finetuning_pipeline.py --models mistral_7b_base llama3_8b_instruct
    
    # Test inference only
    python lora_finetuning_pipeline.py --test-only --models all
    
    # Compare all models
    python lora_finetuning_pipeline.py --compare
    
    # Full pipeline with output
    python lora_finetuning_pipeline.py --models all --output results.json
        """
    )
    
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to process (use 'all' for all models)")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only run inference tests (no training)")
    parser.add_argument("--compare", action="store_true", 
                       help="Run model comparison")
    parser.add_argument("--info", action="store_true", 
                       help="Show model configurations and reproducibility info")
    parser.add_argument("--config-rationale", action="store_true", 
                       help="Show detailed explanation of configuration choices")
    parser.add_argument("--output", type=str, 
                       help="Output file for results")
    parser.add_argument("--question", type=str, 
                       help="Custom test question")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        logger.info("Initializing LoRA Fine-tuning Pipeline...")
        pipeline = LoRAFineTuningPipeline()
        
        # Show info if requested
        if args.info:
            pipeline.print_reproducibility_info()
            return
            
        # Show configuration rationale if requested
        if args.config_rationale:
            pipeline.print_configuration_rationale()
            return
        
        # Determine models to process
        if args.models:
            if "all" in args.models:
                model_keys = list(pipeline.models_config.keys())
            else:
                model_keys = [key for key in args.models if key in pipeline.models_config]
                if not model_keys:
                    print("❌ No valid models specified. Available models:")
                    for key in pipeline.models_config.keys():
                        print(f"   - {key}")
                    return
        else:
            model_keys = list(pipeline.models_config.keys())
        
        if args.test_only:
            # Run inference tests only
            logger.info("Running inference tests...")
            test_question = args.question or "What is multi-factor authentication and why is it important in cybersecurity?"
            
            print(f"\n🧪 Testing Question: {test_question}")
            print("=" * 80)
            
            for model_key in model_keys:
                config = pipeline.models_config[model_key]
                print(f"\n🤖 Testing {config.name}...")
                
                try:
                    response = pipeline.test_model_inference(model_key, test_question)
                    print(f"✅ Response:")
                    print("-" * 40)
                    print(response)
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    
        elif args.compare:
            # Run model comparison
            logger.info("Running comprehensive model comparison...")
            comparison_results = pipeline.compare_models()
            
            print("\n🔄 Model Comparison Results")
            print("=" * 80)
            
            for question, responses in comparison_results.items():
                print(f"\n❓ Question: {question}")
                print("-" * 60)
                for model, response in responses.items():
                    print(f"\n🤖 {model}:")
                    if response.startswith("Error:"):
                        print(f"❌ {response}")
                    else:
                        # Show first 200 chars for readability
                        display_response = response[:200] + "..." if len(response) > 200 else response
                        print(display_response)
                print("=" * 80)
            
            # Save comparison results if requested
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(comparison_results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Comparison results saved to: {args.output}")
                
        else:
            # Run full pipeline
            logger.info("Starting full LoRA fine-tuning pipeline...")
            print("\n🚀 Starting LoRA Fine-tuning Pipeline")
            print("=" * 60)
            print(f"📋 Processing models: {', '.join(model_keys)}")
            
            results = pipeline.run_full_pipeline(model_keys=model_keys)
            
            # Generate and display report
            report = pipeline.generate_report()
            print("\n" + report)
            
            # Save results if output file specified
            if args.output:
                pipeline.save_results(args.output)
                print(f"\n💾 Results saved to: {args.output}")
                
        print("\n✅ Pipeline execution completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\n❌ Pipeline failed: {e}")
        print("\nFor troubleshooting:")
        print("1. Ensure all environment variables are set (PREDIBASE_API_TOKEN)")
        print("2. Check that dataset files exist in the data/ directory")
        print("3. Verify Predibase API connectivity")
        return


if __name__ == "__main__":
    main()