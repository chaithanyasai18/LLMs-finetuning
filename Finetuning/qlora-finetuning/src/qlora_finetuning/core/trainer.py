"""
Main QLora fine-tuning trainer class.

This module contains the core QLorAFineTuner class that orchestrates the training process.
"""

import os
import json
from typing import Optional, Tuple
from pathlib import Path

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from ..core.config import ModelConfig, TrainingConfig
from ..models.registry import ModelRegistry
from ..utils import SystemInfo, get_logger, TrainingLogger, AuthenticationManager


class QLorAFineTuner:
    """Main class for QLora fine-tuning operations."""
    
    def __init__(self, model_name: str, training_config: TrainingConfig):
        """
        Initialize the QLorA Fine-tuner.
        
        Args:
            model_name: Name of the model to fine-tune
            training_config: Training configuration parameters
        """
        self.logger = get_logger("trainer")
        self.training_logger = TrainingLogger()
        
        # Get model configuration
        if not ModelRegistry.is_supported(model_name):
            available_models = ModelRegistry.list_models()
            raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")
        
        self.model_name = model_name
        self.model_config = ModelRegistry.get_model_config(model_name)
        self.training_config = training_config
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self.peft_config = None
        
        # Set up output directory
        self.output_dir = Path(training_config.output_dir) / self.model_config.new_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized QLorA Fine-tuner for {model_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def validate_system_requirements(self) -> bool:
        """
        Validate system requirements for training.
        
        Returns:
            True if system meets requirements, False otherwise
        """
        self.logger.info("Validating system requirements...")
        
        validation = SystemInfo.validate_system_requirements(self.model_name)
        
        # Log system information
        self.logger.info(f"GPU Available: {validation['gpu_available']}")
        if validation['gpu_available']:
            self.logger.info(f"Total VRAM: {validation['total_vram_gb']:.1f} GB")
            self.logger.info(f"GPU Count: {validation['gpu_count']}")
        
        self.logger.info(f"System RAM: {validation['ram_gb']:.1f} GB")
        
        # Log warnings
        for warning in validation['warnings']:
            self.logger.warning(warning)
        
        # Log recommendations
        for recommendation in validation['recommendations']:
            self.logger.info(f"Recommendation: {recommendation}")
        
        return validation['meets_minimum'] or validation['gpu_available']
    
    def setup_authentication(
        self, 
        hf_token: Optional[str] = None, 
        wandb_token: Optional[str] = None
    ) -> dict:
        """
        Set up authentication for external services.
        
        Args:
            hf_token: Hugging Face token
            wandb_token: Weights & Biases token
        
        Returns:
            Authentication status for each service
        """
        self.logger.info("Setting up authentication...")
        return AuthenticationManager.setup_all_auth(hf_token, wandb_token)
    
    def get_torch_dtype_and_attention(self) -> Tuple[torch.dtype, str]:
        """
        Determine the optimal torch dtype and attention implementation.
        
        Returns:
            Tuple of torch dtype and attention implementation string
        """
        return SystemInfo.get_optimal_torch_dtype()
    
    def load_model_and_tokenizer(self) -> None:
        """Load and configure the model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_config.base_model}")
        
        torch_dtype, attn_implementation = self.get_torch_dtype_and_attention()
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=True
        )
        
        # Setup chat format if needed (for instruct models)
        if "instruct" in self.model_config.base_model.lower() or "it" in self.model_config.base_model.lower():
            self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def configure_peft(self) -> None:
        """Configure PEFT (LoRA) for the model."""
        self.logger.info("Configuring LoRA...")
        
        self.peft_config = LoraConfig(
            r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.model_config.target_modules,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def load_and_prepare_dataset(self) -> None:
        """Load and prepare the dataset for training."""
        self.logger.info(f"Loading dataset: {self.model_config.dataset_name}")
        
        # Load dataset
        self.dataset = load_dataset(self.model_config.dataset_name, split="all")
        self.dataset = self.dataset.shuffle(seed=self.training_config.seed)
        
        self.logger.info(f"Dataset size: {len(self.dataset)}")
        
        # Format dataset for chat template
        def format_chat_template(row):
            """Format a row using the chat template."""
            row_json = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}
            ]
            row["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
            return row
        
        self.dataset = self.dataset.map(
            format_chat_template,
            num_proc=4,
        )
        
        # Split dataset
        self.dataset = self.dataset.train_test_split(test_size=self.training_config.test_size, seed=self.training_config.seed)
        
        self.logger.info(f"Train size: {len(self.dataset['train'])}, Test size: {len(self.dataset['test'])}")
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Set up training arguments."""
        torch_dtype, _ = self.get_torch_dtype_and_attention()
        
        return TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            num_train_epochs=self.training_config.num_train_epochs,
            evaluation_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            logging_steps=self.training_config.logging_steps,
            warmup_steps=self.training_config.warmup_steps,
            logging_strategy="steps",
            learning_rate=self.training_config.learning_rate,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            group_by_length=True,
            report_to="wandb",
            run_name=self.model_config.wandb_run_name,
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            seed=self.training_config.seed,
        )
    
    def setup_trainer(self) -> None:
        """Set up the SFT trainer."""
        training_args = self.setup_training_arguments()
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            peft_config=self.peft_config,
            max_seq_length=self.model_config.max_seq_length,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_args,
            packing=False,
        )
        
        self.logger.info("Trainer set up successfully")
    
    def train(self) -> None:
        """Execute the training process."""
        self.training_logger.log_training_start(self.model_name, {
            "effective_batch_size": self.training_config.effective_batch_size,
            "learning_rate": self.training_config.learning_rate,
            "epochs": self.training_config.num_train_epochs,
        })
        
        # Initialize wandb
        wandb.init(
            project=self.model_config.wandb_project,
            job_type="training",
            name=self.model_config.wandb_run_name,
            tags=[self.model_config.base_model.split('/')[-1], "qlora", "fine-tuning"],
            config={
                "model": self.model_config.base_model,
                "dataset": self.model_config.dataset_name,
                "lora_r": self.model_config.lora_r,
                "lora_alpha": self.model_config.lora_alpha,
                "learning_rate": self.training_config.learning_rate,
                "effective_batch_size": self.training_config.effective_batch_size,
                "epochs": self.training_config.num_train_epochs,
            }
        )
        
        try:
            # Train the model
            self.trainer.train()
            
            # Save the model
            self.trainer.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            self.training_logger.log_checkpoint_saved("final", str(self.output_dir))
            
            # Push to hub if requested
            if self.training_config.push_to_hub:
                self.push_to_hub()
            
            self.training_logger.log_training_end(self.model_name, True)
            
        except Exception as e:
            self.training_logger.log_error(e, "training")
            raise
        finally:
            wandb.finish()
    
    def push_to_hub(self) -> None:
        """Push the trained model to Hugging Face Hub."""
        self.logger.info("Pushing model to Hugging Face Hub...")
        
        try:
            self.trainer.model.push_to_hub(
                self.model_config.new_model_name,
                use_temp_dir=False,
                private=self.training_config.private_hub
            )
            self.tokenizer.push_to_hub(
                self.model_config.new_model_name,
                use_temp_dir=False,
                private=self.training_config.private_hub
            )
            
            self.training_logger.log_model_pushed(self.model_name, self.model_config.new_model_name)
            
        except Exception as e:
            self.training_logger.log_error(e, "push_to_hub")
            raise
    
    def merge_and_save_full_model(self, output_path: str) -> Tuple[any, any]:
        """
        Merge LoRA weights with base model and save the full model.
        
        Args:
            output_path: Path to save the merged model
            
        Returns:
            Tuple of merged model and tokenizer
        """
        self.logger.info("Merging LoRA weights with base model...")
        
        # Reload base model
        torch_dtype, attn_implementation = self.get_torch_dtype_and_attention()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=True
        )
        
        if "instruct" in self.model_config.base_model.lower() or "it" in self.model_config.base_model.lower():
            base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        
        # Merge with LoRA weights
        model = PeftModel.from_pretrained(base_model, str(self.output_dir))
        model = model.merge_and_unload()
        
        # Save merged model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        self.logger.info(f"Merged model saved to {output_path}")
        
        return model, tokenizer
    
    def test_model(self, prompt: str = "What is cybersecurity?", max_new_tokens: int = 150) -> str:
        """
        Test the trained model with a sample prompt.
        
        Args:
            prompt: Test prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        self.logger.info("Testing model...")
        
        # Enable caching
        self.model.config.use_cache = True
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        outputs = pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = outputs[0]["generated_text"]
        self.logger.info(f"Generated text length: {len(generated_text)} characters")
        
        return generated_text
