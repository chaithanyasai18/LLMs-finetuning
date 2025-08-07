"""
Training module for QLora fine-tuning package.

Contains the main training logic and CLI interface.
"""

import sys
import argparse
import json
from pathlib import Path

from . import QLorAFineTuner, TrainingConfig, ModelRegistry
from .utils import setup_logging, SystemInfo


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="QLora Fine-tuning for Multiple LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m qlora_finetuning --model gemma-2-2b-instruct
  python train.py --model mistral-7b-instruct --config configs/mistral_config.json
  python train.py --model phi-3-mini --epochs 5 --batch_size 2
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=ModelRegistry.list_models(),
        help="Model to fine-tune"
    )
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint steps")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Hub settings
    parser.add_argument("--no_push_to_hub", action="store_true", help="Don't push to Hugging Face Hub")
    parser.add_argument("--public_hub", action="store_true", help="Make hub repository public")
    
    # Model merging
    parser.add_argument("--merge_and_save", type=str, help="Path to save merged model")
    
    # Testing
    parser.add_argument("--test_prompt", type=str, default="What is cybersecurity?", help="Test prompt")
    parser.add_argument("--no_test", action="store_true", help="Skip model testing")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, help="Log file path")
    
    # System utilities
    parser.add_argument("--check_system", action="store_true", help="Check system requirements only")
    parser.add_argument("--list_models", action="store_true", help="List supported models")
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        console_output=True
    )
    
    # Handle utility commands
    if args.list_models:
        print("Supported models:")
        for model in ModelRegistry.list_models():
            config = ModelRegistry.get_model_config(model)
            print(f"  - {model}: {config.base_model}")
        return
    
    if args.check_system:
        SystemInfo.print_system_report()
        if args.model:
            validation = SystemInfo.validate_system_requirements(args.model)
            print(f"\nValidation for {args.model}:")
            print(f"  Meets minimum requirements: {validation['meets_minimum']}")
            print(f"  Meets recommended requirements: {validation['meets_recommended']}")
        return
    
    try:
        # Load configuration
        if args.config:
            training_config = TrainingConfig.from_json(args.config)
            print(f"Loaded configuration from {args.config}")
        else:
            training_config = TrainingConfig(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                push_to_hub=not args.no_push_to_hub,
                private_hub=not args.public_hub,
            )
        
        # Initialize fine-tuner
        fine_tuner = QLorAFineTuner(args.model, training_config)
        
        # Validate system requirements
        if not fine_tuner.validate_system_requirements():
            print("WARNING: System may not meet minimum requirements. Continue? (y/n)")
            if input().lower() != 'y':
                return
        
        # Set up authentication
        auth_status = fine_tuner.setup_authentication()
        if not auth_status["huggingface"]:
            print("WARNING: Hugging Face authentication failed. Some models may not be accessible.")
        if not auth_status["wandb"]:
            print("WARNING: W&B authentication failed. Experiment tracking will be limited.")
        
        # Load model and tokenizer
        fine_tuner.load_model_and_tokenizer()
        
        # Configure PEFT
        fine_tuner.configure_peft()
        
        # Load and prepare dataset
        fine_tuner.load_and_prepare_dataset()
        
        # Set up trainer
        fine_tuner.setup_trainer()
        
        # Train the model
        fine_tuner.train()
        
        # Test the model (unless skipped)
        if not args.no_test:
            print(f"\nTesting model with prompt: '{args.test_prompt}'")
            result = fine_tuner.test_model(args.test_prompt)
            print("Generated response:")
            print("-" * 50)
            print(result)
            print("-" * 50)
        
        # Merge and save full model if requested
        if args.merge_and_save:
            print(f"\nMerging and saving full model to {args.merge_and_save}")
            fine_tuner.merge_and_save_full_model(args.merge_and_save)
        
        print("\nFine-tuning completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFine-tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
