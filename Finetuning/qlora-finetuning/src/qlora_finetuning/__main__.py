"""
Module entry point for QLora fine-tuning package.

Allows running the package as a module: python -m qlora_finetuning
"""

from .train import main

if __name__ == "__main__":
    main()
