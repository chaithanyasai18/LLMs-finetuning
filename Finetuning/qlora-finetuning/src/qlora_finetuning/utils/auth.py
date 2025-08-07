"""
Authentication utilities for external services.

This module handles authentication for Hugging Face Hub and Weights & Biases.
"""

import os
import logging
from typing import Optional
import wandb
from huggingface_hub import login as hf_login

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Manages authentication for external services."""
    
    @staticmethod
    def setup_huggingface_auth(token: Optional[str] = None) -> bool:
        """
        Set up Hugging Face authentication.
        
        Args:
            token: HF token (optional, can be set via environment)
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if token:
                hf_login(token=token)
                logger.info("Hugging Face authentication successful (provided token)")
                return True
            elif os.getenv("HUGGINGFACE_TOKEN"):
                hf_login(token=os.getenv("HUGGINGFACE_TOKEN"))
                logger.info("Hugging Face authentication successful (environment token)")
                return True
            else:
                logger.warning("No Hugging Face token provided. Some models may not be accessible.")
                return False
        except Exception as e:
            logger.error(f"Hugging Face authentication failed: {e}")
            return False
    
    @staticmethod
    def setup_wandb_auth(token: Optional[str] = None) -> bool:
        """
        Set up Weights & Biases authentication.
        
        Args:
            token: W&B token (optional, can be set via environment)
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if token:
                wandb.login(key=token)
                logger.info("Weights & Biases authentication successful (provided token)")
                return True
            elif os.getenv("WANDB_TOKEN"):
                wandb.login(key=os.getenv("WANDB_TOKEN"))
                logger.info("Weights & Biases authentication successful (environment token)")
                return True
            else:
                logger.warning("No W&B token provided. Experiment tracking will be limited.")
                return False
        except Exception as e:
            logger.error(f"Weights & Biases authentication failed: {e}")
            return False
    
    @classmethod
    def setup_all_auth(
        cls, 
        hf_token: Optional[str] = None, 
        wandb_token: Optional[str] = None
    ) -> dict:
        """
        Set up authentication for all services.
        
        Args:
            hf_token: Hugging Face token
            wandb_token: Weights & Biases token
        
        Returns:
            Dictionary with authentication status for each service
        """
        return {
            "huggingface": cls.setup_huggingface_auth(hf_token),
            "wandb": cls.setup_wandb_auth(wandb_token)
        }
