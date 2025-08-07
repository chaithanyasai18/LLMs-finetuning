"""
System utilities for hardware detection and validation.

This module provides utilities for checking system requirements and hardware capabilities.
"""

import torch
import logging
from psutil import virtual_memory
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)


class SystemInfo:
    """System information and validation utilities."""
    
    @staticmethod
    def check_gpu_availability() -> dict:
        """Check GPU availability and properties."""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "count": 0,
            "devices": [],
            "total_memory_gb": 0
        }
        
        if torch.cuda.is_available():
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["count"]):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": device_props.name,
                    "memory_gb": device_props.total_memory / 1e9,
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                }
                gpu_info["devices"].append(device_info)
                gpu_info["total_memory_gb"] += device_info["memory_gb"]
        
        return gpu_info
    
    @staticmethod
    def check_ram() -> dict:
        """Check system RAM."""
        memory = virtual_memory()
        return {
            "total_gb": memory.total / 1e9,
            "available_gb": memory.available / 1e9,
            "percent_used": memory.percent
        }
    
    @staticmethod
    def check_nvidia_smi() -> dict:
        """Check NVIDIA SMI availability and GPU status."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return {
                "available": result.returncode == 0,
                "output": result.stdout if result.returncode == 0 else result.stderr
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {
                "available": False,
                "output": "nvidia-smi not found or timeout"
            }
    
    @staticmethod
    def get_optimal_torch_dtype() -> Tuple[torch.dtype, str]:
        """Determine optimal torch dtype and attention implementation."""
        if not torch.cuda.is_available():
            return torch.float32, "eager"
        
        # Check compute capability
        major, minor = torch.cuda.get_device_capability()
        
        # Ampere and newer (compute capability >= 8.0) support bfloat16 efficiently
        if major >= 8:
            try:
                # Check if flash attention is available
                import flash_attn
                logger.info("Using bfloat16 with flash attention 2")
                return torch.bfloat16, "flash_attention_2"
            except ImportError:
                logger.info("Flash attention not available, using bfloat16 with eager attention")
                return torch.bfloat16, "eager"
        else:
            logger.info("Using float16 with eager attention for older GPU")
            return torch.float16, "eager"
    
    @staticmethod
    def validate_system_requirements(model_name: str) -> dict:
        """Validate system requirements for a specific model."""
        gpu_info = SystemInfo.check_gpu_availability()
        ram_info = SystemInfo.check_ram()
        
        # Model-specific memory requirements (approximate)
        memory_requirements = {
            "gemma-2-2b": {"min_vram": 8, "recommended_vram": 12},
            "gemma-2-2b-instruct": {"min_vram": 8, "recommended_vram": 12},
            "mistral-7b-instruct": {"min_vram": 12, "recommended_vram": 16},
            "phi-3-mini": {"min_vram": 6, "recommended_vram": 10},
            "phi-2": {"min_vram": 4, "recommended_vram": 8},
        }
        
        requirements = memory_requirements.get(model_name, {"min_vram": 8, "recommended_vram": 12})
        
        validation = {
            "model": model_name,
            "gpu_available": gpu_info["available"],
            "gpu_count": gpu_info["count"],
            "total_vram_gb": gpu_info["total_memory_gb"],
            "ram_gb": ram_info["total_gb"],
            "meets_minimum": False,
            "meets_recommended": False,
            "warnings": [],
            "recommendations": []
        }
        
        if not gpu_info["available"]:
            validation["warnings"].append("No GPU detected. Training will be extremely slow on CPU.")
            validation["recommendations"].append("Use a machine with NVIDIA GPU for practical training.")
        else:
            validation["meets_minimum"] = gpu_info["total_memory_gb"] >= requirements["min_vram"]
            validation["meets_recommended"] = gpu_info["total_memory_gb"] >= requirements["recommended_vram"]
            
            if not validation["meets_minimum"]:
                validation["warnings"].append(
                    f"GPU memory ({gpu_info['total_memory_gb']:.1f}GB) below minimum "
                    f"requirement ({requirements['min_vram']}GB)"
                )
                validation["recommendations"].append("Use low memory configuration or smaller model.")
            elif not validation["meets_recommended"]:
                validation["warnings"].append(
                    f"GPU memory ({gpu_info['total_memory_gb']:.1f}GB) below recommended "
                    f"requirement ({requirements['recommended_vram']}GB)"
                )
                validation["recommendations"].append("Consider using low memory configuration for stability.")
        
        if ram_info["total_gb"] < 16:
            validation["warnings"].append(f"Low system RAM ({ram_info['total_gb']:.1f}GB)")
            validation["recommendations"].append("Consider upgrading to 32GB+ RAM for better performance.")
        
        return validation
    
    @classmethod
    def print_system_report(cls) -> None:
        """Print comprehensive system report."""
        gpu_info = cls.check_gpu_availability()
        ram_info = cls.check_ram()
        nvidia_info = cls.check_nvidia_smi()
        dtype, attention = cls.get_optimal_torch_dtype()
        
        print("System Information Report")
        print("=" * 50)
        
        # GPU Information
        print(f"CUDA Available: {gpu_info['available']}")
        if gpu_info["available"]:
            print(f"GPU Count: {gpu_info['count']}")
            print(f"Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")
            for device in gpu_info["devices"]:
                print(f"  - {device['name']}: {device['memory_gb']:.1f}GB "
                      f"(Compute {device['compute_capability']})")
        
        # RAM Information
        print(f"System RAM: {ram_info['total_gb']:.1f} GB "
              f"({ram_info['available_gb']:.1f} GB available)")
        
        # NVIDIA SMI
        print(f"NVIDIA-SMI: {'Available' if nvidia_info['available'] else 'Not Available'}")
        
        # Optimal settings
        print(f"Recommended dtype: {dtype}")
        print(f"Recommended attention: {attention}")
        
        print("=" * 50)
