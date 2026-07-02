# src/models/__init__.py
import logging
import sys
import torch

from .architecture import ProductRelevanceClassifier
from .inference import run_test_inference

# 1. Initialize Module-Level Telemetry
logger = logging.getLogger("eletrade.models")
logger.setLevel(logging.INFO)

# Create console handler with formatting if it doesn't already exist
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 2. Hardware Guardrail (Fail-Fast Execution)
def _verify_hardware() -> None:
    """
    Executes upon import to ensure the host machine possesses the necessary
    hardware accelerators to run the transformer architecture.
    """
    if torch.cuda.is_available():
        logger.info(f"CUDA initialized. Target accelerator: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS (Metal Performance Shaders) initialized.")
    else:
        logger.warning(
            "CRITICAL: No hardware accelerator (CUDA/MPS) detected. "
            "Model will fallback to CPU. Inference and training will be severely bottlenecked."
        )

# Execute the guardrail immediately when the package is imported
_verify_hardware()

# 3. Explicit Public Interface
__all__ = [
    "ProductRelevanceClassifier",
    "run_test_inference"
]