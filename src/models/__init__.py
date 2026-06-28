# src/models/__init__.py

from .architecture import ProductRelevanceClassifier
from .inference import run_test_inference

__all__ = [
    "ProductRelevanceClassifier",
    "run_test_inference"
]