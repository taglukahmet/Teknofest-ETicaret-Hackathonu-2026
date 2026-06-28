# src/__init__.py

# Expose the global configuration engine to the top-level package
from .config import config

# Restrict wildcard imports to just the config object
__all__ = ["config"]