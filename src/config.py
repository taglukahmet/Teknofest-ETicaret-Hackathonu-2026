import os
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    # --- System Paths ---
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(ROOT_DIR, "data_raw")
    OUTPUT_DIR: str = os.path.join(ROOT_DIR, "outputs")
    
    # --- Data Processing Parameters ---
    SEED: int = 42
    NUM_FOLDS: int = 5
    MAX_SEQUENCE_LENGTH: int = 256
    NEGATIVE_SAMPLING_RATIO: int = 3  # Ratio of YOK rows generated per VAR row
    
    # --- Model Hyperparameters ---
    MODEL_NAME: str = "microsoft/mdeberta-v3-base"
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    EPOCHS: int = 3
    
    # --- Post-Processing Parameters ---
    DEFAULT_THRESHOLD: float = 0.50

# Instantiate a global config object for the team to import
config = PipelineConfig()