import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

# Import the global configuration
from src.config import config

class ProductRelevanceClassifier(nn.Module):
    """
    Cross-Encoder architecture for calculating semantic match relevance 
    between search queries and e-commerce product catalogs.
    """
    def __init__(self, model_name: str = config.MODEL_NAME, num_labels: int = 2):
        super().__init__()
        
        # 1. Instantiate the structural blueprint
        self.transformer_config = AutoConfig.from_pretrained(
            model_name,
            num_labels = num_labels,
            problem_type="single_label_classification"
        )

        # 2. Download and map the pre-trained weights to the classification head
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config = self.transformer_config,
            use_safetensors = True # this flag here to bypass the security block securely
        )

        # 3. Optional production optimization: Gradient Checkpointing
        # Reduces VRAM usage by ~30% at the cost of ~20% slower training speed
        if hasattr(self.encoder.config, "gradient_checkpointing"):
            self.encoder.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Executes the cross-attention forward pass.
        Returns unnormalized raw logits of shape (batch_size, num_labels).
        """
        outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        # We strictly return logits. The loss function (CrossEntropyLoss) 
        # in the training loop expects raw logits, not softmax probabilities.
        return outputs.logits
    
if __name__ == "__main__":
    # Local execution test to verify the architecture compiles
    print("Initializing architecture model template...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = ProductRelevanceClassifier()
    model.to(device)
    
    # Create dummy tensors to simulate a batch of 2 text sequences, each 16 tokens long
    dummy_input_ids = torch.randint(0, 1000, (2, 16)).to(device)
    dummy_attention_mask = torch.ones((2, 16)).to(device)
    
    # Run a test forward pass
    test_logits = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    print(f"Model successfully mapped to target device: {device}")
    print(f"Test Forward Pass Output Shape: {test_logits.shape} (Expected: [2, 2])")