import torch
from transformers import (
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration
)


def load_scl_encoder_weights(model, encoder_path, model_type="encoder", freeze_config=None):
    """
    Load SCL pretrained encoder weights and optionally freeze layers.
    
    Args:
        model: Classification model
        encoder_path: Path to SCL encoder weights
        model_type: "encoder" or "encoder-decoder"
        freeze_config: Dict with freeze settings
    """
    # Load state dict
    state_dict = torch.load(encoder_path, map_location="cpu")
    
    # Load weights
    if model_type == "encoder":
        # For DeBERTa
        missing, unexpected = model.deberta.load_state_dict(state_dict, strict=False)
        encoder = model.deberta
    else:
        # For T5
        missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
        encoder = model.encoder
    
    print("Loaded SCL encoder weights:")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    # Freeze layers if specified
    if freeze_config:
        if model_type == "encoder":
            # Freeze embeddings
            if freeze_config.get('freeze_embeddings', False):
                for param in encoder.embeddings.parameters():
                    param.requires_grad = False
                print("Froze embeddings")
            
            # Freeze first N encoder layers
            freeze_layers = freeze_config.get('freeze_layers', 0)
            if freeze_layers > 0:
                for layer in encoder.encoder.layer[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
                print(f"Froze first {freeze_layers} encoder layers")
        
        else:  # T5
            # Freeze first N blocks
            freeze_layers = freeze_config.get('freeze_layers', 0)
            if freeze_layers > 0:
                n_blocks = len(encoder.block)
                freeze_until = n_blocks - freeze_layers
                for block in encoder.block[:freeze_until]:
                    for param in block.parameters():
                        param.requires_grad = False
                print(f"Froze all but last {freeze_layers} encoder blocks")
    
    return model
