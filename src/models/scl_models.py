import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, T5EncoderModel


class DeBERTaSCLModel(nn.Module):
    """DeBERTa with projection head for contrastive learning."""
    
    def __init__(self, model_name: str, projection_dim: int = 256):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hid_dim = self.encoder.config.hidden_size
        
        # Projection head: Linear -> ReLU -> Linear
        self.projection = nn.Sequential(
            nn.Linear(hid_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward_encoder(self, input_ids, attention_mask):
        """Forward pass through encoder."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token or pooler output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_emb = outputs.pooler_output
        else:
            cls_emb = outputs.last_hidden_state[:, 0]
            
        return cls_emb
    
    def get_projection(self, input_ids, attention_mask):
        """Get normalized projections."""
        h = self.forward_encoder(input_ids, attention_mask)
        z = self.projection(h)
        z = self.layer_norm(z)
        z = F.normalize(z, dim=1)
        return z


class T5SCLModel(nn.Module):
    """T5 Encoder with projection head for contrastive learning."""
    
    def __init__(self, model_name: str, projection_dim: int = 256, cache_dir: str = None):
        super().__init__()
        
        self.encoder = T5EncoderModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        hid_dim = self.encoder.config.d_model
        
        self.projection = nn.Sequential(
            nn.Linear(hid_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward_encoder(self, input_ids, attention_mask):
        """Forward pass with mean pooling."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # [B, seq_len, d_model]
        
        # Mean pooling (ignore padding)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / denom
        
        return mean_pooled
    
    def get_projection(self, input_ids, attention_mask):
        """Get normalized projections."""
        h = self.forward_encoder(input_ids, attention_mask)
        z = self.projection(h)
        z = self.layer_norm(z)
        z = F.normalize(z, dim=1)
        return z
