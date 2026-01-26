import torch
import torch.nn as nn
import timm


class EfficientNetWrapper(nn.Module):
    """
    Wrapper for EfficientNet-B0 from timm that matches the ResNet interface.
    Returns (prediction, encoding) where encoding is the feature vector before the final layer.
    """
    
    def __init__(self, dropout=0.1, use_softplus=False):
        super(EfficientNetWrapper, self).__init__()
        
        # Load EfficientNet-B0 from timm (pretrained=False for now, can be changed)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        # Get the feature dimension from the backbone
        # EfficientNet-B0 outputs 1280 features
        feature_dim = 1280
        
        # Add dropout and final regression layer
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(feature_dim, 1)
        
        # Use softplus for non-negative outputs (e.g., population prediction)
        self.use_softplus = use_softplus
        if use_softplus:
            self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Get features from backbone
        encoding = self.backbone(x)
        
        # Apply dropout and get prediction
        x = self.dropout(encoding)
        prediction = self.linear(x)
        
        # Apply softplus for non-negative outputs if enabled
        if self.use_softplus:
            prediction = self.softplus(prediction)
        
        return prediction, encoding


def efficientnet_b0(dropout=0.1, use_softplus=False):
    """
    Factory function to create EfficientNet-B0 model matching resnet50 interface.
    
    Args:
        dropout: Dropout probability (default: 0.1)
        use_softplus: Whether to use softplus activation for non-negative outputs (default: False)
    
    Returns:
        EfficientNetWrapper model
    """
    return EfficientNetWrapper(dropout=dropout, use_softplus=use_softplus)
