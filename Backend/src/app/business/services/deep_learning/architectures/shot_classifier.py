"""
Shot Classification Model Architecture.

Contains the EnhancedVideoClassifier model that combines video features (R(2+1)D-18) 
with pose keypoints (Bi-LSTM) for shot type classification.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import video as video_models
import numpy as np
from typing import Dict, Tuple


class EnhancedVideoClassifier(nn.Module):
    """
    Enhanced classifier combining video features (R(2+1)D-18) with pose keypoints (Bi-LSTM).
    
    Architecture:
    - Video branch: R(2+1)D-18 pretrained backbone
    - Pose branch: Bidirectional LSTM with attention
    - Fusion: Concatenation + MLP classifier
    """

    def __init__(
        self, 
        num_classes: int, 
        pose_input_size: int = 51, 
        pose_hidden_size: int = 256, 
        pretrained_video: bool = True
    ):
        super().__init__()
        
        # Video backbone
        self.video_model = video_models.r2plus1d_18(
            weights=video_models.R2Plus1D_18_Weights.DEFAULT if pretrained_video else None
        )
        num_video_ftrs = self.video_model.fc.in_features
        self.video_model.fc = nn.Identity()

        # Pose LSTM with attention
        self.pose_lstm = nn.LSTM(
            pose_input_size, 
            pose_hidden_size,
            num_layers=3, 
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        
        self.pose_attention = nn.Sequential(
            nn.Linear(pose_hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Fusion classifier
        fusion_size = num_video_ftrs + (pose_hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, video_x: torch.Tensor, pose_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            video_x: Video tensor of shape (B, C, T, H, W)
            pose_x: Pose keypoints tensor of shape (B, T, 51)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Video features
        video_features = self.video_model(video_x)
        
        # Pose features with attention
        pose_out, _ = self.pose_lstm(pose_x)
        attention_weights = self.pose_attention(pose_out)
        pose_features = torch.sum(pose_out * attention_weights, dim=1)
        
        # Fusion
        combined = torch.cat((video_features, pose_features), dim=1)
        return self.classifier(combined)


def load_shot_model(
    model_path: str, 
    device: torch.device
) -> Tuple[EnhancedVideoClassifier, Dict[int, str], int, float]:
    """
    Load trained shot classification model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, idx_to_label, other_class_idx, confidence_threshold)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    labels_map = checkpoint['labels_map']
    num_classes = len(labels_map)
    config = checkpoint.get('config', {})

    model = EnhancedVideoClassifier(
        num_classes=num_classes,
        pose_input_size=51,
        pose_hidden_size=config.get('pose_hidden_size', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Build index to label mapping
    idx_to_label = checkpoint.get('idx_to_label', {idx: name for name, idx in labels_map.items()})
    if idx_to_label and isinstance(list(idx_to_label.keys())[0], str):
        idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    other_class_idx = checkpoint.get('other_class_idx', labels_map.get('other', -1))
    confidence_threshold = config.get('confidence_threshold', 0.6)

    print(f"Shot model loaded: {model_path}")
    print(f"Classes: {', '.join([idx_to_label[i] for i in range(num_classes)])}")
    print(f"Confidence threshold: {confidence_threshold}")

    return model, idx_to_label, other_class_idx, confidence_threshold


def preprocess_clip(
    frames: torch.Tensor, 
    clip_len: int = 32, 
    resolution: Tuple[int, int] = (112, 112)
) -> torch.Tensor:
    """
    Preprocess video frames for model input.
    
    Args:
        frames: Tensor of frames (T, C, H, W)
        clip_len: Target number of frames
        resolution: Target spatial resolution (H, W)
        
    Returns:
        Preprocessed tensor of shape (1, C, T, H, W)
    """
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    total = len(frames)
    if total >= clip_len:
        indices = np.linspace(0, total - 1, clip_len, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.pad(indices, (0, clip_len - total), 'edge')

    sampled = frames[indices]
    tensor = sampled.permute(1, 0, 2, 3).float() / 255.0
    tensor = transforms.functional.resize(tensor, resolution)
    tensor = (tensor - norm_mean) / norm_std

    return tensor.unsqueeze(0)