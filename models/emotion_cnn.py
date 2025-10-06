"""
CNN Model Architecture for Emotion Detection
"""

import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    """
    CNN model for emotion detection based on ResNet18
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class EmotionCNNSimple(nn.Module):
    """
    Simple CNN model for emotion detection
    """
    def __init__(self, num_classes=7):
        super(EmotionCNNSimple, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_emotion_model(model_type='resnet', num_classes=7):
    """
    Factory function to create emotion detection models
    
    Args:
        model_type (str): Type of model ('resnet' or 'simple')
        num_classes (int): Number of emotion classes
        
    Returns:
        torch.nn.Module: The created model
    """
    if model_type.lower() == 'resnet':
        return EmotionCNN(num_classes)
    elif model_type.lower() == 'simple':
        return EmotionCNNSimple(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")