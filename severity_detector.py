import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x is the feature map from the ResNet model
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)  # Sigmoid to create an attention mask

        return x * attention , attention # Apply attention mask to the input feature map

class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        # Load pre-trained ResNet and remove its fully connected layer
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))

        # Spatial Attention Module
        self.attention = SpatialAttentionModule()

        # Fully connected layer for classification
        self.fc = nn.Linear(2048, 3)  # Assuming 3 classes for severity

    def forward(self, x ,return_attention=False):
        # Extract features from ResNet
        features = self.resnet(x)

        # Apply attention
        attention_applied,attention_map = self.attention(features)

        # Global average pooling
        gap = nn.functional.adaptive_avg_pool2d(attention_applied, (1, 1))
        gap = gap.view(gap.size(0), -1)

        # Classification
        output = self.fc(gap)
        
        if return_attention:
            return output, attention_map
        else:
            return output

class SeverityDetector:
    damage = ['Minor','Moderate','Severe'] 
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModifiedResNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.damage = ['Minor','Moderate','Severe'] 
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Predict the severity of damage in an image.
        """
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        return self.damage[predicted.item()]