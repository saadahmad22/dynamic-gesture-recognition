'''Model classes for the Dynamic Optical Flow Recognizer'''

from config import nn, torch
from conv import conv_builder
from seblock import SEBlock

class DynamicOpticalFlowRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(DynamicOpticalFlowRecognizer, self).__init__()

        self.spatial = nn.Sequential(
            conv_builder(3, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.AvgPool2d(2),

            conv_builder(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.AvgPool2d(2),

            conv_builder(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.AvgPool2d(2)
        )
        self.temporal = nn.Sequential(
            conv_builder(3, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2),

            conv_builder(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2),

            conv_builder(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2)
        )

        # Feature Fusion
        self.pool_fusion = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),  # 128 from spatial and 128 from temporal
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, spatial_x, temporal_x):
        spatial_x = self.spatial(spatial_x)
        temporal_x = self.temporal(temporal_x)

        # Feature Fusion
        spatial_x = self.pool_fusion(spatial_x).view(spatial_x.size(0), -1)
        temporal_x = self.pool_fusion(temporal_x).view(temporal_x.size(0), -1)

        # Concatenate spatial and temporal features
        fused = torch.cat([spatial_x, temporal_x], dim=1)

        # Fully Connected Layers
        fused = self.fc(fused)

        return fused