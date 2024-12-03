import torch.nn as nn


class ASLNet(nn.Module):
    def __init__(self, num_classes):
        super(ASLNet, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2)
        )

        # Second block with residual
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.32),
            nn.MaxPool2d(2)
        )

        # Third block with residual
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )

        # Calculate the size of flattened features
        self.flatten_size = 256 * 12 * 12

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)

        return x