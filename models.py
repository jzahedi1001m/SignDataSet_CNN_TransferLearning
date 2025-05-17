import os
import math
import random
import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

################################################################################

class SignCnn(nn.Module):
    """
    SignCNN: conv -> relu -> pool -> conv -> relu -> maxpool -> fc
    """
    def __init__(self, num_classes):
        super(SignCnn, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding = 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding= 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()       
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)      
        return self.fc(x)

##################################################################################

class SignMobileNetV2(nn.Module):
    """
    MobileNetV2-based sign detector with additional custom layers for classification.
    """
    def __init__(self, num_classes):
        super(SignMobileNetV2, self).__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        self.last_channel = base_model.last_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

####################### ResNet50 ##################################
# Identity Block
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, f, filters):
        super(IdentityBlock, self).__init__()
        
        F1, F2, F3 = filters
        
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding='same')  
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(F3)

    def forward(self, x):       
        shortcut = x        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += shortcut
        x = F.relu(x)
        return x

# Convolutional Block
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, f, filters, s=2):
        super(ConvolutionalBlock, self).__init__()

        F1, F2, F3 = filters
        
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.shortcut_conv = nn.Conv2d(in_channels, F3, kernel_size=1, stride=s)
        self.shortcut_bn = nn.BatchNorm2d(F3)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        shortcut = self.shortcut_conv(shortcut)
        shortcut = self.shortcut_bn(shortcut)
        x += shortcut
        return F.relu(x)

# ResNet50 model
class SignResnet50(nn.Module):
    """
    SignResnet50: ResNet50 
    """
    def __init__(self, input_channels=3, num_classes=6):
        super(SignResnet50, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_block1 = ConvolutionalBlock(64, f=3, filters=[64, 64, 256], s=1)
        self.id_block1 = IdentityBlock(256, f=3, filters=[64, 64, 256])
        self.id_block2 = IdentityBlock(256, f=3, filters=[64, 64, 256])
        self.conv_block2 = ConvolutionalBlock(256, f=3, filters=[128, 128, 512], s=2)
        self.id_block3 = IdentityBlock(512, f=3, filters=[128, 128, 512])
        self.id_block4 = IdentityBlock(512, f=3, filters=[128, 128, 512])
        self.id_block5 = IdentityBlock(512, f=3, filters=[128, 128, 512])
        self.conv_block3 = ConvolutionalBlock(512, f=3, filters=[256, 256, 1024], s=2)
        self.id_block6 = IdentityBlock(1024, f=3, filters=[256, 256, 1024])
        self.id_block7 = IdentityBlock(1024, f=3, filters=[256, 256, 1024])
        self.id_block8 = IdentityBlock(1024, f=3, filters=[256, 256, 1024])
        self.id_block9 = IdentityBlock(1024, f=3, filters=[256, 256, 1024])
        self.id_block10 = IdentityBlock(1024, f=3, filters=[256, 256, 1024])
        self.conv_block4 = ConvolutionalBlock(1024, f=3, filters=[512, 512, 2048], s=2)
        self.id_block11 = IdentityBlock(2048, f=3, filters=[512, 512, 2048])
        self.id_block12 = IdentityBlock(2048, f=3, filters=[512, 512, 2048])
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(2048, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv_block1(x)
        x = self.id_block1(x)
        x = self.id_block2(x)
        x = self.conv_block2(x)
        x = self.id_block3(x)
        x = self.id_block4(x)
        x = self.id_block5(x)
        x = self.conv_block3(x)
        x = self.id_block6(x)
        x = self.id_block7(x)
        x = self.id_block8(x)
        x = self.id_block9(x)
        x = self.id_block10(x)
        x = self.conv_block4(x)
        x = self.id_block11(x)
        x = self.id_block12(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to vector       
        return self.fc(x)