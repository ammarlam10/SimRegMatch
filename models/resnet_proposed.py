import math
import torch
import torch.nn as nn
from torchvision.models import resnet50 as torchvision_resnet50


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, dropout=None, use_softplus=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, 1)
        self.dropout = nn.Dropout(p=dropout)
        
        # Use softplus for non-negative outputs (e.g., population prediction)
        self.use_softplus = use_softplus
        if use_softplus:
            self.softplus = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        encoding = x
        
        x = self.dropout(x)
        x = self.linear(x)
        
        # Apply softplus for non-negative outputs if enabled
        if self.use_softplus:
            x = self.softplus(x)

        return x, encoding


def resnet50(dropout, use_softplus=False, pretrained=True):
    """
    Create ResNet50 model with optional pretrained ImageNet weights.
    
    Args:
        dropout: Dropout probability
        use_softplus: Whether to use softplus activation
        pretrained: Whether to load ImageNet pretrained weights (default: True)
    
    Returns:
        ResNet model with regression head
    """
    if pretrained:
        # Load pretrained ResNet50 from torchvision
        print("Loading ImageNet pretrained ResNet50 weights...")
        pretrained_model = torchvision_resnet50(pretrained=True)
        
        # Create our custom model
        model = ResNet(Bottleneck, [3, 4, 6, 3], dropout, use_softplus=use_softplus)
        
        # Copy pretrained weights (excluding final FC layer)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out the final FC layer (fc.weight, fc.bias)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and not k.startswith('fc.')}
        
        # Update model with pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"Loaded pretrained weights for {len(pretrained_dict)} layers")
        return model
    else:
        return ResNet(Bottleneck, [3, 4, 6, 3], dropout, use_softplus=use_softplus)
