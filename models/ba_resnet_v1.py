import torch
import torch.nn as nn
from thop import profile, clever_format


def conv3x3(in_channel, out_channel, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channel, out_channel, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BALayer(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(BALayer, self).__init__()
        self.fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_channel, out_channel // reduction, bias=False),
                nn.BatchNorm1d(out_channel // reduction)
            )
                for in_channel in in_channel]
        )
        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features):
        b, c, _, _ = features[-1].size()

        fusions = [self.fusions[i](features[i].view(b, -1)) for i in range(len(features))]
        fusion = sum(fusions)

        att_weights = self.generation(fusion).view(b, c, 1, 1)

        return att_weights


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # Global Average Pooling
        self.avg = nn.AdaptiveAvgPool2d(1)
        # Bridge Attention Layer
        self.ba = BALayer([out_channel, out_channel], out_channel)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        x1 = self.avg(out)

        out = self.conv2(out)
        out = self.bn2(out)
        x2 = self.avg(out)

        # Bridge Channel Attention
        attn = self.ba([x1, x2])
        out = out * attn.expand_as(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel, stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = conv1x1(out_channel, out_channel * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        # Global Average Pooling
        self.avg = nn.AdaptiveAvgPool2d(1)
        # Bridge Attention Layer
        self.ba = BALayer([out_channel, out_channel, out_channel * self.expansion], out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        x1 = self.avg(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        x2 = self.avg(out)

        out = self.conv3(out)
        out = self.bn3(out)
        x3 = self.avg(out)

        # Bridge Channel Attention
        attn = self.ba([x1, x2, x3])
        out = out * attn.expand_as(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, channel * block.expansion, stride),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ba_resnet18_v1(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ba_resnet34_v1(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ba_resnet50_v1(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ba_resnet101_v1(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


if __name__ == '__main__':
    model = ba_resnet50_v1(num_classes=1000)

    input = torch.rand([10, 3, 224, 224])  # [B, C, H, W]
    print('Image_input.shape = ', input.shape)

    output = model(input)
    print('Model_output.shape = ', output.shape)

    flops, params = profile(model, inputs=(torch.rand([1, 3, 224, 224]), ))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"The number of parameters: {params}")
    print(f"number of GFLOPs: {flops}")

