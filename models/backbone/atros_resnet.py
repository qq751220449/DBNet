import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)  # ? Why no bias


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    # ? Why no bias: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, bias=False, padding=1)
        self.bn1 = norm_layer(planes)
        self.activate1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.activate2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, bias=False, padding=1)
        self.bn3 = norm_layer(planes)
        self.activate3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate3(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.activate1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.activate2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=4*planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(4*planes)
        self.activate3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride=16, BatchNormal=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
        elif output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 4]
        elif output_stride == 4:
            stride = [1, 1, 1, 1]
            dilation = [1, 2, 4, 8]
        else:
            raise NotImplementedError
        self.out_channels = []
        if BatchNormal is None:
            BatchNormal = nn.BatchNorm2d
        blocks = [1, 2, 4]
        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNormal(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0], dilation=dilation[0], BatchNorm=BatchNormal)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1], dilation=dilation[1], BatchNorm=BatchNormal)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2], dilation=dilation[2], BatchNorm=BatchNormal)
        # self.layer4 = self._make_MG_layer(block, 512, blocks=blocks, stride=stride[3], dilation=dilation[3], BatchNorm=BatchNormal)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3], dilation=dilation[3], BatchNorm=BatchNormal)

        self._init_weight()

        # if pretrained:
        #     self._load_pretrained_model()
    def get_channels(self):
        return self.out_channels

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                BatchNorm(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=BatchNorm))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？


    def _make_MG_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                BatchNorm(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, blocks[0] * dilation, downsample, BatchNorm))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, len(blocks)):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, dilation=blocks[_] * dilation, norm_layer=BatchNorm))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5  # 返回多个尺度的特征图

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def atros_resnet101(output_stride=4, BatchNorm=nn.BatchNorm2d, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def atros_resnet18(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


if __name__ == "__main__":
    import torch
    model = atros_resnet18(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=4)
    input = torch.rand(1, 3, 1024, 1024)
    x2, x3, x4, x5 = model(input)
    print(x2.size())
    print(x3.size())
    print(x4.size())
    print(x5.size())
    print(model.get_channels())
