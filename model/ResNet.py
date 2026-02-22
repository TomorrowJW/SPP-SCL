import torch
import torch.nn as nn
from torchvision import models

class resnet(nn.Module):
    """
    ResNet-50 backbone for image feature extraction.
    用于图像特征提取的ResNet-50主干网络。

    Paper description:
        "We adopt ResNet-50 as the image encoder and
         use the output of the last convolutional block."

    Output:
        Feature map from layer4: [B, 2048, 7, 7]
    """
    def __init__(self):
        super().__init__()
        # 加载标准 ResNet-50 结构
        self.net = models.resnet50(pretrained=False)
        # 加载 ImageNet 预训练权重
        self.net.load_state_dict(torch.load('F:/BBBB/MVSA-S/pre_train/resnet50-19c8e357.pth'), strict=False)

    def forward(self,input):
        """
        Forward pass of ResNet-50 backbone.

        Input:
            input : [B, 3, H, W]

        Output:
            output : [B, 2048, 7, 7]
                     feature map from layer4
        """

        # Standard ResNet forward until layer4
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output3 = self.net.layer3(output)

        # Final convolution block (used in paper)
        output = self.net.layer4(output3)

        return output



