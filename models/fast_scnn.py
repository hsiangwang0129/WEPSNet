###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################

"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [2, 5, 2])
        self.feature_fusion = FeatureFusionModuleConcat(64, 128, 128)
        self.classifier = Classifier(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)





class _StripFocus(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        """
        StripFocus module:
        - in_channels: Number of input channels
        - out_channels: Number of output channels (default is 64)
        """
        super(_StripFocus, self).__init__()
        self.out_channels = out_channels

        # 1x1 Conv for channel reduction 
        # (input channels are 4 times in_channels, output channels are out_channels)
        self.channel_reduction = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """
        Forward method of StripFocus:
        - Input: x, with shape (B, C, H, W)
        - Process:
            1. Horizontally split into 4 sections
            2. Rearrange the split sections into the channel dimension 
               (C becomes 4C, W becomes W/4)
            3. Apply 1x1 Conv to reduce channels to the specified out_channels
        - Output: (B, out_channels, H, W/4)
        """
        B, C, H, W = x.shape
        assert W % 4 == 0, "Width must be divisible by 4"

        # Split horizontally into 4 sections
        strips = torch.chunk(x, 4, dim=-1)  # Shape: [(B, C, H, W/4), (B, C, H, W/4), ...]

        # Concatenate along the channel dimension (B, 4C, H, W/4)
        x = torch.cat(strips, dim=1) 
        # Apply 1x1 Conv for channel reduction, output shape: (B, out_channels, H, W/4)
        x = self.channel_reduction(x)

        return x




class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class FAFN(nn.Module):
    """FAFN used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(FAFN, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            # _ConvBNReLU(in_channels, in_channels * t, 1),
            nn.Conv2d(in_channels, in_channels * t, 1, bias=True),

            # dw
            _DWConv(in_channels * t, in_channels * t, stride),

            # pw-linear
            # nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=True),
            # nn.BatchNorm2d(out_channels)

            # LeakyReLU (Negative Slope = 0.01)
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2,1)
        self.stripfocus = _StripFocus(dw_channels1,dw_channels2)
        # self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        # x = self.dsconv1(x)
        x = self.stripfocus(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(2, 5, 2), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(FAFN, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(FAFN, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(FAFN, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModuleConcat(nn.Module):
    """Feature Fusion Module (FFM) with Concatenation Instead of Addition"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModuleConcat, self).__init__()
        self.scale_factor = scale_factor

        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # use 1x1 Conv compress channel,ensure the output size match out_channels
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),  # Concatenation will make C times 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, higher_res_feature, lower_res_feature):
        # Upsample lower_res_feature to match higher_res_feature's spatial dimensions
        lower_res_feature = F.interpolate(lower_res_feature, 
                                          size=higher_res_feature.size()[2:], 
                                          mode='bilinear', 
                                          align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        # Concatenate features
        fused_feature = torch.cat([higher_res_feature, lower_res_feature], dim=1)  

        # Use 1x1 Conv to compress channel and ensure output size matches out_channels
        out = self.conv_fusion(fused_feature)
        return out



class FeatureFusionModuleConcat(nn.Module):
    """Feature Fusion Module (FFM) with Concatenation Instead of Addition"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModuleConcat, self).__init__()
        self.scale_factor = scale_factor

        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # use 1x1 Conv compress channel,ensure the output size match out_channels
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),  # Concatenation will make C times 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        # Add To Concatenation
        fused_feature = torch.cat([higher_res_feature, lower_res_feature], dim=1)  

        # use 1x1 Conv compress channel,ensure the output size match out_channels
        out = self.conv_fusion(fused_feature)
        # print("we're in FFM_CONCAT!")
        # print("out.shape = ",out.shape) # torch.Size([2, 128, 192, 48])
        return out





class InvertedStrip(nn.Module):
    """
    InvertedStrip: Reverses the effect of StripFocus by rearranging feature map from channels back to spatial width.
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
        - in_channels: The input number of channels (should be 4 times out_channels)
        - out_channels: The desired number of output channels after transformation
        """
        super(InvertedStrip, self).__init__()
        assert in_channels % 4 == 0, "InvertedStrip requires in_channels to be divisible by 4"
        self.out_channels = out_channels

    def forward(self, x):
        """
        Forward method:
        - Input: x with shape (B, C, H, W) where C = 4 * out_channels
        - Process: Rearrange channels back to spatial width
        - Output: (B, out_channels, H, W*4)
        """
        B, C, H, W = x.shape
        assert C % 4 == 0, "Input channels must be divisible by 4"

        # Reshape C (通道) → 變成 4 個區塊
        x = x.view(B, 4, self.out_channels, H, W)  # (B, 4, out_channels, H, W)

        # 重新排列，將 `4` 軸移動到 `W`
        x = x.permute(0, 2, 3, 1, 4)  # (B, out_channels, H, 4, W)

        # 合併 `4` 軸到 `W`
        x = x.reshape(B, self.out_channels, H, W * 4)

        


        return x




class Classifier(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifier, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        # self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, 48, stride)
        # self.conv = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Conv2d(dw_channels, num_classes, 1)
        # )
        self.inverted_strip = InvertedStrip(48, 12)

        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(12, num_classes, 1)  # 最終輸出到 num_classes
        )

    def forward(self, x):
        x = self.dsconv1(x)
        # print("we're in classifier1")
        # print(x.shape)
        
        x = self.dsconv2(x)
        # print("we're in classifier2")
        
        x = self.inverted_strip(x)  
        # print("after InvertedStrip")
        # print(x.shape)

        x = self.conv(x)
        return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data_loader import datasets
    model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 512)
    model = get_fast_scnn('citys')
    outputs = model(img)
