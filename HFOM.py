import torch.nn as nn
import torch.nn.functional as F
import torch


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上平均池化 [b,1,h,w]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        out = torch.cat([avg_out, max_out], dim=1)
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        out = self.conv1(out)
        return self.sigmoid(out) * x + x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        # 最大池化和全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.temperature = nn.Parameter(torch.tensor(1.0))

        # 利用1x1卷积代替全连接，1*1卷积效果比全连接更好，参考ECA机制
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  # 返回权重矩阵

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        out = avg_out + max_out
        # return self.Sigmoid(out/self.temperature)#是否加入温度系数
        return self.sigmoid(out) * x


# 池化 -> 1*1 卷积 -> 上采样
class Pooling(nn.Sequential):
    def __init__(self, in_channels):
        super(Pooling, self).__init__(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应均值池化
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            # 调试
            # nn.Dropout(p=0.2),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(Pooling, self).forward(x)
        # 上采样
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class HFOM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        # 实例化通道注意力机制
        self.channel_attention = ChannelAttention(in_planes)
        # 实例化空间注意力机制
        self.spatial_attention = SpatialAttention()

    # 前向传播
    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        y = ca * sa
        x = torch.add(ca, sa)
        x = torch.add(x, y)
        return x


if __name__ == '__main__':
    x = torch.randn((3, 256, 161, 161))
    model = HFOM(256)
    preds = model(x)
    print(preds.shape)
