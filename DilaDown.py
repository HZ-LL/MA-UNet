import torch
import torch.nn as nn
from HFOM import ChannelAttention, SpatialAttention

class DilaConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilaConv, self).__init__()
        # 实例化通道注意力机制
        self.channel_attention = ChannelAttention(out_channels)
        # 实例化空间注意力机制
        self.spatial_attention = SpatialAttention()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False)
        self.conv3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3, bias=False)
        self.conv3_5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.br = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.spatial_attention(self.conv3_1(x))
        z = self.channel_attention(self.conv3_3(torch.add(x, y)))
        x = self.spatial_attention(self.conv3_5(torch.add(x, z)))
        x = torch.add(x, y)
        x = torch.add(x, z)
        return self.br(x)

if __name__ == '__main__':
    x = torch.randn(1, 128, 256, 256)
    model = DilaConv(128,256)
    out = model(x)
    print(out.shape)