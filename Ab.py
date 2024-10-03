import torch
import torch.nn as nn
from thop import profile
from selfattention import SSA



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # 网络推进
    def forward(self, x):
        return self.conv3(x)

class GroupConvShuffle(nn.Module):
    def __init__(self, in_channels):
        super(GroupConvShuffle, self).__init__()
        self.groups = in_channels
        self.group_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                            groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def shuffle_channels(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, self.groups, num_channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        x = self.group_conv(x)
        x = self.shuffle_channels(x)
        return x

class MFAA(nn.Module):  # UNet主体
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(MFAA, self).__init__()
        # 声明list用于上采样和下采样存储
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sk = nn.ModuleList()
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            # 下采样
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            # 上采样--包括一个卷积和一个转置卷积
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        # unet网络底层卷积
        self.shuffleconv = nn.Sequential(
            GroupConvShuffle(features[-1]),
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=1)
        )
        self.sa = SSA(features[-1] * 2, features[-1] * 3, num_heads=8, num_blocks=1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for i in range(len(self.downs)):
            x = self.downs[i](x)
            # 将此处状态加入跳跃连接list
            skip_connections.append(x)
            # 进行池化操作
            x = self.pool(x)

        x = self.shuffleconv(x)
        x = self.sa(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            # 先进行转置卷积
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 256, 256).to(device)
    model = MFAA(in_channels=3, out_channels=1).to(device)
    # 将x传入模型
    preds = model(x).to(device)
    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops}, Params: {params}")
    print(x.shape)
    print(preds.shape)