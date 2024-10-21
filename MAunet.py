import torch
import torch.nn as nn
#from thop import profile
from Dynamic import Dynamic_conv2d
from selfattention import SSA
from HFOM import HFOM, Pooling, ChannelAttention, SpatialAttention
from DilaDown import DilaConv

class downDynamicCnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downDynamicCnn, self).__init__()
        self.dyconv3 = nn.Sequential(
            Dynamic_conv2d(in_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(out_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    # 网络推进
    def forward(self, x):
        return self.dyconv3(x) + self.conv5(x)

class upDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upDoubleConv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SpatialAttention(),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.br = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # 网络推进
    def forward(self, x):
        y = self.conv3(x)
        z = self.conv5(x)
        y = torch.cat([y, z], dim=1)
        y = self.br(y)
        return y

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

class MyNet(nn.Module):  # UNet主体
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(MyNet, self).__init__()
        # 声明list用于上采样和下采样存储
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sk = nn.ModuleList()
        self.br = nn.ModuleList()
        self.pl = nn.ModuleList()
        self.temp = nn.ModuleList()
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            # 下采样
            if feature <= 128:
                self.downs.append(downDynamicCnn(in_channels, feature))
                in_channels = feature
            else:
                self.downs.append(DilaConv(in_channels, feature))
                in_channels = feature

        for feature in reversed(features):
            # 上采样--包括一个卷积和一个转置卷积
            self.sk.append(HFOM(feature))
            self.pl.append(Pooling(feature))
            self.temp.append(nn.Sequential(
                nn.Conv2d(feature, feature, 3, 1, 1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
            ))
            self.br.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, 3, 1, 1, bias = False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(upDoubleConv(feature * 2, feature))
        # unet网络底层卷积
        self.shuffleconv = nn.Sequential(
            GroupConvShuffle(features[-1]),
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=1)
        )
        # num_block非固定，可酌情设置
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
        temp = x
        x = self.sa(x)
        temp_sa = x
        skip_connections = skip_connections[::-1]
        # 底层groupshuffle+SA_feature
        temp = torch.add(temp, temp_sa)
        for j in range(len(skip_connections)):
            # 转置卷积
            temp = self.ups[j*2](temp)
            pl= self.pl[j](skip_connections[j])
            # groupshrffle和下采样相加
            temp = torch.add(skip_connections[j], temp)
            temp = self.temp[j](temp)
            # 施加混合注意力
            temp1 = self.sk[j](temp)
            temp1 = self.br[j](torch.cat([temp1, pl], dim = 1))
            skip_connections[j] = temp1

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
    model = MyNet(in_channels=3, out_channels=1).to(device)
    # 将x传入模型
    preds = model(x).to(device)
    # flops, params = profile(model, inputs=(x,), verbose=False)
    # print(f"FLOPs: {flops}, Params: {params}")
    print(x.shape)
    print(preds.shape)