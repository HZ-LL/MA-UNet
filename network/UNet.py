import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from thop import profile

class DoubleConv(nn.Module):#卷积类
    def __init__(self, in_channels, out_channels):#固定方法
        super(DoubleConv, self).__init__()#继承DoubleConv类
        self.conv = nn.Sequential(
            #二维卷积
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #标准化
            nn.BatchNorm2d(out_channels),
            #激活层
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    #网络推进
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):#UNet主体
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        #声明list用于上采样和下采样存储
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        #池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            #下采样
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            #上采样--包括一个卷积和一个转置卷积
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        #unet网络底层卷积
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in range(0, len(self.downs)):
            #对x进行下采样
            x = self.downs[down](x)
            #将此处状态加入跳跃连接list
            skip_connections.append(x)
            #进行池化操作
            x = self.pool(x)

        x = self.bottleneck(x)
        #因为上采样是自下而上，所以反转当前列表
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            #先进行转置卷积
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            #执行跳跃连接部分
            concat_skip = torch.cat((skip_connection, x), dim=1)
            #粘贴后的两个特征在进行一次卷积操作
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)  # 最后的1*1卷积操作


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=1)
    #将x传入模型
    preds = model(x)
    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops}, Params: {params}")
    print(preds.shape)