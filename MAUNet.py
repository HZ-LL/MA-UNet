import torch
import torch.nn as nn
import torch.nn.functional as F

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

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
        return self.sigmoid(out)*x

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
        return self.sigmoid(out)*x

class Pooling(nn.Sequential):
    def __init__(self, in_channels):
        super(Pooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            # 1
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            # 2
            # nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(Pooling, self).forward(x)
        # 上采样
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x
        # return self.conv(x)

class AT(nn.Module):
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

# 参考https://github.com/kaijieshi7/Dynamic-convolution-Pytorch
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, ratio=0.25,
                 dilation=1, bias=False, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes,
                                                                    self.in_planes//self.groups, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class FEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEM, self).__init__()
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.spatial_attention(self.conv3_1(x))
        z = self.channel_attention(self.conv3_3(torch.add(x, y)))
        x = self.spatial_attention(self.conv3_5(torch.add(x, z)))
        x = torch.add(x, y)
        x = torch.add(x, z)
        return self.br(x)

class REM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(REM, self).__init__()
        self.dyconv3 = nn.Sequential(
            Dynamic_conv2d(in_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(out_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
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
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
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

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        # 头数必须为维度数整除
        assert embed_dim % num_heads == 0, "Embed dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reshape = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            # nn.Dropout(p=0.1)
        )

        self.query = nn.Conv2d(in_channels, embed_dim, 1)
        self.key = nn.Conv2d(in_channels, embed_dim, 1)
        self.value = nn.Conv2d(in_channels, embed_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, C, width, height = x.size()
        seq_len = width * height
        x_normalized = self.bn(x)
        x_activated = self.relu(x_normalized)

        query = self.query(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(2, 3)
        # query 的形状: [batch_size, num_heads, seq_len, head_dim]
        key = self.key(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(3, 2)
        # key 的原始形状: [batch_size, num_heads, head_dim, seq_len]
        # key 转置后的形状: [batch_size, num_heads, seq_len, head_dim]，与 query 匹配
        value = self.value(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(2, 3)
        # value 的形状: [batch_size, num_heads, seq_len, head_dim]

        # multi-head attention
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.temperature
        attention_scores = attention_scores - attention_scores.amax(dim=-1, keepdim=True)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        out = out.transpose(2, 3).contiguous().view(batch_size, -1, width, height)
        out = torch.add(self.gamma * out, self.conv(x))
        return self.reshape(out)

class SSA(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_blocks):
        super(SSA, self).__init__()
        self.ssa_blocks = nn.ModuleList([
            MultiHeadSelfAttentionBlock(in_channels, embed_dim, num_heads) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for ssa in self.ssa_blocks:
            x = ssa(x)
        return x

class MA(nn.Module):  # UNet主体
    def __init__(self, in_channels, out_channels):
        super(MA, self).__init__()
        features = [64, 128, 256, 512]
        # 声明list用于上采样和下采样存储
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sk = nn.ModuleList()
        self.br = nn.ModuleList()
#        self.pl = nn.ModuleList()
#        self.temp = nn.ModuleList()
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            # 下采样
            if feature <= 128:
                self.downs.append(REM(in_channels, feature))
                in_channels = feature
            else:
                self.downs.append(FEM(in_channels, feature))
                in_channels = feature

        for feature in reversed(features):
            # 上采样--包括一个卷积和一个转置卷积
            self.sk.append(AT(feature))

            self.br.append(nn.Sequential(
                nn.Conv2d(feature, feature, 1, bias = False),
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
        self.sa = SSA(features[-1] * 2, features[-1] * 3, num_heads=8, num_blocks=1)
#        self.sa = MultiHeadSelfAttentionBlock(features[-1] * 2, features[-1] * 3, num_heads=8)
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
        # groupshuffle+SA_feature
        temp = torch.add(temp, temp_sa)
        for j in range(len(skip_connections)):
            temp = self.ups[j*2](temp)
            temp = torch.add(skip_connections[j], temp)
            temp1 = self.sk[j](temp)
            skip_connections[j] = self.br[j](temp1)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 256, 256).to(device)
    model = MA(in_channels=3, out_channels=1).to(device)
    preds = model(x).to(device)
    print(x.shape)
    print(preds.shape)
