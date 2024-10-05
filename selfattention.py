import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        #头数必须为维度数整除
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
            nn.Conv2d(in_channels, embed_dim,1),
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

        #multi-head attention
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.temperature
        attention_scores = attention_scores - attention_scores.amax(dim=-1,keepdim=True)
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
            x = ssa(x) + x
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(3, 32, 64, 64).to(device)
    model = SSA(32, 128, 8, 2).to(device)
    out = model(x)
    print(out.shape)