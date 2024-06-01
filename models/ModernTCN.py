from torch import optim
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
# from models.DLbase import DLForecastModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# B：batch size
# channels：多变量序列的变量数
# seq_len：过去序列的长度
# pred_len: 预测序列的长度
# N: 分Patch后Patch的个数
# D：每个变量的通道数
# P：kernel size of embedding layer
# S：stride of embedding layer

class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, channels, seq_len]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, channels, seq_len] -> [B, channels, 1, seq_len]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, channels, 1, seq_len] -> [B*channels, 1, seq_len]
        x_pad = F.pad(
            x,
            pad=(0, self.P - self.S),
            mode='replicate'
        )  # [B*channels, 1, seq_len] -> [B*channels, 1, seq_len+P-S]

        x_emb = self.conv(x_pad)  # [B*channels, 1, seq_len+P-S] -> [B*channels, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*channels, D, N] -> [B, channels, D, N]

        return x_emb  # x_emb: [B, channels, D, N]


class ConvFFN(nn.Module):
    def __init__(self, channels, D, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = channels if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=channels * D,
            out_channels=r * channels * D,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r * channels * D,
            out_channels=channels * D,
            kernel_size=1,
            groups=groups_num
        )

    def forward(self, x):
        # x: [B, channels*D, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, channels*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, channels, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=channels * D,
            out_channels=channels * D,
            kernel_size=kernel_size,
            groups=channels * D,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(channels * D)
        self.conv_ffn1 = ConvFFN(channels, D, r, one=True)
        self.conv_ffn2 = ConvFFN(channels, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, channels, D, N]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')  # [B, channels, D, N] -> [B, channels*D, N]
        x = self.dw_conv(x)  # [B, channels*D, N] -> [B, channels*D, N]
        x = self.bn(x)  # [B, channels*D, N] -> [B, channels*D, N]
        x = self.conv_ffn1(x)  # [B, channels*D, N] -> [B, channels*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)  # [B, channels*D, N] -> [B, channels, D, N]
        x = x.permute(0, 2, 1, 3)  # [B, channels, D, N] -> [B, D, channels, N]
        x = rearrange(x, 'b d m n -> b (d m) n')  # [B, D, channels, N] -> [B, D*channels, N]

        x = self.conv_ffn2(x)  # [B, D*channels, N] -> [B, D*channels, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)  # [B, D*channels, N] -> [B, D, channels, N]
        x = x.permute(0, 2, 1, 3)  # [B, D, channels, N] -> [B, channels, D, N]

        out = x + x_emb

        return out  # out: [B, channels, D, N]

class ModernTCN():
    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = Model(args).to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def forward(self, x_enc):
        # Forward pass through the entire model
        return self.model(x_enc)
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seq_norm = configs.seq_norm
        self.final_channels = configs.c_out

        # 初始化参数
        D = configs.d_model  # Embedding dimension
        P = 16#configs.patch_len    # Kernel size for embedding
        S = 8#configs.stride     # Stride for embedding
        r = 1#configs.r     # Expansion ratio for ConvFFN
        kernel_size = 51#configs.kernel_size  # Kernel size for TCN block
        num_layers = 2#configs.num_layers   # Number of TCN blocks

        N = self.seq_len // S  # Number of patches

        # 嵌入层
        self.embed_layer = Embedding(P, S, D)

        # TCN Blocks
        self.tcn_blocks = nn.Sequential(*[
            ModernTCNBlock(self.final_channels, D, kernel_size, r) for _ in range(num_layers)
        ])

        # Prediction Head
        self.head = nn.Linear(D * N, self.pred_len*self.final_channels)

    def forward(self,  x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.seq_norm == 'Diff':
            # 提取最后一个时间步的数据并进行差分操作
            last_index = self.seq_len - 1
            seq_last = x_enc[:, last_index, :].detach()
            seq_last = seq_last.reshape(x_enc.size(0), 1, self.final_channels)
            x_enc = x_enc - seq_last
        elif self.seq_norm == 'RevIN':
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            pass

        x_enc = x_enc.permute(0, 2, 1)  # x_enc: [B, channels, seq_len]
        x_emb = self.embed_layer(x_enc)  # [B, channels, seq_len] -> [B, channels, D, N]

        # for i in range(self.num_layers):
        #     x_emb = self.model[i](x_emb)  # [B, channels, D, N] -> [B, channels, D, N]
        # TCN Blocks
        x_tcn = self.tcn_blocks(x_emb)

        # Flatten
        z = rearrange(x_tcn, 'b m d n -> b m (d n)')  # [B, channels, D, N] -> [B, channels, D*N]
        dec_out = self.head(z)  # [B, channels, D*N] -> [B, channels, pred_len]
        dec_out = dec_out.permute(0, 2, 1)

        if self.seq_norm == 'Diff':
            # 将差分操作的影响逆转，恢复到原始数据的相对尺度
            dec_out = dec_out + seq_last
        elif self.seq_norm == 'RevIN':
            dec_out = dec_out * stdev + means
        else:
            pass

        return dec_out





if __name__ == '__main__':
    class Args:
        def __init__(self, seq_len, pred_len, channels):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.channels = channels


    # 假设序列长度为96，预测长度为192，变量数（通道数）为4
    args = Args(seq_len=96, pred_len=192, channels=4)


    past_series = torch.rand(2, 4, 96)
    print(past_series)
    model = ModernTCN( args)
    pred_series = model(past_series)
    print(pred_series.shape)
    # torch.Size([2, 4, 192])