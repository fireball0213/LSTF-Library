import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.seq_norm = configs.seq_norm
        self.seq_len=configs.seq_len
        self.final_channels = configs.c_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
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

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        if self.seq_norm == 'Diff':
            # 将差分操作的影响逆转，恢复到原始数据的相对尺度
            enc_out = enc_out + seq_last
        elif self.seq_norm == 'RevIN':
            enc_out = enc_out * stdev + means
        else:
            pass
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
