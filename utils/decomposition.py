import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import time
from statsmodels.tsa.seasonal import STL
import os
class SeriesDecomposition(nn.Module):
    def __init__(self, args):
        super(SeriesDecomposition, self).__init__()
        self.resid=args.resid
        self.args=args
        self.kernel_size = args.kernel_size
        self.times=args.trend_dec_times
        self.seasonal_period = args.period
        self.decomp_method = args.decomp_method
        self.device = self._acquire_device()
        self.kernel_sizes = [(self.kernel_size - 1) * (2 ** i) + 1 for i in range(self.times)]
        self.first_execution = True
        self.first_execution = False#不画图
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def forward(self, x):
        start_time = time.time()
        x = self._check_input(x)
        trend, seasonal, residual = self._decompose(x)
        x = self._check_input(x)
        if time.time() - start_time > 0.5:
            print(f"Decom: { time.time() - start_time:.4f}s")
        if self.first_execution:
            self._plot_results(x, trend, seasonal, residual,0,767)
            self.first_execution = False
        trend, seasonal, residual = self._handle_residual(trend, seasonal, residual, self.resid)
        trend=self._check_input(trend).to(self.device, non_blocking=True)
        seasonal=self._check_input(seasonal).to(self.device, non_blocking=True)
        residual=self._check_input(residual).to(self.device, non_blocking=True)
        return trend, seasonal

    def _check_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x

    def _decompose(self, x):
        raise NotImplementedError

    def _handle_residual(self, trend, seasonal, residual, resid):
        if resid == 'trend':
            trend += residual
        elif resid == 'seasonal':
            seasonal += residual
        else:
            pass
        return trend, seasonal, residual

    def _plot_results(self, x, trend, seasonal, residual,t1,t2):
        plt.figure(figsize=(16, 5))
        # 检查是否是tensor，如果是，转移到cpu，然后转为numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            trend = trend.cpu().numpy()
            season = seasonal.cpu().numpy()
            resid = residual.cpu().numpy()
        # 如果x是三维的，去掉最后一个维度
        plot_column = -1
        plt.plot(trend[0,t1:t2, plot_column], label='trend', color='red')
        plt.plot(season[0,t1:t2, plot_column], label='season', color='blue')
        plt.plot(resid[0,t1:t2, plot_column], label='resid', color='lightgreen')
        plt.plot(x[0,t1:t2, plot_column], label='original', color='grey')
        plt.title(self.decomp_method+'+kernel_size='+str(int(self.kernel_size))+'+period='+str(int(self.seasonal_period))+'+times='+str(int(self.times)))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.show()


class MovingAvgDecomposition1(SeriesDecomposition):
    def __init__(self, args):
        super(MovingAvgDecomposition1, self).__init__(args)



    def _decompose(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        trend = torch.zeros_like(x)
        x_residual = x.clone()
        for kernel_size in self.kernel_sizes:
            front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x_pad = torch.cat([front, x, end], dim=1)
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
            current_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
            trend += current_trend
            x_residual -=current_trend


        seasonal = x - trend
        residual = x - trend - seasonal
        return trend, seasonal, residual


class MovingAvgDecomposition2(SeriesDecomposition):
    def __init__(self,args):
        super(MovingAvgDecomposition2, self).__init__(args)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def _decompose(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, channels = x.shape

        # padding = self.seasonal_period // 2
        # x_padded = F.pad(x, (0, 0, padding - 1, padding), mode='reflect')
        #
        # # Moving average using unfold
        # unfolded = x_padded.unfold(1, self.seasonal_period, 1)
        # moving_avg = unfolded.mean(dim=-1)
        # trend = moving_avg[:, :seq_len, :]

        trend = torch.zeros_like(x)
        x_residual = x.clone()
        for kernel_size in self.kernel_sizes:
            front = x_residual[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x_residual[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x_pad = torch.cat([front, x_residual, end], dim=1)
            current_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
            trend += current_trend
            x_residual -= current_trend

        deseasonalized = x - trend

        # Seasonal
        seasonal = torch.zeros_like(x)
        for i in range(channels):
            deseasonalized_channel = deseasonalized[:, :, i]
            seasonal_channel = seasonal[:, :, i]
            for k in range(self.seasonal_period):
                seasonal_indices = torch.arange(k, seq_len, self.seasonal_period)
                if len(seasonal_indices) > 0:
                    seasonal_mean = deseasonalized_channel[:, seasonal_indices].mean(dim=1, keepdim=True)
                    seasonal_channel[:, seasonal_indices] = seasonal_mean
            seasonal[:, :, i] = seasonal_channel

        residual = deseasonalized - seasonal

        return trend, seasonal, residual

class DFTSeriesDecomposition(SeriesDecomposition):
    def __init__(self, args):
        super(DFTSeriesDecomposition, self).__init__(args)
        self.top_k = args.top_k

    def _decompose(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        seasonal = torch.fft.irfft(xf)
        trend = x - seasonal
        residual = torch.zeros_like(x)
        return trend, seasonal, residual

class STLDecomposition(SeriesDecomposition):
    def __init__(self, args):
        super(STLDecomposition, self).__init__(args)


    def _decompose(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        trend = np.zeros_like(x)
        seasonal = np.zeros_like(x)
        residual = np.zeros_like(x)
        # result = STL(x, period=self.seasonal_period).fit()
        batch_size, seq_len, channels = x.shape[0], x.shape[1], x.shape[2]
        for j in range(batch_size):
            for i in range(channels):
                result = STL(x[j, :, i], period=self.seasonal_period).fit()
                trend[j, :, i] = result.trend
                seasonal[j, :, i] = result.seasonal
                residual[j, :, i] = result.resid

        return trend, seasonal, residual

def decomposition_method(method_name, args):
    if method_name == 'MA1':
        return MovingAvgDecomposition1(args)
    elif method_name == 'MA2':
        return MovingAvgDecomposition2(args)
    elif method_name == 'DFT':
        return DFTSeriesDecomposition(args)
    elif method_name == 'STL':
        return STLDecomposition(args)
    else:
        raise ValueError(f"Unknown decomposition method: {method_name}")

# 使用示例
if __name__ == '__main__':
    x = torch.randn(100)  # 示例输入数据
    method_name = 'dft'  # 可选 'moving_avg1', 'moving_avg2', 'dft', 'stl'
    kwargs = {'top_k': 5}  # 对应方法的参数
    decomp_method = decomposition_method(method_name, **kwargs)
    trend, seasonal, residual = decomp_method(x, resid='trend')
