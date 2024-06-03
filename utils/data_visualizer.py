import numpy as np
import matplotlib.pyplot as plt
import random
import random
import torch
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# def plot_dimension_mape(args,all_dimension_mape, param_name, param_values, DAY):
#     plt.figure(figsize=(16, 5))
#     for i, (dimension_mape, value) in enumerate(zip(all_dimension_mape, param_values)):
#         plt.plot(dimension_mape, linestyle='-', label=args.model + '+' +param_name+'='+ str(int(value)))
#     # plt.title("Ensemble Forecast Across Different Periods")
#     plt.title(args.decomp_method + '+kernel_size=' + str(int(args.kernel_size)) + '+resid=' + str(args.resid))
#     plt.xlim([0, args.pred_len])  # 设置x轴的范围
#     xticks_positions = np.arange(0, args.pred_len + 1, DAY // 2)  # 每天的位置，从0到pred_len
#     xticks_labels = [str(int(x / DAY)) for x in xticks_positions]  # 将刻度转换为“天”单位
#     plt.xticks(xticks_positions, xticks_labels,fontsize=14)  # 设置x轴的刻度和标签
#     plt.xlabel("Pred_len(单位：天)",fontsize=16)
#     plt.ylabel("MAPE",fontsize=16)
#     plt.grid(True)
#     plt.legend(fontsize=18)
#     plt.show()
# def plot_date_mape(args,all_date_mape, param_name, param_values, DAY):
#     plt.figure(figsize=(16, 5))
#     for i, (date_mape, value) in enumerate(zip(all_date_mape, param_values)):
#         plt.plot(date_mape, linestyle='-', label=args.model + '+' + param_name + '=' + str(int(value)))
#     # plt.title("Ensemble Forecast Across Different Periods")
#     plt.title(args.decomp_method + '+kernel_size=' + str(int(args.kernel_size)) + '+resid=' + str(args.resid))
#     # plt.xlim([0, args.pred_len])  # 设置x轴的范围
#     # xticks_positions = np.arange(0, args.pred_len + 1, DAY // 2)  # 每天的位置，从0到pred_len
#     # xticks_labels = [str(int(x / DAY)) for x in xticks_positions]  # 将刻度转换为“天”单位
#     # plt.xticks(xticks_positions, xticks_labels,fontsize=14)  # 设置x轴的刻度和标签
#     plt.xlabel("预测稳定性(15min粒度)",fontsize=12)
#     plt.ylabel("MAPE",fontsize=16)
#     plt.grid(True)
#     plt.legend(fontsize=18)
#     plt.show()

def plot_combined_mape(args, all_dimension_mape, all_date_mape, param_name, param_values, DAY):
    plt.figure(figsize=(16, 10))

    # 上半部分子图：dimension_mape
    plt.subplot(2, 1, 1)
    for i, (dimension_mape, value) in enumerate(zip(all_dimension_mape, param_values)):
        plt.plot(dimension_mape, linestyle='-', label=args.model + '+' + param_name + '=' + str(int(value)))
    plt.title(args.decomp_method + '+kernel_size=' + str(int(args.kernel_size)) + '+resid=' + str(args.resid))
    plt.xlim([0, args.pred_len])  # 设置x轴的范围
    xticks_positions = np.arange(0, args.pred_len + 1, DAY // 2)  # 每天的位置，从0到pred_len
    xticks_labels = [str(int(x / DAY)) for x in xticks_positions]  # 将刻度转换为“天”单位
    plt.xticks(xticks_positions, xticks_labels, fontsize=14)  # 设置x轴的刻度和标签
    plt.xlabel("Pred_len维度(单位：天)", fontsize=16)
    plt.ylabel("MAPE", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=18)

    # 下半部分子图：date_mape
    plt.subplot(2, 1, 2)
    for i, (date_mape, value) in enumerate(zip(all_date_mape, param_values)):
        plt.plot(date_mape, linestyle='-', label=args.model + '+' + param_name + '=' + str(int(value)))
    plt.xlabel("样本数量维度(15min粒度)", fontsize=12)
    plt.ylabel("MAPE", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=18)

    plt.tight_layout()
    plt.show()