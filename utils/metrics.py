import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def mape_dimension(predict, target):
    # 先找到所有非零的位置
    non_zero = target != 0
    # 初始化 mape_value 数组
    mape_value = np.zeros_like(target)
    # 只在非零位置上计算 MAPE
    mape_value[non_zero] = np.abs((target[non_zero] - predict[non_zero]) / target[non_zero])
    # 计算指定维度上的所有MAPE值
    dimension_mape = np.mean(mape_value, axis=0)

    date_mape = np.mean(mape_value, axis=1)
    return np.mean(mape_value),dimension_mape,date_mape

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mape, dimension_mape, date_mape= mape_dimension(pred, true)

    return mae, mse, rmse, mape, mspe, dimension_mape, date_mape
