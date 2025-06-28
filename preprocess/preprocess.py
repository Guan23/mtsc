# _*_ encoding: utf-8 _*_
# 文件: preprocess
# 时间: 2025/6/28_14:09
# 作者: GuanXK

# system
import os

# third_party

# custom


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def generate_time_series(n=365, seasonal_period=7, trend_strength=0.5, noise_level=0.1):
    """生成带趋势和季节性的示例时序数据"""
    # 创建日期索引
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    # 生成季节性成分 (使用正弦函数)
    seasonality = np.sin(np.arange(n) * 2 * np.pi / seasonal_period)

    # 生成趋势成分
    trend = np.linspace(0, trend_strength * n, n)

    # 生成随机噪声
    noise = np.random.normal(0, noise_level, n)

    # 组合各成分
    values = trend + seasonality + noise

    return pd.Series(values, index=dates, name="value")


def introduce_missing_values(series, missing_ratio=0.1, missing_pattern="random"):
    """在时序数据中引入缺失值"""
    mask = np.zeros(len(series), dtype=bool)

    if missing_pattern == "random":
        # 随机缺失
        mask[np.random.choice(len(series), size=int(len(series) * missing_ratio), replace=False)] = True
    elif missing_pattern == "chunk":
        # 连续缺失 (块状)
        chunk_size = int(len(series) * missing_ratio)
        start_idx = np.random.randint(0, len(series) - chunk_size)
        mask[start_idx:start_idx + chunk_size] = True

    series_with_nan = series.copy()
    series_with_nan[mask] = np.nan
    return series_with_nan


def introduce_outliers(series, outlier_ratio=0.05, outlier_strength=3.0):
    """在时序数据中引入异常值"""
    series_with_outliers = series.copy()
    n_outliers = int(len(series) * outlier_ratio)

    # 随机选择位置添加异常值
    outlier_indices = np.random.choice(len(series), size=n_outliers, replace=False)

    for idx in outlier_indices:
        # 随机决定异常值是向上还是向下偏移
        direction = np.random.choice([-1, 1])
        # 添加基于标准差的异常值
        series_with_outliers.iloc[idx] += direction * outlier_strength * series.std()

    return series_with_outliers


def handle_missing_values(series, method="ffill", seasonal_period=None):
    """处理时序数据中的缺失值"""
    if method == "drop":
        return series.dropna()

    elif method == "mean":
        return series.fillna(series.mean())

    elif method == "median":
        return series.fillna(series.median())

    elif method == "ffill":  # 前向填充
        return series.fillna(method="ffill")

    elif method == "bfill":  # 后向填充
        return series.fillna(method="bfill")

    elif method == "interpolate_linear":  # 线性插值
        return series.interpolate(method="linear")

    elif method == "interpolate_time":  # 基于时间的插值
        return series.interpolate(method="time")

    elif method == "seasonal":  # 季节性填充
        if seasonal_period is None:
            raise ValueError("季节性填充需要指定seasonal_period参数")

        # 计算季节性均值
        seasonal_means = series.groupby(series.index.dayofweek).mean()

        # 对每个缺失值，使用对应季节的均值填充
        filled_series = series.copy()
        for idx, value in filled_series.items():
            if pd.isna(value):
                day_of_week = idx.dayofweek
                filled_series[idx] = seasonal_means[day_of_week]

        return filled_series

    else:
        raise ValueError(f"不支持的填充方法: {method}")


def detect_outliers_iqr(series, window=20, threshold=1.5):
    """使用IQR方法检测异常值"""
    outliers = pd.Series(False, index=series.index)

    # 对每个数据点，计算其前后window大小的窗口内的IQR
    for i in range(len(series)):
        # 确定窗口边界
        start = max(0, i - window)
        end = min(len(series), i + window + 1)

        # 排除当前点，避免自引用
        window_data = series.iloc[start:end].drop(series.index[i])

        # 计算IQR
        q1 = window_data.quantile(0.25)
        q3 = window_data.quantile(0.75)
        iqr = q3 - q1

        # 判断是否为异常值
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        if series.iloc[i] < lower_bound or series.iloc[i] > upper_bound:
            outliers.iloc[i] = True

    return outliers


def detect_outliers_zscore(series, window=20, threshold=3.0):
    """使用Z-score方法检测异常值"""
    outliers = pd.Series(False, index=series.index)

    for i in range(len(series)):
        # 确定窗口边界
        start = max(0, i - window)
        end = min(len(series), i + window + 1)

        # 排除当前点
        window_data = series.iloc[start:end].drop(series.index[i])

        # 计算Z-score
        mean = window_data.mean()
        std = window_data.std()

        if std == 0:  # 防止除以零
            continue

        z_score = (series.iloc[i] - mean) / std

        if abs(z_score) > threshold:
            outliers.iloc[i] = True

    return outliers


def handle_outliers(series, outliers_mask, method="clip", window=20):
    """处理检测到的异常值"""
    cleaned_series = series.copy()

    if method == "remove":
        # 删除异常值（设置为NaN）
        cleaned_series[outliers_mask] = np.nan
        return cleaned_series

    elif method == "clip":
        # 替换为上下限
        for i in range(len(cleaned_series)):
            if outliers_mask.iloc[i]:
                # 确定窗口边界
                start = max(0, i - window)
                end = min(len(cleaned_series), i + window + 1)

                # 排除当前点
                window_data = cleaned_series.iloc[start:end].drop(cleaned_series.index[i])

                # 计算上下限
                q1 = window_data.quantile(0.25)
                q3 = window_data.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # 裁剪异常值
                if cleaned_series.iloc[i] < lower_bound:
                    cleaned_series.iloc[i] = lower_bound
                elif cleaned_series.iloc[i] > upper_bound:
                    cleaned_series.iloc[i] = upper_bound

        return cleaned_series

    elif method == "interpolate":
        # 用插值替换异常值
        cleaned_series[outliers_mask] = np.nan
        return cleaned_series.interpolate(method="time")

    else:
        raise ValueError(f"不支持的异常值处理方法: {method}")


def visualize_results(original, with_missing, filled, with_outliers, cleaned, title="时序数据清洗结果"):
    """可视化数据清洗结果"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # 原始数据
    axes[0].plot(original, label="原始数据", color="blue")
    axes[0].set_title("原始时序数据")
    axes[0].legend()
    axes[0].grid(True)

    # 缺失值处理
    axes[1].plot(original, label="原始数据", color="blue", alpha=0.5)
    axes[1].plot(with_missing, label="含缺失值", color="red", alpha=0.5)
    axes[1].plot(filled, label="填充后", color="green")
    axes[1].set_title("缺失值处理对比")
    axes[1].legend()
    axes[1].grid(True)

    # 异常值处理
    axes[2].plot(original, label="原始数据", color="blue", alpha=0.5)
    axes[2].plot(with_outliers, label="含异常值", color="red", alpha=0.5)
    axes[2].plot(cleaned, label="处理后", color="purple")
    axes[2].set_title("异常值处理对比")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()


def evaluate_cleaning(original, cleaned):
    """评估清洗效果"""
    # 计算均方误差
    mse = ((original - cleaned) ** 2).mean()

    # 计算相关性
    corr = np.corrcoef(original.dropna(), cleaned.dropna())[0, 1]

    print(f"清洗后MSE: {mse:.4f}")
    print(f"与原始数据的相关性: {corr:.4f}")

    return {"mse": mse, "correlation": corr}


def main():
    # 1. 生成示例数据
    print("生成示例时序数据...")
    original_series = generate_time_series(
        n=365,
        seasonal_period=7,
        trend_strength=0.5,
        noise_level=0.1
    )

    # 2. 引入缺失值
    print("引入缺失值...")
    series_with_missing = introduce_missing_values(original_series, missing_ratio=0.1, missing_pattern="random")

    # 3. 处理缺失值
    print("处理缺失值...")
    filled_series = handle_missing_values(
        series_with_missing,
        method="interpolate_time",  # 可尝试其他方法: mean, median, ffill, seasonal
        seasonal_period=7
    )

    # 4. 引入异常值
    print("引入异常值...")
    series_with_outliers = introduce_outliers(filled_series, outlier_ratio=0.05, outlier_strength=3.0)

    # 5. 检测异常值
    print("检测异常值...")
    outliers_mask = detect_outliers_zscore(series_with_outliers, window=20, threshold=3.0)
    # 也可以尝试IQR方法: outliers_mask = detect_outliers_iqr(series_with_outliers, window=20, threshold=1.5)

    # 6. 处理异常值
    print("处理异常值...")
    cleaned_series = handle_outliers(
        series_with_outliers,
        outliers_mask,
        method="clip",  # 可尝试其他方法: remove, interpolate
        window=20
    )

    # 7. 可视化结果
    print("可视化数据清洗结果...")
    visualize_results(
        original_series,
        series_with_missing,
        filled_series,
        series_with_outliers,
        cleaned_series,
        title="时序数据缺失值与异常值处理示例"
    )

    # 8. 评估清洗效果
    print("\n评估清洗效果:")
    evaluate_cleaning(original_series, cleaned_series)

    # 9. 保存结果（可选）
    # result_df = pd.DataFrame({
    #     'original': original_series,
    #     'with_missing': series_with_missing,
    #     'filled': filled_series,
    #     'with_outliers': series_with_outliers,
    #     'cleaned': cleaned_series
    # })
    # result_df.to_csv('time_series_cleaning_result.csv')


if __name__ == "__main__":
    print("\n-------------- start --------------\n")
    main()

    print("\n--------------- end ---------------\n")
