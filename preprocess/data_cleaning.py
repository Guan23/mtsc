# _*_ encoding: utf-8 _*_
# 文件: data_cleaning
# 时间: 2025/7/15_15:12
# 作者: GuanXK

# system
import os
from typing import Union, Optional, List, Dict, Any

# third_party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import interpolate

# custom


# 设置全局字体为 SimHei（黑体）
rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei' 等其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 正确显示负号


class DataCleaning:
    def __init__(
            self,
            data: Union[str, pd.DataFrame] = None,
            date_col: Optional[str] = None,
            value_cols: Optional[Union[str, List[str]]] = None,
            freq: Optional[str] = None,
    ):
        """
        时序数据清洗类

        参数:
            data: 数据输入，可以是文件路径或DataFrame
            date_col: 日期列名称
            value_col: 值列名称
            freq: 时间序列频率，如'D'(天),'H'(小时),'T'(分钟)
        """
        super(DataCleaning, self).__init__()

        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data

        self.date_col = date_col
        # 处理单变量或多变量情况
        if isinstance(value_cols, str):
            self.value_cols = [value_cols]
        else:
            self.value_cols = value_cols
        self.freq = freq
        self.original_data = self.data.copy()

        # 如果提供了日期列和值列，设置为时间序列
        # 如果提供了日期列和值列，设置为时间序列
        if date_col and value_cols:
            self.set_time_series(date_col, value_cols, freq)
        self.original_data_time_series = self.data.copy()

    def set_time_series(self, date_col: str, value_cols: Union[str, List[str]], freq: Optional[str] = None):
        """将数据设置为时间序列"""
        self.date_col = date_col
        self.freq = freq

        # 处理单变量或多变量情况
        if isinstance(value_cols, str):
            self.value_cols = [value_cols]
        else:
            self.value_cols = value_cols

        # 确保日期列是datetime类型
        self.data[date_col] = pd.to_datetime(self.data[date_col])

        # 设置索引
        self.data.set_index(date_col, inplace=True)

        # 如果提供了频率，重采样数据
        if freq:
            self.data = self.data.asfreq(freq)

        return self

    # 检测异常值
    def detect_outliers(self, method: int = 0, threshold: float = 3.0,
                        window: int = 20, contamination: float = 0.05) -> Dict[str, pd.Series]:
        """
        检测异常值（支持多列）

        参数:
            method: 检测方法 (0: Z-score, 1: 四分位距, 2: 移动窗口Z-score, 3: 孤立森林)
            threshold: Z-score阈值
            window: 移动窗口大小
            contamination: 异常值比例(仅用于孤立森林)

        返回:
            字典，键为列名，值为布尔Series指示每个数据点是否为异常值
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        result = {}
        for col in self.value_cols:
            values = self.data[col]

            if method == 0:  # Z-score
                z_score = np.abs((values - values.mean()) / values.std())
                result[col] = z_score > threshold

            elif method == 1:  # 四分位距
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                result[col] = (values < lower_bound) | (values > upper_bound)

            elif method == 2:  # 移动窗口Z-score
                rolling_mean = values.rolling(window=window, min_periods=1).mean()
                rolling_std = values.rolling(window=window, min_periods=1).std()
                rolling_z_score = np.abs((values - rolling_mean) / rolling_std)
                result[col] = rolling_z_score > threshold

            elif method == 3:  # 孤立森林
                try:
                    from sklearn.ensemble import IsolationForest
                    X = values.values.reshape(-1, 1)
                    model = IsolationForest(contamination=contamination, random_state=42)
                    predictions = model.fit_predict(X)
                    result[col] = pd.Series(predictions == -1, index=values.index)
                except ImportError:
                    print("需要安装sklearn库以使用孤立森林方法")
                    result[col] = pd.Series([False] * len(values), index=values.index)

            else:
                raise ValueError("不支持的异常检测方法")

        return result

    # 检测缺失值
    def detect_missing_values(self, method: int = 0) -> Dict[str, pd.Series]:
        """
        检测缺失值（支持多列）

        参数:
            method: 检测方法 (0: 检测NaN, 1: 检测NaN和0)

        返回:
            字典，键为列名，值为布尔Series指示每个数据点是否缺失
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        result = {}
        for col in self.value_cols:
            if method == 0:
                result[col] = self.data[col].isna()
            elif method == 1:
                result[col] = (self.data[col].isna()) | (self.data[col] == 0)
            else:
                raise ValueError("不支持的缺失值检测方法")

        return result

    # 处理异常值
    def handle_outliers(self, method: int = 0, detect_method: int = 0, threshold: float = 3.0,
                        window: int = 20, replace_with: str = 'nan') -> 'DataCleaning':
        """
        处理异常值（支持多列）

        参数:
            method: 处理方法 (0: 替换为NaN, 1: 替换为上下限, 2: 插值)
            threshold: Z-score阈值
            window: 移动窗口大小
            replace_with: 替换值 ('nan', 'mean', 'median', 'interpolation')

        返回:
            self
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        outliers_dict = self.detect_outliers(method=detect_method, threshold=threshold, window=window)

        for col in self.value_cols:
            outliers = outliers_dict[col]

            if method == 0:  # 替换为NaN
                self.data.loc[outliers, col] = np.nan

            elif method == 1:  # 替换为上下限
                values = self.data[col]
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                self.data.loc[outliers & (values < lower_bound), col] = lower_bound
                self.data.loc[outliers & (values > upper_bound), col] = upper_bound

            elif method == 2:  # 插值
                self.data.loc[outliers, col] = np.nan
                # 使用线性插值处理NaN
                self.data[col] = self.data[col].interpolate(method='linear')

            else:
                raise ValueError("不支持的异常值处理方法")

        return self

    # 处理缺失值
    def handle_missing_values(self, method: int = 0, window: Optional[int] = None) -> 'DataCleaning':
        """
        处理缺失值（支持多列）

        参数:
            method: 处理方法 (0: 移动平均, 1: 线性插值, 2: 样条插值, 3: 季节性分解插值)
            window: 移动平均窗口大小

        返回:
            self
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        if window is None:
            # 根据序列长度自动确定窗口大小
            n = len(self.data)
            window = max(3, min(20, n // 10))

        for col in self.value_cols:
            if method == 0:  # 移动平均
                self.data[col] = self.data[col].fillna(
                    self.data[col].rolling(window=window, min_periods=1).mean()
                )

            elif method == 1:  # 线性插值
                self.data[col] = self.data[col].interpolate(method='linear')

            elif method == 2:  # 样条插值
                try:
                    # 确保有足够的非NaN值进行样条插值
                    if self.data[col].count() > 3:
                        index = self.data.index
                        x = np.arange(len(index))
                        y = self.data[col].values

                        mask = ~np.isnan(y)
                        x_fit = x[mask]
                        y_fit = y[mask]

                        tck = interpolate.splrep(x_fit, y_fit, s=0)
                        y_spline = interpolate.splev(x, tck, der=0)

                        self.data[col] = y_spline
                    else:
                        # 如果非NaN值不足，使用线性插值
                        self.data[col] = self.data[col].interpolate(method='linear')
                except Exception as e:
                    print(f"样条插值失败: {e}，使用线性插值")
                    self.data[col] = self.data[col].interpolate(method='linear')

            elif method == 3:  # 季节性分解插值
                try:
                    # 尝试季节性分解
                    decomposition = seasonal_decompose(self.data[col], model='additive', period=7)
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid

                    # 填充缺失值
                    trend_filled = trend.interpolate(method='linear')
                    seasonal_filled = seasonal
                    residual_filled = residual.interpolate(method='linear')

                    # 重建序列
                    self.data[col] = trend_filled + seasonal_filled + residual_filled
                except Exception as e:
                    print(f"季节性分解失败: {e}，使用线性插值")
                    self.data[col] = self.data[col].interpolate(method='linear')

            else:
                raise ValueError("不支持的缺失值处理方法")

        return self

    # 下采样
    def downsample(self, rule: str = 'D', method: int = 0) -> 'DataCleaning':
        """
        下采样，降低时间戳的频率（支持多列）

        参数:
            rule: 下采样规则 ('D': 天, 'W': 周, 'M': 月, 'Q': 季, 'Y': 年)
            method: 聚合方法 (0: 均值, 1: 总和, 2: 最大值, 3: 最小值)

        返回:
            self
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        for col in self.value_cols:
            if method == 0:  # 均值
                self.data[col] = self.data[col].resample(rule).mean()
            elif method == 1:  # 总和
                self.data[col] = self.data[col].resample(rule).sum()
            elif method == 2:  # 最大值
                self.data[col] = self.data[col].resample(rule).max()
            elif method == 3:  # 最小值
                self.data[col] = self.data[col].resample(rule).min()
            else:
                raise ValueError("不支持的下采样方法")

        return self

    # 上采样
    def upsample(self, rule: str = 'H', method: int = 0) -> 'DataCleaning':
        """
        上采样，提高时间戳的频率（支持多列）

        参数:
            rule: 上采样规则 ('H': 小时, 'T': 分钟, 'S': 秒)
            method: 插值方法 (0: 向前填充, 1: 线性插值, 2: 二次插值)

        返回:
            self
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        # 先重采样所有列
        self.data = self.data.asfreq(rule)

        for col in self.value_cols:
            if method == 0:  # 向前填充
                self.data[col] = self.data[col].ffill()
            elif method == 1:  # 线性插值
                self.data[col] = self.data[col].interpolate(method='linear')
            elif method == 2:  # 二次插值
                self.data[col] = self.data[col].interpolate(method='quadratic')
            else:
                raise ValueError("不支持的上采样方法")

        return self

    # 数据平滑
    def smoothing(self, method: int = 0, window: int = 3, alpha: float = 0.2,
                  frac: float = 0.1) -> 'DataCleaning':
        """
        数据平滑（支持多列）

        参数:
            method: 平滑方法 (0: 移动平均, 1: 指数平滑, 2: LOESS局部回归)
            window: 移动平均窗口大小
            alpha: 指数平滑权重(0-1)
            frac: LOESS采样比例(0-1)

        返回:
            self
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        for col in self.value_cols:
            if method == 0:  # 移动平均
                self.data[f'{col}_smoothed'] = self.data[col].rolling(
                    window=window, center=True).mean()
                self.data[col] = self.data[f'{col}_smoothed'].fillna(
                    self.data[col])
                self.data.drop(columns=[f'{col}_smoothed'], inplace=True)

            elif method == 1:  # 指数平滑
                self.data[f'{col}_smoothed'] = self.data[col].ewm(
                    alpha=alpha, adjust=False).mean()
                self.data[col] = self.data[f'{col}_smoothed']
                self.data.drop(columns=[f'{col}_smoothed'], inplace=True)

            elif method == 2:  # LOESS
                try:
                    x = np.arange(len(self.data))
                    y = self.data[col].values
                    smoothed = lowess(y, x, frac=frac, it=0, return_sorted=False)
                    self.data[f'{col}_smoothed'] = smoothed
                    self.data[col] = self.data[f'{col}_smoothed']
                    self.data.drop(columns=[f'{col}_smoothed'], inplace=True)
                except Exception as e:
                    print(f"LOESS平滑失败: {e}，使用移动平均")
                    self.smoothing(method=0, window=window)

            else:
                raise ValueError("不支持的数据平滑方法")

        return self

    # 可视化对比
    def visualize(self, title: str = "数据清洗对比", figsize: tuple = (12, 8),
                  x_axis_interval: int = 7, cols_per_plot: int = 2) -> None:
        """
        可视化原始数据和处理后的数据对比（支持多列）

        参数:
            title: 图表标题
            figsize: 图表大小
            x_axis_interval: x轴标签显示间隔(天数)
            cols_per_plot: 每个图表显示的列数
        """
        if not self.value_cols:
            raise ValueError("请先设置值列")

        # 计算需要的行数和列数
        n_cols = len(self.value_cols)
        n_plots = (n_cols + cols_per_plot - 1) // cols_per_plot

        # 原始数据图表
        plt.figure(figsize=(figsize[0], figsize[1] * n_plots))
        for i, col in enumerate(self.value_cols):
            plt.subplot(n_plots, cols_per_plot, i + 1)
            if self.original_data_time_series is not None and col in self.original_data_time_series.columns:
                plt.plot(self.original_data_time_series.index, self.original_data_time_series[col], 'b-')
                plt.title(f"{title} - 原始数据 - {col}")
                plt.xlabel('日期')
                plt.ylabel(col)
                plt.grid(True)

                # 设置x轴标签间隔，使其更密集
                if isinstance(self.original_data_time_series.index, pd.DatetimeIndex):
                    plt.xticks(self.original_data_time_series.index[::x_axis_interval],
                               rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        # 处理后的数据图表
        plt.figure(figsize=(figsize[0], figsize[1] * n_plots))
        for i, col in enumerate(self.value_cols):
            plt.subplot(n_plots, cols_per_plot, i + 1)
            plt.plot(self.data.index, self.data[col], 'r-')
            plt.title(f"{title} - 处理后数据 - {col}")
            plt.xlabel('日期')
            plt.ylabel(col)
            plt.grid(True)

            # 设置x轴标签间隔，使其更密集
            if isinstance(self.data.index, pd.DatetimeIndex):
                plt.xticks(self.data.index[::x_axis_interval],
                           rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    # 返回原始数据
    def get_original_data(self) -> pd.DataFrame:
        return self.original_data

    # 返回处理后的数据
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.data

    # 保存处理后的数据
    def save_cleaned_data(self, path: str, index: bool = True) -> None:
        self.data.to_csv(path, index=index)
        print(f"数据已保存至 {path}")


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    # data = "./single.csv"
    # date_col="date"
    # value_cols = ["temperature"]
    # cleaner = DataCleaning(data=data, date_col=date_col, value_cols=value_cols, freq="D")

    data = "./multi.csv"
    date_col = "timestamp"
    value_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    cleaner = DataCleaning(data=data, date_col=date_col, value_cols=value_cols, freq="H")

    cleaner.handle_missing_values(method=1)  # 处理缺失值
    cleaner.handle_outliers(method=2, detect_method=2, threshold=2.5, window=10)  # 处理异常值
    # 数据平滑
    # cleaner.smoothing(method=1, alpha=0.3)
    # 可视化处理结果
    cleaner.visualize()
    # 获取处理后的数据
    cleaned_data = cleaner.get_cleaned_data()

    save_path = "{}{}.csv".format(data.rsplit(".", maxsplit=1)[0], "_cleaned")
    cleaner.save_cleaned_data(save_path)

    print("\n--------------- end ---------------\n")
