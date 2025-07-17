# _*_ encoding: utf-8 _*_
# 文件: stationarity_tests.py
# 时间: 2025/7/15_17:49
# 作者: GuanXK

# system
from typing import Union, Optional, List, Dict, Any

# third_party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox

# 设置全局字体为 SimHei（黑体）
rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei' 等其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 正确显示负号


class StationarityTester:
    def __init__(self, data: Union[str, pd.DataFrame, pd.Series] = None,
                 column_name: Optional[Union[str, List[str]]] = None,
                 date_col: Optional[str] = None,
                 value_cols: Optional[Union[str, List[str]]] = None):
        """
        时序数据平稳性检验类，支持自动数据变换

        参数:
            data: 时序数据，可以是DataFrame、Series或文件路径
            column_name: 列名或列名列表(用于图表和报告)
            date_col: 日期列名(仅当data为DataFrame或文件路径时需要)
            value_cols: 值列名或列名列表(仅当data为DataFrame或文件路径时需要)
        """
        self.original_data = None  # 存储原始数据
        self.transformed_data = None  # 存储变换后的数据
        self.transformations = {}  # 存储每列的变换方法
        self.column_names = []
        self.results = {}

        if data is not None:
            self.set_data(data, column_name, date_col, value_cols)

    def set_data(self, data: Union[str, pd.DataFrame, pd.Series],
                 column_name: Optional[Union[str, List[str]]] = None,
                 date_col: Optional[str] = None,
                 value_col: Optional[Union[str, List[str]]] = None) -> 'StationarityTester':
        """设置检验数据"""
        # 处理单变量或多变量输入
        if isinstance(value_col, str):
            value_col = [value_col]

        if isinstance(column_name, str):
            column_name = [column_name]

        if isinstance(data, str):
            # 从文件路径读取数据
            self.original_data = pd.read_csv(data)
            if date_col and value_col:
                self.original_data[date_col] = pd.to_datetime(self.original_data[date_col])
                self.original_data = self.original_data.set_index(date_col)[value_col]
                self.column_names = value_col if column_name is None else column_name
            else:
                raise ValueError("当data为文件路径时，必须提供date_col和value_col")

        elif isinstance(data, pd.DataFrame):
            # 从DataFrame获取数据
            if date_col and value_col:
                self.original_data = data.set_index(date_col)[value_col]
                self.column_names = value_col if column_name is None else column_name
            else:
                raise ValueError("当data为DataFrame时，必须提供date_col和value_col")

        elif isinstance(data, pd.Series):
            # 直接使用Series(单变量)
            self.original_data = pd.DataFrame(data)
            self.column_names = [data.name] if column_name is None else column_name

        else:
            raise TypeError("data必须是str、DataFrame或Series类型")

        # 初始化变换后的数据为原始数据
        self.transformed_data = self.original_data.copy()
        self.transformations = {col: 'none' for col in self.column_names}

        return self

    def plot_time_series(self, figsize: tuple = (12, 8), x_axis_interval: int = 7, cols_per_plot: int = 2) -> None:
        """
        绘制时序图进行平稳性分析

        参数:
            figsize: 图表大小
            cols_per_plot: 每个图表显示的列数
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        # 计算需要的行数和列数
        n_cols = len(self.column_names)
        n_plots = (n_cols + cols_per_plot - 1) // cols_per_plot

        plt.figure(figsize=(figsize[0], figsize[1] * n_plots))
        for i, col in enumerate(self.column_names):
            plt.subplot(n_plots, cols_per_plot, i + 1)
            plt.plot(self.data.index, self.data[col], "b-")
            plt.title(f"时序图分析 - {col}")
            plt.xlabel('日期')
            plt.ylabel(col)
            plt.grid(True)

            # 设置x轴标签间隔，使其更密集
            if isinstance(self.data.index, pd.DatetimeIndex):
                plt.xticks(self.data.index[::x_axis_interval],
                           rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        # 分析趋势和季节性
        print("=== 时序图分析 ===")
        for col in self.column_names:
            print(f"\n{col}:")
            series = self.data[col]

            print("1. 趋势分析:")
            if series.is_monotonic_increasing:
                print("  - 明显上升趋势")
            elif series.is_monotonic_decreasing:
                print("  - 明显下降趋势")
            else:
                print("  - 趋势不明显或复杂")

            print("2. 季节性分析:")
            # 年度数据，需检查滞后1自相关性，并且至少需要2年的数据才能进行lag1计算
            # 季度数据，需检查滞后4（1年4个季度）自相关性，并且至少需要8个季度的数据才能进行lag4计算
            # 月度数据，需检查滞后12（1年12个月）自相关性，并且至少需要24个月的数据才能进行lag12计算
            # 周度数据，需检查滞后52（1年52个周）自相关性，并且至少需要104个周的数据才能进行lag52计算
            # 日度数据，需检查滞后365（1年365天）自相关性，并且至少需要730天的数据才能进行lag365计算
            if len(series) > 24:  # 假设月度数据
                autocorr = series.autocorr(12)
                # 自相关系数绝对值大于0.5为可能存在相关性，
                # 正数正相关，即2021年11月和2022年11月这两个月销量都高，每年均如此
                # 负数负相关，即2021年11月和2022年11月这两个月销量一高一低，2年一循环
                if abs(autocorr) > 0.5:
                    print(f"  - 可能存在季节性(滞后12的自相关系数: {autocorr:.4f})")
                else:
                    print("  - 季节性不明显")
            else:
                print("  - 数据点不足，无法进行季节性分析")

    def plot_acf_pacf(self, lags: int = 40, figsize: tuple = (12, 8), cols_per_plot: int = 1) -> None:
        """
        绘制自相关函数(ACF)和偏自相关函数(PACF)图

        参数:
            lags: 滞后阶数
            figsize: 图表大小
            cols_per_plot: 每个图表显示的变量数(1:每个变量一个图表，2:每个变量ACF和PACF并排)
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        for col in self.column_names:
            series = self.data[col]

            if cols_per_plot == 1:
                # 每个变量一个图表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
                fig.suptitle(f"ACF/PACF分析 - {col}")

                # 绘制ACF
                sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax1)
                ax1.set_title(f"自相关函数(ACF) - {col}")

                # 绘制PACF
                sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax2)
                ax2.set_title(f"偏自相关函数(PACF) - {col}")

            else:
                # 多个变量共享图表
                plt.figure(figsize=figsize)

                plt.subplot(1, 2, 1)
                sm.graphics.tsa.plot_acf(series, lags=lags)
                plt.title(f"自相关函数(ACF) - {col}")

                plt.subplot(1, 2, 2)
                sm.graphics.tsa.plot_pacf(series, lags=lags)
                plt.title(f"偏自相关函数(PACF) - {col}")

            plt.tight_layout()
            plt.show()

            # 分析ACF和PACF
            print(f"\n=== {col}的ACF/PACF分析 ===")
            print("1. 自相关函数(ACF)分析:")
            acf_values, confint = sm.tsa.acf(series, nlags=lags, alpha=0.05, fft=True)
            significant_lags = np.where(np.abs(acf_values) > confint[:, 1] - confint[:, 0])[0]

            if len(significant_lags) > 0:
                print(f"  - 显著滞后阶数: {list(significant_lags)}")
                if len(significant_lags) > lags / 2:
                    print("  - ACF衰减缓慢，可能非平稳")
                else:
                    print(f"  - ACF在滞后{significant_lags[-1]}后截断，可能为AR({significant_lags[-1]})过程")
            else:
                print("  - 无显著滞后阶数，序列可能接近白噪声")

            print("\n2. 偏自相关函数(PACF)分析:")
            pacf_values = sm.tsa.pacf(series, nlags=lags)
            significant_pacf_lags = np.where(np.abs(pacf_values) > 1.96 / np.sqrt(len(series)))[0]

            if len(significant_pacf_lags) > 0:
                print(f"  - 显著滞后阶数: {list(significant_pacf_lags)}")
                if len(significant_pacf_lags) > lags / 2:
                    print("  - PACF衰减缓慢，可能非平稳")
                else:
                    print(f"  - PACF在滞后{significant_pacf_lags[-1]}后截断，可能为MA({significant_pacf_lags[-1]})过程")
            else:
                print("  - 无显著滞后阶数，序列可能接近白噪声")

    def adf_test(self, autolag: str = 'AIC') -> None:
        """
        执行ADF单位根检验

        参数:
            autolag: 滞后阶数选择方法('AIC', 'BIC', 't-stat', None)
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        print("=== ADF单位根检验 ===")

        self.results['adf'] = {}

        for col in self.column_names:
            series = self.data[col]
            # regression不同的选择
            # *'c': constant only(default).
            # *'ct': constant and trend.
            # *'ctt': constant, and linear and quadratic trend.
            # *'nc': no constant, no trend.
            result = adfuller(series, regression="c", autolag=autolag)

            self.results['adf'][col] = {
                'statistic': result[0],
                'pvalue': result[1],
                'usedlag': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'icbest': result[5] if len(result) > 5 else None
            }

            print(f"\n{col}:")
            print(f"  检验统计量: {result[0]:.4f}")
            print(f"  p值: {result[1]:.4f}")
            print(f"  使用的滞后阶数: {result[2]}")
            print(f"  样本量: {result[3]}")

            print("\n  临界值:")
            for key, value in result[4].items():
                print(f"    {key}: {value:.4f}")

            # 判断
            if result[1] <= 0.05:
                print("\n  结论:")
                print(f"    p值({result[1]:.4f}) <= 0.05，拒绝原假设")
                print("    序列不存在单位根，是平稳的")
            else:
                print("\n  结论:")
                print(f"    p值({result[1]:.4f}) > 0.05，无法拒绝原假设")
                print("    序列存在单位根，是非平稳的")

    def pp_test(self, lags: int = None) -> None:
        """
        执行PP(Phillips-Perron)单位根检验

        参数:
            lags: 滞后阶数
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        print("=== PP(Phillips-Perron)单位根检验 ===")

        self.results['pp'] = {}

        try:
            # 尝试使用arch库中的PhillipsPerron检验
            from arch.unitroot import PhillipsPerron

            for col in self.column_names:
                series = self.data[col]
                pp_test = PhillipsPerron(series, trend="c", lags=lags)

                self.results['pp'][col] = {
                    'statistic': pp_test.stat,
                    'pvalue': pp_test.pvalue,
                    'usedlag': pp_test.lags,
                    'nobs': pp_test.nobs,
                    'critical_values': pp_test.critical_values
                }

                print(f"\n{col}:")
                print(f"  检验统计量: {pp_test.stat:.4f}")
                print(f"  p值: {pp_test.pvalue:.4f}")
                print(f"  使用的滞后阶数: {pp_test.lags}")
                print(f"  样本量: {pp_test.nobs}")

                print("\n  临界值:")
                for key, value in pp_test.critical_values.items():
                    print(f"    {key}: {value:.4f}")

                # 判断
                if pp_test.pvalue <= 0.05:
                    print("\n  结论:")
                    print(f"    p值({pp_test.pvalue:.4f}) <= 0.05，拒绝原假设")
                    print("    序列不存在单位根，是平稳的")
                else:
                    print("\n  结论:")
                    print(f"    p值({pp_test.pvalue:.4f}) > 0.05，无法拒绝原假设")
                    print("    序列存在单位根，是非平稳的")

        except ImportError:
            # 如果arch库不可用，则使用adfuller作为替代
            print("警告: 未安装arch库，将使用adfuller替代PP检验")
            print("提示: 安装arch库以获得更准确的PP检验结果: pip install arch")

            for col in self.column_names:
                series = self.data[col]
                result = adfuller(series, regression="c", maxlag=lags)

                result_dict = {
                    'statistic': result[0],
                    'pvalue': result[1],
                    'usedlag': result[2],
                    'nobs': result[3],
                    'critical_values': result[4],
                    'icbest': result[5] if len(result) > 5 else None
                }

                self.results['pp'][col] = result_dict

                print(f"\n{col}:")
                print(f"  检验统计量: {result_dict['statistic']:.4f}")
                print(f"  p值: {result_dict['pvalue']:.4f}")
                print(f"  使用的滞后阶数: {result_dict['usedlag']}")
                print(f"  样本量: {result_dict['nobs']}")

                print("\n  临界值:")
                for key, value in result_dict['critical_values'].items():
                    print(f"    {key}: {value:.4f}")

                # 判断
                if result_dict['pvalue'] <= 0.05:
                    print("\n  结论:")
                    print(f"    p值({result_dict['pvalue']:.4f}) <= 0.05，拒绝原假设")
                    print("    序列不存在单位根，是平稳的")
                else:
                    print("\n  结论:")
                    print(f"    p值({result_dict['pvalue']:.4f}) > 0.05，无法拒绝原假设")
                    print("    序列存在单位根，是非平稳的")

    def kpss_test(self, regression: str = 'c', lags: str = 'auto') -> None:
        """
        执行KPSS平稳性检验

        参数:
            regression: 回归类型('c': 含常数项, 'ct': 含常数项和趋势项)
            lags: 滞后阶数('auto'或整数值)
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        print("=== KPSS平稳性检验 ===")

        self.results['kpss'] = {}

        for col in self.column_names:
            series = self.data[col]
            result = kpss(series, regression=regression, nlags=lags)

            self.results['kpss'][col] = {
                'statistic': result[0],
                'pvalue': result[1],
                'usedlag': result[2],
                'critical_values': result[3]
            }

            print(f"\n{col}:")
            print(f"  检验统计量: {result[0]:.4f}")
            print(f"  p值: {result[1]:.4f}")
            print(f"  使用的滞后阶数: {result[2]}")

            print("\n  临界值:")
            for key, value in result[3].items():
                print(f"    {key}: {value:.4f}")

            # 判断
            if result[1] <= 0.05:
                print("\n  结论:")
                print(f"    p值({result[1]:.4f}) <= 0.05，拒绝原假设")
                print("    序列存在单位根，是非平稳的")
            else:
                print("\n  结论:")
                print(f"    p值({result[1]:.4f}) > 0.05，无法拒绝原假设")
                print("    序列是平稳的")

    def summarize_results(self) -> None:
        """总结所有检验结果"""
        if not self.results:
            print("请先执行至少一种平稳性检验")
            return

        print("\n\n=== 平稳性检验总结 ===")

        for col in self.column_names:
            print(f"\n\n===== {col} =====")

            # 时间序列图分析结果
            print("\n1. 时间序列图分析:")
            print("   - 需查看图形判断趋势和季节性")

            # ACF/PACF分析结果
            print("\n2. ACF/PACF分析:")
            print("   - 需查看图形判断自相关性衰减特性")

            # 单位根检验结果
            print("\n3. 单位根检验:")
            if 'adf' in self.results and col in self.results['adf']:
                adf_p = self.results['adf'][col]['pvalue']
                adf_conclusion = "平稳" if adf_p <= 0.05 else "非平稳"
                print(f"   - ADF检验: p值={adf_p:.4f}，结论: {adf_conclusion}")

            if 'pp' in self.results and col in self.results['pp']:
                pp_p = self.results['pp'][col]['pvalue']
                pp_conclusion = "平稳" if pp_p <= 0.05 else "非平稳"
                print(f"   - PP检验: p值={pp_p:.4f}，结论: {pp_conclusion}")

            # KPSS检验结果
            print("\n4. KPSS检验:")
            if 'kpss' in self.results and col in self.results['kpss']:
                kpss_p = self.results['kpss'][col]['pvalue']
                kpss_conclusion = "非平稳" if kpss_p <= 0.05 else "平稳"
                print(f"   - KPSS检验: p值={kpss_p:.4f}，结论: {kpss_conclusion}")

            # 综合判断
            print("\n5. 综合判断:")
            if ('adf' in self.results and col in self.results['adf'] and
                    'kpss' in self.results and col in self.results['kpss']):

                adf_conclusion = self.results['adf'][col]['pvalue'] <= 0.05
                kpss_conclusion = self.results['kpss'][col]['pvalue'] > 0.05

                if adf_conclusion and kpss_conclusion:
                    print("   - 序列是平稳的 (ADF和KPSS检验均支持)")
                elif not adf_conclusion and not kpss_conclusion:
                    print("   - 序列是非平稳的 (ADF和KPSS检验均支持)")
                elif adf_conclusion and not kpss_conclusion:
                    print("   - 序列是趋势平稳的 (ADF支持平稳，KPSS支持非平稳)")
                else:
                    print("   - 序列是非平稳的 (ADF支持非平稳，KPSS支持平稳)")
                    print("   - 建议进行差分处理后再次检验")

    def test_stationarity(self, alpha: float = 0.05, print_results: bool = True) -> Dict[str, Dict[str, bool]]:
        """
        执行所有平稳性检验并返回结果

        参数:
            alpha: 显著性水平
            print_results: 是否打印结果

        返回:
            各检验的平稳性判断结果
        """
        if self.transformed_data is None:
            raise ValueError("请先设置数据")

        self.results['stationarity'] = {}

        for col in self.column_names:
            series = self.transformed_data[col]

            # ADF检验
            adf_result = adfuller(series)
            adf_p = adf_result[1]
            adf_stationary = adf_p <= alpha

            # KPSS检验
            kpss_result = kpss(series)
            kpss_p = kpss_result[1]
            kpss_stationary = kpss_p > alpha  # KPSS原假设是平稳

            # 综合判断
            if adf_stationary and kpss_stationary:
                conclusion = '严格平稳'  # 强平稳
                conclusion_code = 'strictly_stationary'
            elif not adf_stationary and not kpss_stationary:
                conclusion = '非平稳'  # 非平稳
                conclusion_code = 'non_stationary'
            elif adf_stationary and not kpss_stationary:
                conclusion = '趋势平稳'  # 趋势平稳
                conclusion_code = 'trend_stationary'
            else:
                conclusion = '差分平稳'  # 差分平稳
                conclusion_code = 'difference_stationary'

            self.results['stationarity'][col] = {
                'adf_p': adf_p,
                'adf_stationary': adf_stationary,
                'kpss_p': kpss_p,
                'kpss_stationary': kpss_stationary,
                'conclusion': conclusion_code,
                'conclusion_text': conclusion
            }

        if print_results:
            self._print_stationarity_results()

        return self.results['stationarity']

    def _print_stationarity_results(self) -> None:
        """打印平稳性检验结果"""
        print("\n\n=== 平稳性检验结果 ===")

        for col in self.column_names:
            result = self.results['stationarity'][col]

            print(f"\n{col}:")
            print(f"  ADF检验: p值 = {result['adf_p']:.4f}, {'平稳' if result['adf_stationary'] else '非平稳'}")
            print(f"  KPSS检验: p值 = {result['kpss_p']:.4f}, {'平稳' if result['kpss_stationary'] else '非平稳'}")

            # 彩色打印综合结论
            if result['conclusion'] == 'strictly_stationary':
                print("\033[92m  综合结论: 严格平稳 ✅\033[0m")  # 绿色
            elif result['conclusion'] == 'non_stationary':
                print("\033[91m  综合结论: 非平稳 ❗\033[0m")  # 红色
            elif result['conclusion'] == 'trend_stationary':
                print("\033[93m  综合结论: 趋势平稳 ⚠️\033[0m")  # 黄色
            else:
                print("\033[94m  综合结论: 差分平稳 ➡️\033[0m")  # 蓝色

    def transform_data(self, method: str = 'auto', columns: Optional[List[str]] = None,
                       **kwargs) -> 'StationarityTester':
        """
        对数据进行变换

        参数:
            method: 变换方法 ('auto', 'diff', 'log', 'boxcox', 'seasonal_diff', 'sqrt')
            columns: 要变换的列名列表，默认为所有列
            **kwargs: 变换参数
        """
        if self.original_data is None:
            raise ValueError("请先设置数据")

        columns = columns or self.column_names

        for col in columns:
            if method == 'auto':
                # 自动检测非平稳类型并选择变换方法
                stationarity = self.test_stationarity()[col]

                if stationarity['conclusion'] == 'non_stationary':
                    # 检查是否有季节性
                    if self._detect_seasonality(self.original_data[col]):
                        # 季节性非平稳
                        self.transformed_data[col] = self._apply_seasonal_diff(
                            self.original_data[col],
                            period=kwargs.get('period', 12)
                        )
                        self.transformations[col] = f'seasonal_diff_{kwargs.get("period", 12)}'
                    else:
                        # 趋势非平稳
                        self.transformed_data[col] = self._apply_diff(
                            self.original_data[col],
                            order=kwargs.get('order', 1)
                        )
                        self.transformations[col] = f'diff_{kwargs.get("order", 1)}'

                elif stationarity['conclusion'] == 'trend_stationary':
                    # 趋势平稳，尝试对数变换
                    self.transformed_data[col] = self._apply_log(self.original_data[col])
                    self.transformations[col] = 'log'

                else:
                    # 已经是平稳的，不需要变换
                    print(f"列 {col} 已经是平稳的，跳过变换")

            elif method == 'diff':
                self.transformed_data[col] = self._apply_diff(
                    self.original_data[col] if self.transformations[col] == 'none'
                    else self.transformed_data[col],
                    order=kwargs.get('order', 1)
                )
                self.transformations[col] = f'diff_{kwargs.get("order", 1)}'

            elif method == 'log':
                self.transformed_data[col] = self._apply_log(
                    self.original_data[col] if self.transformations[col] == 'none'
                    else self.transformed_data[col]
                )
                self.transformations[col] = 'log'

            elif method == 'boxcox':
                self.transformed_data[col] = self._apply_boxcox(
                    self.original_data[col] if self.transformations[col] == 'none'
                    else self.transformed_data[col]
                )
                self.transformations[col] = 'boxcox'

            elif method == 'seasonal_diff':
                self.transformed_data[col] = self._apply_seasonal_diff(
                    self.original_data[col] if self.transformations[col] == 'none'
                    else self.transformed_data[col],
                    period=kwargs.get('period', 12)
                )
                self.transformations[col] = f'seasonal_diff_{kwargs.get("period", 12)}'

            elif method == 'sqrt':
                self.transformed_data[col] = self._apply_sqrt(
                    self.original_data[col] if self.transformations[col] == 'none'
                    else self.transformed_data[col]
                )
                self.transformations[col] = 'sqrt'

            else:
                raise ValueError(f"不支持的变换方法: {method}")

        return self

    def _detect_seasonality(self, series: pd.Series, period: int = 12) -> bool:
        """检测序列是否存在季节性"""
        if len(series) < 2 * period:
            return False

        # 检查滞后period的自相关系数
        autocorr = series.autocorr(period)
        return abs(autocorr) > 0.5

    def _apply_diff(self, series: pd.Series, order: int = 1) -> pd.Series:
        """应用差分变换"""
        result = series.copy()
        for _ in range(order):
            result = result.diff().dropna()
        return result

    def _apply_log(self, series: pd.Series) -> pd.Series:
        """应用对数变换"""
        # 确保所有值都是正数
        if (series <= 0).any():
            series = series + abs(series.min()) + 1e-10
        return np.log(series)

    def _apply_boxcox(self, series: pd.Series) -> pd.Series:
        """应用Box-Cox变换"""
        # 确保所有值都是正数
        if (series <= 0).any():
            series = series + abs(series.min()) + 1e-10

        transformed, _ = boxcox(series)
        return pd.Series(transformed, index=series.index)

    def _apply_seasonal_diff(self, series: pd.Series, period: int = 12) -> pd.Series:
        """应用季节性差分"""
        return series.diff(period).dropna()

    def _apply_sqrt(self, series: pd.Series) -> pd.Series:
        """应用平方根变换"""
        # 确保所有值都是非负的
        if (series < 0).any():
            series = series + abs(series.min())
        return np.sqrt(series)

    def plot_comparison(self, columns: Optional[List[str]] = None, figsize: tuple = (12, 8)) -> None:
        """
        绘制原始数据和变换后数据的对比图

        参数:
            columns: 要绘制的列名列表，默认为所有列
            figsize: 图表大小
        """
        if self.original_data is None or self.transformed_data is None:
            raise ValueError("请先设置数据并执行变换")

        columns = columns or self.column_names

        for col in columns:
            plt.figure(figsize=figsize)

            plt.subplot(2, 1, 1)
            plt.plot(self.original_data.index, self.original_data[col])
            plt.title(f'原始数据 - {col}')
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(self.transformed_data.index, self.transformed_data[col])
            plt.title(f'变换后数据 - {self.transformations[col]}')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    def summarize_transformations(self) -> None:
        """总结所有列的变换方法"""
        print("\n=== 数据变换总结 ===")
        for col, method in self.transformations.items():
            print(f"\n{col}:")
            print(f"  变换方法: {method}")

            if method != 'none':
                # 显示变换前后的平稳性检验结果
                before = self.test_stationarity(columns=[col])[col]
                print(f"  变换前: {before['conclusion']}")

                # 重新测试变换后的数据
                after = self.test_stationarity(columns=[col])[col]
                print(f"  变换后: {after['conclusion']}")

                if after['conclusion'] == 'strictly_stationary':
                    print("  ✅ 变换成功，序列已变为平稳")
                else:
                    print("  ❗ 变换后序列仍非平稳，建议尝试其他变换方法")
            else:
                print("  无需变换，序列已平稳")

    def export_transformed_data(self, file_path: str, index: bool = True, **kwargs) -> None:
        """
        将变换后的数据导出为CSV文件

        参数:
            file_path: 输出文件路径
            index: 是否包含索引
            **kwargs: pandas.to_csv()的其他参数
        """
        if self.transformed_data is None:
            raise ValueError("请先设置数据并执行变换")

        try:
            # 保存变换后的数据
            self.transformed_data.to_csv(file_path, index=index, **kwargs)

            # 保存变换方法记录
            method_path = file_path.rsplit('.', 1)[0] + '_transformations.csv'
            pd.DataFrame({
                'column': list(self.transformations.keys()),
                'transformation': list(self.transformations.values())
            }).to_csv(method_path, index=False)

            print(f"\n✅ 变换后的数据已保存至: {file_path}")
            print(f"✅ 变换方法记录已保存至: {method_path}")

        except Exception as e:
            print(f"❗ 导出数据时出错: {e}")


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    data = "./multi_cleaned.csv"
    date_col = "timestamp"
    value_cols = ["temperature", "humidity", "pressure", "wind_speed"]

    # 1. 使用文件路径
    tester = StationarityTester(
        data=data,
        date_col=date_col,
        value_cols=value_cols
    )
    # 2. 使用DataFrame
    # df = pd.read_csv("temperature_data.csv")
    # tester = StationarityTester(
    #     data=df,
    #     date_col="date",
    #     value_col="temperature"
    # )
    # 3. 使用Series
    # series = pd.read_csv("temperature_data.csv", index_col="date")["temperature"]
    # tester = StationarityTester(data=series)

    # 1. 时序图分析
    # tester.plot_time_series()

    # 2. ACF/PACF分析
    # tester.plot_acf_pacf(lags=12)

    # 3. 单位根检验
    # tester.adf_test()
    # tester.pp_test()

    # 4. KPSS检验
    # tester.kpss_test()

    # 5. 总结所有检验结果
    # tester.summarize_results()

    # 测试平稳性
    tester.test_stationarity()

    # 自动变换数据
    tester.transform_data(method='auto')

    # 导出数据
    tester.export_transformed_data("transformed_sales_data.csv")
    # tester.export_transformed_data("transformed_sales_data.csv", index=False)  # 导出时不包含索引

    # 查看变换总结
    # tester.summarize_transformations()

    # 绘制对比图
    # tester.plot_comparison()

    # 对变换后的数据再次进行平稳性检验
    # tester.test_stationarity()

    print("\n--------------- end ---------------\n")
