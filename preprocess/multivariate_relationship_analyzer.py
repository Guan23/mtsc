# _*_ encoding: utf-8 _*_
# 文件: multivariate_relationship_analyzer.py
# 时间: 2025/7/16_17:32
# 作者: GuanXK

# system
from typing import Union, List, Dict, Tuple, Optional

# third_party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from statsmodels.tsa.stattools import ccf, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置全局字体为 SimHei（黑体）
rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei' 等其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 正确显示负号


class MultivariateTimeSeriesAnalyzer:
    def __init__(self, data: Union[str, pd.DataFrame, pd.Series] = None,
                 date_col: Optional[str] = None,
                 value_cols: Optional[Union[str, List[str]]] = None):
        """
        多维时序数据维度关系检测类

        参数:
            data: 时序数据，可以是DataFrame、Series或文件路径
            date_col: 日期列名(仅当data为DataFrame或文件路径时需要)
            value_cols: 值列名或列名列表(仅当data为DataFrame或文件路径时需要)
        """
        self.original_data = None  # 存储原始数据
        self.data = None  # 存储处理后的数据
        self.results = {}

        if data is not None:
            self.set_data(data, date_col, value_cols)

    def set_data(self, data: Union[str, pd.DataFrame, pd.Series],
                 date_col: Optional[str] = None,
                 value_col: Optional[Union[str, List[str]]] = None) -> 'MultivariateTimeSeriesAnalyzer':
        """设置分析数据"""
        # 处理单变量或多变量输入
        if isinstance(value_col, str):
            value_col = [value_col]

        if isinstance(data, str):
            # 从文件路径读取数据
            self.original_data = pd.read_csv(data)
            if date_col and value_col:
                self.original_data[date_col] = pd.to_datetime(self.original_data[date_col])
                self.data = self.original_data.set_index(date_col)[value_col]
            else:
                raise ValueError("当data为文件路径时，必须提供date_col和value_col")

        elif isinstance(data, pd.DataFrame):
            # 从DataFrame获取数据
            if date_col and value_col:
                self.original_data = data
                self.data = data.set_index(date_col)[value_col]
            else:
                raise ValueError("当data为DataFrame时，必须提供date_col和value_col")

        elif isinstance(data, pd.Series):
            # 直接使用Series(单变量)
            self.original_data = pd.DataFrame(data)
            self.data = self.original_data
            if value_col is None and data.name is not None:
                self.data.columns = [data.name]
            elif value_col:
                self.data.columns = value_col
            else:
                self.data.columns = ['value']

        else:
            raise TypeError("data必须是str、DataFrame或Series类型")

        return self

    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        计算变量间的相关系数矩阵

        参数:
            method: 相关系数计算方法 ('pearson', 'kendall', 'spearman')
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        corr_matrix = self.data.corr(method=method)
        self.results['correlation'] = {
            'matrix': corr_matrix,
            'method': method
        }
        return corr_matrix

    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                                 annot: bool = True, cmap: str = 'coolwarm') -> None:
        """
        绘制相关系数热力图

        参数:
            figsize: 图表大小
            annot: 是否显示数值
            cmap: 颜色映射
        """
        if 'correlation' not in self.results:
            self.calculate_correlation()

        plt.figure(figsize=figsize)
        sns.heatmap(self.results['correlation']['matrix'], annot=annot, cmap=cmap,
                    square=True, linewidths=.5, vmin=-1, vmax=1)
        plt.title(f"变量间{self.results['correlation']['method']}相关系数热力图")
        plt.tight_layout()
        plt.show()

    def calculate_cross_correlation(self, var1: str, var2: str, lags: int = 20) -> np.ndarray:
        """
        计算两个变量的交叉相关函数

        参数:
            var1: 第一个变量名
            var2: 第二个变量名
            lags: 滞后阶数
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        if var1 not in self.data.columns or var2 not in self.data.columns:
            raise ValueError(f"变量名不存在: {var1} 或 {var2}")

        # 计算交叉相关函数
        cc = ccf(self.data[var1], self.data[var2])[:lags + 1]  # 取正滞后部分

        self.results['cross_correlation'] = {
            'var1': var1,
            'var2': var2,
            'lags': lags,
            'values': cc
        }
        return cc

    def plot_cross_correlation(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """绘制交叉相关函数图"""
        if 'cross_correlation' not in self.results:
            raise ValueError("请先计算交叉相关函数")

        cc = self.results['cross_correlation']

        plt.figure(figsize=figsize)
        plt.stem(range(cc['lags'] + 1), cc['values'])
        plt.axhline(y=0, linestyle='-', color='gray', alpha=0.3)
        plt.axhline(y=1.96 / np.sqrt(len(self.data)), linestyle='--', color='red', alpha=0.3)
        plt.axhline(y=-1.96 / np.sqrt(len(self.data)), linestyle='--', color='red', alpha=0.3)
        plt.title(f"{cc['var1']} 与 {cc['var2']} 的交叉相关函数")
        plt.xlabel('滞后阶数')
        plt.ylabel('相关系数')
        plt.tight_layout()
        plt.show()

    def granger_causality_test(self, dependent_var: str, independent_var: str,
                               maxlag: int = 4) -> Dict:
        """
        执行格兰杰因果检验

        参数:
            dependent_var: 因变量名
            independent_var: 自变量名
            maxlag: 最大滞后阶数
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        if dependent_var not in self.data.columns or independent_var not in self.data.columns:
            raise ValueError(f"变量名不存在: {dependent_var} 或 {independent_var}")

        # 准备数据
        df = pd.DataFrame({
            'y': self.data[dependent_var],
            'x': self.data[independent_var]
        })

        # 执行格兰杰因果检验
        results = grangercausalitytests(df, maxlag=maxlag, verbose=False)

        # 提取F检验结果
        p_values = {lag: results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)}

        self.results['granger_causality'] = {
            'dependent_var': dependent_var,
            'independent_var': independent_var,
            'maxlag': maxlag,
            'p_values': p_values,
            'conclusion': self._interpret_granger(p_values)
        }

        return self.results['granger_causality']

    def _interpret_granger(self, p_values: Dict) -> str:
        """解释格兰杰因果检验结果"""
        significant_lags = [lag for lag, p in p_values.items() if p < 0.05]

        if significant_lags:
            return f"在滞后阶数 {', '.join(map(str, significant_lags))} 下存在格兰杰因果关系"
        else:
            return "不存在格兰杰因果关系"

    def cointegration_test(self, var1: str, var2: str, method: str = 'engle-granger') -> Dict:
        """
        执行协整检验

        参数:
            var1: 第一个变量名
            var2: 第二个变量名
            method: 检验方法 ('engle-granger', 'johansen')
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        if var1 not in self.data.columns or var2 not in self.data.columns:
            raise ValueError(f"变量名不存在: {var1} 或 {var2}")

        if method == 'engle-granger':
            # Engle-Granger协整检验
            result = coint(self.data[var1], self.data[var2])
            p_value = result[1]

            self.results['cointegration'] = {
                'method': method,
                'var1': var1,
                'var2': var2,
                'p_value': p_value,
                'is_cointegrated': p_value < 0.05,
                'statistic': result[0],
                'critical_values': result[2]
            }

        elif method == 'johansen':
            # Johansen协整检验（适用于多变量）
            if len(self.data.columns) < 2:
                raise ValueError("Johansen检验需要至少两个变量")

            result = coint_johansen(self.data, det_order=0, k_ar_diff=1)

            self.results['cointegration'] = {
                'method': method,
                'trace_stat': result.lr1,
                'eigen_stat': result.lr2,
                'critical_values_trace': result.cvt,
                'critical_values_eigen': result.cvm,
                'max_rank': result.max_eig_rank,
                'eigenvalues': result.eig
            }

        else:
            raise ValueError(f"不支持的协整检验方法: {method}")

        return self.results['cointegration']

    def perform_pca(self, n_components: Optional[int] = None,
                    scale: bool = True) -> pd.DataFrame:
        """
        执行主成分分析

        参数:
            n_components: 主成分数量，默认为变量数量
            scale: 是否标准化数据
        """
        if self.data is None:
            raise ValueError("请先设置数据")

        # 准备数据
        X = self.data.values

        # 标准化
        if scale:
            X = StandardScaler().fit_transform(X)

        # 执行PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X)

        # 创建主成分DataFrame
        pc_columns = [f'PC{i + 1}' for i in range(principal_components.shape[1])]
        pc_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=self.data.index)

        self.results['pca'] = {
            'components': pc_df,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': pd.DataFrame(pca.components_.T,
                                     columns=pc_columns,
                                     index=self.data.columns)
        }

        return pc_df

    def plot_pca_variance_explained(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """绘制PCA方差解释率图"""
        if 'pca' not in self.results:
            raise ValueError("请先执行PCA")

        pca_results = self.results['pca']

        plt.figure(figsize=figsize)

        # 绘制方差解释率
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['explained_variance_ratio'])
        plt.xlabel('主成分')
        plt.ylabel('方差解释率')
        plt.title('各主成分的方差解释率')

        # 绘制累积方差解释率
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(pca_results['cumulative_variance']) + 1),
                 pca_results['cumulative_variance'], 'o-')
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('主成分数量')
        plt.ylabel('累积方差解释率')
        plt.title('累积方差解释率')

        plt.tight_layout()
        plt.show()

    def plot_pca_loadings(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """绘制PCA载荷热图"""
        if 'pca' not in self.results:
            raise ValueError("请先执行PCA")

        plt.figure(figsize=figsize)
        sns.heatmap(self.results['pca']['loadings'], annot=True, cmap='coolwarm',
                    fmt='.2f', linewidths=.5)
        plt.title('主成分载荷热图')
        plt.tight_layout()
        plt.show()

    def summarize_relationships(self) -> None:
        """总结所有关系分析结果"""
        print("\n\n=== 多维时序数据关系分析总结 ===")

        # 相关性分析总结
        if 'correlation' in self.results:
            print("\n1. 相关性分析:")
            method = self.results['correlation']['method']
            corr_matrix = self.results['correlation']['matrix']

            # 找出强相关变量对
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_correlations.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            if strong_correlations:
                print(f"  发现{len(strong_correlations)}对强{method}相关性:")
                for var1, var2, corr in strong_correlations:
                    print(f"    {var1} 与 {var2}: {corr:.4f}")
            else:
                print(f"  未发现强{method}相关性变量对")

        # 交叉相关性总结
        if 'cross_correlation' in self.results:
            print("\n2. 交叉相关性分析:")
            cc = self.results['cross_correlation']
            significant_lags = [i for i, val in enumerate(cc['values'])
                                if abs(val) > 1.96 / np.sqrt(len(self.data))]

            if significant_lags:
                print(f"  {cc['var1']} 与 {cc['var2']} 在以下滞后阶数存在显著相关性:")
                print(f"    {significant_lags}")
            else:
                print(f"  {cc['var1']} 与 {cc['var2']} 不存在显著交叉相关性")

        # 格兰杰因果总结
        if 'granger_causality' in self.results:
            print("\n3. 格兰杰因果检验:")
            gc = self.results['granger_causality']
            print(f"  {gc['independent_var']} 是否格兰杰引起 {gc['dependent_var']}:")
            print(f"    {gc['conclusion']}")
            print("    各滞后阶数p值:")
            for lag, p in gc['p_values'].items():
                print(f"      滞后{lag}: {p:.4f} {'*' if p < 0.05 else ''}")

        # 协整总结
        if 'cointegration' in self.results:
            print("\n4. 协整检验:")
            coint_result = self.results['cointegration']

            if coint_result['method'] == 'engle-granger':
                print(f"  {coint_result['var1']} 与 {coint_result['var2']} 的Engle-Granger协整检验:")
                print(f"    p值: {coint_result['p_value']:.4f}")
                print(f"    {'存在' if coint_result['is_cointegrated'] else '不存在'}协整关系")
            else:
                print(f"  Johansen协整检验:")
                print(f"    最大秩: {coint_result['max_rank']}")
                print(f"    特征值: {[round(eig, 4) for eig in coint_result['eigenvalues']]}")

        # PCA总结
        if 'pca' in self.results:
            print("\n5. 主成分分析:")
            pca_result = self.results['pca']

            # 确定需要的主成分数量以解释至少80%的方差
            n_pc_80 = np.argmax(pca_result['cumulative_variance'] >= 0.8) + 1

            print(f"  解释至少80%方差所需的主成分数量: {n_pc_80}")
            print(f"  前{n_pc_80}个主成分的方差解释率:")
            for i in range(n_pc_80):
                print(f"    PC{i + 1}: {pca_result['explained_variance_ratio'][i]:.4f}")

            # 找出对每个主成分贡献最大的变量
            print("\n  各主成分的主要贡献变量:")
            for i in range(n_pc_80):
                pc = f'PC{i + 1}'
                top_features = pca_result['loadings'][pc].abs().sort_values(ascending=False).head(3)
                print(f"    {pc}: {', '.join([f'{feat}({val:.2f})' for feat, val in top_features.items()])}")

    def handle_high_correlation(self, method: str = 'pca', threshold: float = 0.7,
                                **kwargs) -> 'MultivariateTimeSeriesAnalyzer':
        """
        处理强相关维度

        参数:
            method: 处理方法 ('auto', 'pca', 'drop', 'combine', 'select')
            threshold: 相关性阈值，超过此值被视为强相关
            **kwargs: 特定方法的额外参数
        """
        if 'correlation' not in self.results:
            self.calculate_correlation()

        corr_matrix = self.results['correlation']['matrix']

        # 找出所有强相关的变量对
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    strong_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        if not strong_pairs:
            print("未发现强相关变量对，无需处理")
            return self

        print(f"发现{len(strong_pairs)}对强相关变量，处理方法: {method}")

        if method == 'auto':
            # 自动选择最优处理方法
            method = self._select_optimal_method(strong_pairs, corr_matrix, **kwargs)
            self.results['correlation_handling_method'] = f"auto → {method}"
            return self.handle_high_correlation(method=method, threshold=threshold, **kwargs)

        elif method == 'pca':
            # 使用PCA降维
            n_components = kwargs.get('n_components', min(3, len(self.data.columns)))
            self.perform_pca(n_components=n_components)

            # 用主成分替换原始数据
            self.data = self.results['pca']['components']
            self.results['correlation_post_processing'] = self.data.corr()

        elif method == 'drop':
            # 删除冗余特征
            to_drop = self._select_features_to_drop(strong_pairs, corr_matrix)
            self.data = self.data.drop(columns=to_drop)
            self.results['dropped_features'] = to_drop
            self.calculate_correlation()  # 重新计算相关性

        elif method == 'combine':
            # 组合强相关特征
            combined_features = self._combine_features(strong_pairs, **kwargs)
            self.data = pd.concat([self.data.drop(columns=[pair[0] for pair in strong_pairs]),
                                   combined_features], axis=1)
            self.calculate_correlation()  # 重新计算相关性

        elif method == 'select':
            # 基于重要性选择特征
            selected = self._select_features(strong_pairs, **kwargs)
            self.data = self.data[selected]
            self.results['selected_features'] = selected
            self.calculate_correlation()  # 重新计算相关性

        else:
            raise ValueError(f"不支持的处理方法: {method}")

        return self

    def _select_optimal_method(self, strong_pairs: List[Tuple[str, str]],
                               corr_matrix: pd.DataFrame, **kwargs) -> str:
        """
        基于数据特性自动选择最优的高相关性处理方法

        决策逻辑:
        1. 若存在高度相关的变量组(平均相关系数>0.85) → PCA降维
        2. 若变量间相关模式较复杂(相关系数分布分散) → PCA降维
        3. 若变量重要性差异大 → 特征选择
        4. 若变量重要性相近 → 特征组合
        5. 默认使用PCA
        """
        # 计算强相关变量组的平均相关系数
        strong_corr_values = [corr_matrix.loc[var1, var2] for var1, var2 in strong_pairs]
        avg_strong_corr = np.mean(np.abs(strong_corr_values))

        # 计算相关系数的标准差(衡量分布分散程度)
        std_strong_corr = np.std(np.abs(strong_corr_values))

        # 检查是否提供了特征重要性
        importance = kwargs.get('importance')
        if importance is None:
            # 默认使用方差作为重要性指标
            importance = self.data.var().to_dict()

        # 计算重要性的变异系数(标准差/均值)
        importance_values = list(importance.values())
        cv_importance = np.std(importance_values) / np.mean(importance_values)

        # 决策逻辑
        if avg_strong_corr > 0.85:
            print("检测到高度相关变量组，选择PCA降维")
            return 'pca'

        elif std_strong_corr > 0.2:
            print("相关系数分布分散，选择PCA降维")
            return 'pca'

        elif cv_importance > 0.5:
            print("特征重要性差异大，选择特征选择")
            return 'select'

        else:
            print("特征重要性相近，选择特征组合")
            return 'combine'

    def _select_features_to_drop(self, strong_pairs: List[Tuple[str, str]],
                                 corr_matrix: pd.DataFrame) -> List[str]:
        """基于相关性选择要删除的特征"""
        # 计算每个特征的总相关性强度
        corr_strength = corr_matrix.abs().sum() - 1  # 减去自身相关性

        # 决定删除哪些特征
        to_drop = set()
        for var1, var2 in strong_pairs:
            if var1 not in to_drop and var2 not in to_drop:
                # 删除总相关性更强的特征(更冗余)
                if corr_strength[var1] > corr_strength[var2]:
                    to_drop.add(var1)
                else:
                    to_drop.add(var2)

        return list(to_drop)

    def _combine_features(self, strong_pairs: List[Tuple[str, str]],
                          method: str = 'average') -> pd.DataFrame:
        """组合强相关特征"""
        combined = pd.DataFrame(index=self.data.index)

        for i, (var1, var2) in enumerate(strong_pairs):
            if method == 'average':
                # 取平均值
                combined[f'combined_{i + 1}'] = (self.data[var1] + self.data[var2]) / 2
            elif method == 'weighted':
                # 加权平均，权重基于方差
                var1_var = self.data[var1].var()
                var2_var = self.data[var2].var()
                weight1 = var2_var / (var1_var + var2_var)
                weight2 = var1_var / (var1_var + var2_var)
                combined[f'combined_{i + 1}'] = weight1 * self.data[var1] + weight2 * self.data[var2]
            else:
                raise ValueError(f"不支持的组合方法: {method}")

        return combined

    def _select_features(self, strong_pairs: List[Tuple[str, str]],
                         importance: Optional[Dict[str, float]] = None) -> List[str]:
        """基于重要性选择特征"""
        if importance is None:
            # 默认使用方差作为重要性指标
            importance = self.data.var().to_dict()

        selected = set(self.data.columns)

        for var1, var2 in strong_pairs:
            if var1 in selected and var2 in selected:
                # 保留更重要的特征
                if importance.get(var1, 0) > importance.get(var2, 0):
                    selected.discard(var2)
                else:
                    selected.discard(var1)

        return list(selected)

    def compare_pre_post_correlation(self, figsize: Tuple[int, int] = (15, 6)) -> None:
        """比较处理前后的相关性矩阵"""
        if 'correlation_post_processing' not in self.results:
            raise ValueError("请先执行相关性处理")

        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        sns.heatmap(self.results['correlation']['matrix'], annot=False, cmap='coolwarm',
                    square=True, linewidths=.5, vmin=-1, vmax=1)
        plt.title("处理前的相关性矩阵")

        plt.subplot(1, 2, 2)
        sns.heatmap(self.results['correlation_post_processing'], annot=False, cmap='coolwarm',
                    square=True, linewidths=.5, vmin=-1, vmax=1)
        plt.title("处理后的相关性矩阵")

        plt.tight_layout()
        plt.show()

    def summarize_correlation_handling(self) -> None:
        """总结相关性处理结果"""
        if 'correlation_post_processing' not in self.results:
            raise ValueError("请先执行相关性处理")

        print("\n\n=== 相关性处理总结 ===")

        method = self.results.get('correlation_handling_method', '未知')
        print(f"处理方法: {method}")

        # 计算处理前后的平均绝对相关性
        pre_corr = self.results['correlation']['matrix'].abs()
        np.fill_diagonal(pre_corr.values, 0)  # 排除对角线
        avg_pre = pre_corr.mean().mean()

        post_corr = self.results['correlation_post_processing'].abs()
        np.fill_diagonal(post_corr.values, 0)  # 排除对角线
        avg_post = post_corr.mean().mean()

        print(f"平均绝对相关性: 处理前={avg_pre:.4f}, 处理后={avg_post:.4f}")
        print(f"相关性降低比例: {(avg_pre - avg_post) / avg_pre * 100:.2f}%")

        if method == 'pca':
            print(f"使用PCA降维至{self.results['pca']['components'].shape[1]}个主成分")
            print(f"保留方差比例: {sum(self.results['pca']['explained_variance_ratio']):.4f}")

        elif method == 'drop':
            dropped = self.results.get('dropped_features', [])
            print(f"删除特征: {', '.join(dropped)}")

        elif method == 'select':
            selected = self.results.get('selected_features', [])
            print(f"保留特征: {', '.join(selected)}")

    def export_transformed_data(self, file_path: str, index: bool = True, **kwargs) -> None:
        """
        将变换后的数据导出为CSV文件

        参数:
            file_path: 输出文件路径
            index: 是否包含索引
            **kwargs: pandas.to_csv()的其他参数
        """
        if self.data is None:
            raise ValueError("请先设置数据并执行变换")

        try:
            # 保存变换后的数据
            self.data.to_csv(file_path, index=index, **kwargs)

            # 保存处理记录
            processing_path = file_path.rsplit('.', 1)[0] + '_processing.csv'
            processing_log = self._generate_processing_log()
            pd.DataFrame(processing_log).to_csv(processing_path, index=False)

            print(f"\n✅ 变换后的数据已保存至: {file_path}")
            print(f"✅ 处理记录已保存至: {processing_path}")

        except Exception as e:
            print(f"❗ 导出数据时出错: {e}")

    def _generate_processing_log(self) -> List[Dict]:
        """生成数据处理记录"""
        log = []

        # 添加相关性处理记录
        if 'correlation_handling_method' in self.results:
            method = self.results['correlation_handling_method']
            log.append({
                'step': '相关性处理',
                'method': method,
                'details': self._get_correlation_handling_details()
            })

        # 添加PCA处理记录
        if 'pca' in self.results:
            pca = self.results['pca']
            log.append({
                'step': 'PCA降维',
                'n_components': pca['components'].shape[1],
                'explained_variance': sum(pca['explained_variance_ratio']),
                'details': f"主成分: {', '.join(pca['components'].columns)}"
            })

        # 可添加更多处理步骤的记录...

        return log

    def _get_correlation_handling_details(self) -> str:
        """获取相关性处理的详细信息"""
        method = self.results['correlation_handling_method']

        if 'pca' in method:
            return f"降维至{self.results['pca']['components'].shape[1]}个主成分"

        elif 'drop' in method:
            dropped = self.results.get('dropped_features', [])
            return f"删除特征: {', '.join(dropped)}"

        elif 'select' in method:
            selected = self.results.get('selected_features', [])
            return f"保留特征: {', '.join(selected)}"

        elif 'combine' in method:
            combined = self.data.columns[self.data.columns.str.startswith('combined_')]
            return f"创建组合特征: {', '.join(combined)}"

        return "未知处理方法"


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    data = "./multi_cleaned.csv"
    date_col = "timestamp"
    value_cols = ["temperature", "humidity", "pressure", "wind_speed"]

    analyzer = MultivariateTimeSeriesAnalyzer(
        data=data,
        date_col=date_col,
        value_cols=value_cols
    )

    # 1. 相关性分析
    analyzer.calculate_correlation()
    analyzer.plot_correlation_heatmap()

    # 2. 交叉相关性分析
    analyzer.calculate_cross_correlation('temperature', 'humidity', lags=10)
    analyzer.plot_cross_correlation()

    # 3. 格兰杰因果检验
    analyzer.granger_causality_test('pressure', 'wind_speed', maxlag=5)

    # 4. 协整检验
    analyzer.cointegration_test('temperature', 'pressure')

    # 5. 主成分分析
    analyzer.perform_pca(n_components=3)
    analyzer.plot_pca_variance_explained()
    analyzer.plot_pca_loadings()

    # 6. 生成综合报告
    analyzer.summarize_relationships()

    # 自动处理强相关性
    analyzer.handle_high_correlation(method='auto', threshold=0.7)

    # 查看自动选择的方法和处理效果
    print(f"自动选择的处理方法: {analyzer.results['correlation_handling_method']}")
    analyzer.compare_pre_post_correlation()
    analyzer.summarize_correlation_handling()

    # 可选：提供自定义特征重要性
    importance = {'feature1': 0.8, 'feature2': 0.3, 'feature3': 0.6}
    analyzer.handle_high_correlation(method='auto', threshold=0.7, importance=importance)

    # 导出处理后的数据
    analyzer.export_transformed_data("processed_timeseries.csv")

    print("\n--------------- end ---------------\n")
