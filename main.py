# _*_ encoding: utf-8 _*_
# 文件: main
# 时间: 2025/6/28_13:21
# 作者: GuanXK

# system
import os
import sys

# third_party
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import dataset, dataloader


# custom

class Demo01Dataset:
    _EXT = [".csv"]

    def __init__(
            self,
            root_dir: str = None,
    ):
        super(Demo01Dataset, self).__init__()
        self.root_dir = root_dir
        self.datasets = {
            "train": {},
            "val": {},
            "test": {}
        }

        # datasets字典示例
        '''        
        {
            train:
            {
                NG20:
                {
                    0001.csv:
                    [
                        [1,2,3,4,5],
                        [2,3,4,5,6],
                        ...
                    ]
                }
            },
            val:
            {
                NG20:
                {
                    0001.csv:
                    [
                        [1,2,3,4,5],
                        [2,3,4,5,6],
                        ...
                    ]
                }
            }
        }
        '''

        self._preprocess()

    @staticmethod
    def _normalize(path, method="minmax"):
        data = pd.read_csv(path).to_numpy()
        data = data[:, 1:]
        if method == "minmax":
            # 最小-最大规范化：将数据缩放到 [0, 1] 范围
            if data.size > 0:  # 确保数据非空
                min_val = np.min(data)
                max_val = np.max(data)
                if max_val != min_val:  # 避免除零错误
                    data = (data - min_val) / (max_val - min_val)
                else:
                    data = np.zeros_like(data)  # 处理所有值相同的情况
        elif method == "meanstd":
            # Z-score 规范化：将数据转换为均值为 0，标准差为 1 的分布
            if data.size > 0:  # 确保数据非空
                mean_val = np.mean(data)
                std_val = np.std(data)
                if std_val != 0:  # 避免除零错误
                    data = (data - mean_val) / std_val
                else:
                    data = data - mean_val  # 处理标准差为零的情况
        else:
            raise ValueError(f"不支持的规范化方法：{method}")

        return data

    def _preprocess(self):
        for Root, Dir, Files in os.walk(self.root_dir):
            for File in Files:
                basename, ext = os.path.splitext(File)
                if ext not in self._EXT:
                    continue

                path = os.path.join(Root, File)
                data = self._normalize(path)

                class_name = os.path.basename(Root)
                split_info = os.path.basename(os.path.dirname(Root))
                if class_name not in self.datasets[split_info].keys():
                    self.datasets[split_info][class_name] = {

                    }
                else:
                    self.datasets[split_info][class_name].update({1: 2})

                print(split_info, class_name)
                print(path)

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return True

    def __repr__(self):
        pass


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    root_dir = "./datasets/demo01"

    Demo01Dataset(root_dir=root_dir)

    print("\n--------------- end ---------------\n")
