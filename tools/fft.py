# _*_ encoding: utf-8 _*_
# 文件: fft
# 时间: 2025/7/15_13:46
# 作者: GuanXK

import numpy as np
import matplotlib.pyplot as plt


def fft(x):
    """递归实现Cooley-Tukey快速傅里叶变换算法"""
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]

    # 基本情况：如果长度为1，直接返回
    if n == 1:
        return x

    # 检查长度是否为2的幂
    if n % 2 != 0:
        raise ValueError("输入长度必须是2的幂")

    # 递归计算偶数和奇数部分
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # 合并结果
    k = np.arange(n // 2)
    twiddle_factors = np.exp(-2j * np.pi * k / n)
    return np.concatenate([even + twiddle_factors * odd,
                           even - twiddle_factors * odd])


def ifft(X):
    """快速傅里叶逆变换"""
    # 利用FFT实现IFFT，通过共轭对称性
    X_conj = np.conj(X)
    x_conj = fft(X_conj)
    x = np.conj(x_conj) / len(X)
    return x


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    # 生成测试信号
    t = np.linspace(0, 1, 1024, endpoint=False)  # 时间向量
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)  # 信号

    # 使用自定义FFT计算频谱
    X_custom = fft(x)

    # 使用NumPy的FFT作为对比
    X_numpy = np.fft.fft(x)

    # 计算频率轴
    freq = np.fft.fftfreq(len(t), t[1] - t[0])

    # 绘制原始信号
    plt.figure(figsize=(12, 10))
    plt.subplot(311)
    plt.plot(t, x)
    plt.title('原始信号')
    plt.xlabel('时间 [秒]')
    plt.ylabel('幅度')

    # 绘制自定义FFT结果
    plt.subplot(312)
    plt.plot(freq[:len(freq) // 2], 2.0 / len(x) * np.abs(X_custom[:len(freq) // 2]))
    plt.title('自定义FFT幅度谱')
    plt.xlabel('频率 [Hz]')
    plt.ylabel('幅度')
    plt.grid()

    # 绘制NumPy FFT结果
    plt.subplot(313)
    plt.plot(freq[:len(freq) // 2], 2.0 / len(x) * np.abs(X_numpy[:len(freq) // 2]))
    plt.title('NumPy FFT幅度谱')
    plt.xlabel('频率 [Hz]')
    plt.ylabel('幅度')
    plt.grid()

    plt.tight_layout()
    plt.legend()
    plt.show()

    # 验证逆变换
    x_reconstructed = ifft(X_custom)
    max_error = np.max(np.abs(x - x_reconstructed))
    print(f"逆变换最大误差: {max_error}")

    print("\n--------------- end ---------------\n")
