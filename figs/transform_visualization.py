import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import stumpy
from typing import List
import time

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_random_walk(n_steps=1000, seed=42):
    """生成随机游走时间序列"""
    np.random.seed(seed)
    steps = np.random.randn(n_steps)
    return np.cumsum(steps)

def signed_log_transform(x):
    """符号对数变换"""
    return np.sign(x) * np.log(1 + np.abs(x))

def asinh_transform(x):
    """双曲正弦反函数变换"""
    return np.arcsinh(x)

def offset_log_transform(x):
    """偏移对数变换"""
    offset = abs(np.min(x)) + 1  # 确保所有值为正
    return np.log(x + offset)

def piecewise_log_transform(x):
    """分段对数变换"""
    result = np.zeros_like(x)
    pos_mask = x > 0
    neg_mask = x < 0
    zero_mask = x == 0
    
    result[pos_mask] = np.log(x[pos_mask])
    result[neg_mask] = -np.log(-x[neg_mask])
    result[zero_mask] = 0
    
    return result

def cumsum_transform(x):
    """累积和变换"""
    return np.cumsum(x)

def diff_transform(x):
    """差分变换"""
    return np.diff(x, prepend=0)  # prepend=0保持数组长度一致

def rollsum_transform(x):
    """滚动和变换"""
    return np.convolve(x, np.ones(5), mode='same')

def slog1p_transform(x):
    """保号对数变换 SLOG1P"""
    return np.sign(x) * np.log1p(np.abs(x))

def signed_pow05_transform(x):
    """保号幂变换 SPOW05"""
    alpha = 0.5
    return np.sign(x) * (np.abs(x) ** alpha)

def hinge_pos_transform(x):
    """分段线性"铰链"基 HINGE_POS"""
    m = np.median(x)
    return np.maximum(x - m, 0.0)

def hinge_neg_transform(x):
    """分段线性"铰链"基 HINGE_NEG"""
    m = np.median(x)
    return np.maximum(m - x, 0.0)

def drawdown_transform(x):
    """回撤变换 DRAWDOWN"""
    cummax = np.maximum.accumulate(x)
    return x - cummax

def cummax_transform(x):
    """历史新高变换 CUMMAX"""
    return np.maximum.accumulate(x)

def asinh_diff_transform(x):
    """组合型：ASINH_DIFF（先非线性再差分）"""
    a = np.arcsinh(x)
    d = np.empty_like(a, dtype=float)
    d[0] = 0.0
    d[1:] = a[1:] - a[:-1]
    return d

def diff_lag7_transform(x):
    """季节/周期差分 DIFF_LAG7"""
    result = np.zeros_like(x)
    result[7:] = x[7:] - x[:-7]
    return result

def morph_gradient_w5_transform(x):
    """形态学梯度 MORPH_GRAD_w5"""
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - 2)
        end = min(len(x), i + 3)
        window = x[start:end]
        result[i] = np.max(window) - np.min(window)
    return result

def sobel_like_transform(x):
    """对称卷积核 - 类Sobel边缘检测"""
    kernel = np.array([1, 0, -1])  # 类Sobel算子核
    return np.convolve(x, kernel, mode='same') 

def orthogonal_kernel_transform(x):
    """正交核变换 - 使用Haar小波核"""
    kernel = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    return np.convolve(x, kernel, mode='same')  

def matrix_profile_transform(x):
    from stumpy import config
    config.STUMPY_THREADS_PER_BLOCK = 1024  # 最大线程数，适合 T4

    """矩阵轮廓"""
    s = time.time()
    m = 10
    mp = stumpy.stump(x, m)  # gpu_stump
    e = time.time()
    print(f'matrix_profile time: {e-s}')
    return mp[:,0]

def plot_transformations():
    """绘制原始时间序列和各种变换的对比图"""
    # 生成随机游走数据
    data = generate_random_walk(1000)
    time_steps = np.arange(len(data))
    
    # 创建子图 - 使用5行4列布局，优化大小和间距
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    # fig.suptitle('Time Series Transformation Comparison', fontsize=22, fontweight='bold', y=0.98)
    
    # 原始时间序列
    axes[0, 0].plot(time_steps, data, 'b-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Original Random Walk Time Series', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 保号对数变换 SLOG1P
    slog1p_data = slog1p_transform(data)
    axes[0, 1].plot(time_steps, slog1p_data, 'darkgreen', linewidth=1, alpha=0.8)
    axes[0, 1].set_title('Signed Log1p Transform\nsign(x) * log1p(1 + |x|)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 双曲正弦反函数变换
    asinh_data = asinh_transform(data)
    axes[0, 2].plot(time_steps, asinh_data, 'r-', linewidth=1, alpha=0.8)
    axes[0, 2].set_title('Inverse Hyperbolic Sine Transform\narsinh(x)', fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 保号幂变换 SPOW05
    spow05_data = signed_pow05_transform(data)
    axes[0, 3].plot(time_steps, spow05_data, 'darkblue', linewidth=1, alpha=0.8)
    axes[0, 3].set_title('Signed Power 0.5 Transform\nsign(x) * |x|^0.5', fontsize=12)
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 偏移对数变换
    offset_log_data = offset_log_transform(data)
    axes[1, 0].plot(time_steps, offset_log_data, 'm-', linewidth=1, alpha=0.8)
    axes[1, 0].set_title('Offset Log Transform\nlog(x + offset)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 分段对数变换
    data_safe = data.copy()
    data_safe[np.abs(data_safe) < 1e-10] = 1e-10  # 将接近零的值设为小正数
    piecewise_log_data = piecewise_log_transform(data_safe)
    axes[1, 1].plot(time_steps, piecewise_log_data, 'c-', linewidth=1, alpha=0.8)
    axes[1, 1].set_title('Piecewise Log Transform\nlog(|x|) * sign(x)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # CUMSUM变换
    cumsum_data = cumsum_transform(data)
    axes[1, 2].plot(time_steps, cumsum_data, 'orange', linewidth=1, alpha=0.8)
    axes[1, 2].set_title('Cumulative Sum Transform\ncumsum(x)', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # DIFF变换
    diff_data = diff_transform(data)
    axes[1, 3].plot(time_steps, diff_data, 'brown', linewidth=1, alpha=0.8)
    axes[1, 3].set_title('Difference Transform\ndiff(x)', fontsize=12)
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # ROLLSUM变换
    rollsum_data = rollsum_transform(data)
    axes[2, 0].plot(time_steps, rollsum_data, 'purple', linewidth=1, alpha=0.8)
    axes[2, 0].set_title('Rolling Sum Transform\nrollsum(x)', fontsize=12)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 分段线性"铰链"基 HINGE_POS
    hinge_pos_data = hinge_pos_transform(data)
    axes[2, 1].plot(time_steps, hinge_pos_data, 'teal', linewidth=1, alpha=0.8)
    axes[2, 1].set_title('Hinge Positive Transform\nmax(x - median(x), 0)', fontsize=12)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 分段线性"铰链"基 HINGE_NEG
    hinge_neg_data = hinge_neg_transform(data)
    axes[2, 2].plot(time_steps, hinge_neg_data, 'navy', linewidth=1, alpha=0.8)
    axes[2, 2].set_title('Hinge Negative Transform\nmax(median(x) - x, 0)', fontsize=12)
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 回撤变换 DRAWDOWN
    drawdown_data = drawdown_transform(data)
    axes[2, 3].plot(time_steps, drawdown_data, 'crimson', linewidth=1, alpha=0.8)
    axes[2, 3].set_title('Drawdown Transform\nx - cummax(x)', fontsize=12)
    axes[2, 3].grid(True, alpha=0.3)
    axes[2, 3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 历史新高变换 CUMMAX
    cummax_data = cummax_transform(data)
    axes[3, 0].plot(time_steps, cummax_data, 'darkred', linewidth=1, alpha=0.8)
    axes[3, 0].set_title('Cumulative Maximum Transform\ncummax(x)', fontsize=12)
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 组合型：ASINH_DIFF
    asinh_diff_data = asinh_diff_transform(data)
    axes[3, 1].plot(time_steps, asinh_diff_data, 'darkorange', linewidth=1, alpha=0.8)
    axes[3, 1].set_title('Asinh Diff Transform\ndiff(asinh(x))', fontsize=12)
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 季节/周期差分 DIFF_LAG7
    diff_lag7_data = diff_lag7_transform(data)
    axes[3, 2].plot(time_steps, diff_lag7_data, 'darkviolet', linewidth=1, alpha=0.8)
    axes[3, 2].set_title('Diff Lag 7 Transform\nx_t - x_{t-7}', fontsize=12)
    axes[3, 2].grid(True, alpha=0.3)
    axes[3, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 形态学梯度 MORPH_GRAD_w5
    morph_grad_data = morph_gradient_w5_transform(data)
    axes[3, 3].plot(time_steps, morph_grad_data, 'indigo', linewidth=1, alpha=0.8)
    axes[3, 3].set_title('Morphological Gradient w5\nrolling_max - rolling_min', fontsize=12)
    axes[3, 3].grid(True, alpha=0.3)
    axes[3, 3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 对称卷积核 - 类Sobel边缘检测
    sobel_data = sobel_like_transform(data)
    axes[4, 0].plot(time_steps, sobel_data, 'darkslategray', linewidth=1, alpha=0.8)
    axes[4, 0].set_title('Sobel-like Transform\nconv([1,0,-1])', fontsize=12)
    axes[4, 0].grid(True, alpha=0.3)
    axes[4, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 正交核变换 - 使用Haar小波核
    orthogonal_data = orthogonal_kernel_transform(data)
    axes[4, 1].plot(time_steps, orthogonal_data, 'darkgoldenrod', linewidth=1, alpha=0.8)
    axes[4, 1].set_title('Orthogonal Kernel Transform\nconv([1/√2,-1/√2])', fontsize=12)
    axes[4, 1].grid(True, alpha=0.3)
    axes[4, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 矩阵轮廓
    mp_data = matrix_profile_transform(data)
    axes[4, 2].plot(time_steps[:len(mp_data)], mp_data, 'black', linewidth=1, alpha=0.8)
    axes[4, 2].set_title('Matrix Profile', fontsize=12)
    axes[4, 2].grid(True, alpha=0.3)
    axes[4, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 统计信息对比
#     axes[4, 3].axis('off')
#     stats_text = f"""
# Statistical Comparison:

# Original Data:
#   Mean: {np.mean(data):.3f}
#   Std: {np.std(data):.3f}
#   Min: {np.min(data):.3f}
#   Max: {np.max(data):.3f}

# Signed Log Transform:
#   Mean: {np.mean(signed_log_data):.3f}
#   Std: {np.std(signed_log_data):.3f}

# Asinh Transform:
#   Mean: {np.mean(asinh_data):.3f}
#   Std: {np.std(asinh_data):.3f}

# SLOG1P Transform:
#   Mean: {np.mean(slog1p_data):.3f}
#   Std: {np.std(slog1p_data):.3f}

# SPOW05 Transform:
#   Mean: {np.mean(spow05_data):.3f}
#   Std: {np.std(spow05_data):.3f}

# Hinge Pos Transform:
#   Mean: {np.mean(hinge_pos_data):.3f}
#   Std: {np.std(hinge_pos_data):.3f}

# Hinge Neg Transform:
#   Mean: {np.mean(hinge_neg_data):.3f}
#   Std: {np.std(hinge_neg_data):.3f}

# Drawdown Transform:
#   Mean: {np.mean(drawdown_data):.3f}
#   Std: {np.std(drawdown_data):.3f}

# Cummax Transform:
#   Mean: {np.mean(cummax_data):.3f}
#   Std: {np.std(cummax_data):.3f}
#     """
#     axes[4, 3].text(0.05, 0.95, stats_text, transform=axes[4, 3].transAxes, 
#                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 优化子图间距和布局 - 增加上下间距
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.7)
    
    # 设置所有子图的字体大小一致，取消x、y轴标签
    for i in range(5):
        for j in range(4):
            if not (i == 4 and j == 3):
                axes[i, j].set_title(axes[i, j].get_title(), fontsize=12)
                axes[i, j].tick_params(axis='both', which='major', labelsize=10)
                # 移除x轴和y轴标签
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel('')
    plt.savefig('transform_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("图表已保存为 'transform_comparison.png'")
    print("\n变换方法说明:")
    print("1. 符号对数变换: 保持符号，对绝对值进行对数变换")
    print("2. asinh变换: 类似对数但可处理负值，在金融分析中常用")
    print("3. 偏移对数变换: 通过偏移使所有值为正后进行对数变换")
    print("4. 分段对数变换: 对正负值分别进行对数变换")
    print("5. CUMSUM变换: 累积和变换，用于趋势分析")
    print("6. DIFF变换: 差分变换，用于去除趋势和平稳化")
    print("7. SLOG1P变换: 保号对数变换，对绝对值进行log1p并保留符号")
    print("8. SPOW05变换: 保号幂变换，对绝对值进行0.5次幂并保留符号")
    print("9. HINGE变换: 分段线性铰链基，构造(x-k)+与(k-x)+")
    print("10. DRAWDOWN/CUMMAX变换: 回撤与新高，路径依赖特征")
    print("11. ASINH_DIFF变换: 组合型变换，先非线性再差分")
    print("12. DIFF_LAG7变换: 季节/周期差分，固定滞后差分")
    print("13. MORPH_GRAD_w5变换: 形态学梯度，窗口内最大最小值之差")

# def compare_transform_properties():
#     """比较不同变换的数学性质"""
#     # 生成测试数据
#     x = np.linspace(-10, 10, 1000)
#     x = x[x != 0]  # 移除零值以避免log(0)
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     fig.suptitle('Mathematical Properties of Transform Functions', fontsize=16, fontweight='bold')
    
#     # 符号对数变换
#     y1 = signed_log_transform(x)
#     axes[0, 0].plot(x, y1, 'g-', linewidth=2)
#     axes[0, 0].set_title('Signed Log Transform')
#     axes[0, 0].set_xlabel('Input Value x')
#     axes[0, 0].set_ylabel('sign(x) * log(1 + |x|)')
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
#     axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
#     # asinh变换
#     y2 = asinh_transform(x)
#     axes[0, 1].plot(x, y2, 'r-', linewidth=2)
#     axes[0, 1].set_title('Asinh Transform')
#     axes[0, 1].set_xlabel('Input Value x')
#     axes[0, 1].set_ylabel('asinh(x)')
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
#     axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
#     # 偏移对数变换（仅正值部分）
#     x_pos = x[x > 0]
#     offset = 1
#     y3 = np.log(x_pos + offset)
#     axes[1, 0].plot(x_pos, y3, 'm-', linewidth=2)
#     axes[1, 0].set_title('Offset Log Transform (Positive Only)')
#     axes[1, 0].set_xlabel('Input Value x')
#     axes[1, 0].set_ylabel('log(x + offset)')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # 分段对数变换
#     x_safe = x.copy()
#     x_safe[np.abs(x_safe) < 1e-10] = 1e-10
#     y4 = piecewise_log_transform(x_safe)
#     axes[1, 1].plot(x_safe, y4, 'c-', linewidth=2)
#     axes[1, 1].set_title('Piecewise Log Transform')
#     axes[1, 1].set_xlabel('Input Value x')
#     axes[1, 1].set_ylabel('log(|x|) * sign(x)')
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
#     axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('transform_properties.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("变换性质图已保存为 'transform_properties.png'")

if __name__ == "__main__":
    print("正在生成时间序列变换对比图...")
    plot_transformations()
    
    # print("\n正在生成变换函数性质对比图...")
    # compare_transform_properties()
    
    print("\n所有图表生成完成！")