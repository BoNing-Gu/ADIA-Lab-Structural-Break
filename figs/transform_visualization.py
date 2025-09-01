import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

def plot_transformations():
    """绘制原始时间序列和各种变换的对比图"""
    # 生成随机游走数据
    data = generate_random_walk(1000)
    time_steps = np.arange(len(data))
    
    # 创建子图 - 增加到3行3列以容纳更多变换
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Time Series Transformation Comparison', fontsize=16, fontweight='bold')
    
    # 原始时间序列
    axes[0, 0].plot(time_steps, data, 'b-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Original Random Walk Time Series', fontsize=14)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 符号对数变换
    signed_log_data = signed_log_transform(data)
    axes[0, 1].plot(time_steps, signed_log_data, 'g-', linewidth=1, alpha=0.8)
    axes[0, 1].set_title('Signed Log Transform\nsign(x) * log(1 + |x|)', fontsize=14)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Transformed Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 双曲正弦反函数变换
    asinh_data = asinh_transform(data)
    axes[0, 2].plot(time_steps, asinh_data, 'r-', linewidth=1, alpha=0.8)
    axes[0, 2].set_title('Inverse Hyperbolic Sine Transform\narsinh(x)', fontsize=14)
    axes[0, 2].set_xlabel('Time Steps')
    axes[0, 2].set_ylabel('Transformed Value')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 偏移对数变换
    offset_log_data = offset_log_transform(data)
    axes[1, 0].plot(time_steps, offset_log_data, 'm-', linewidth=1, alpha=0.8)
    axes[1, 0].set_title('Offset Log Transform\nlog(x + offset)', fontsize=14)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Transformed Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 分段对数变换
    # 为了避免log(0)的问题，我们需要处理接近零的值
    data_safe = data.copy()
    data_safe[np.abs(data_safe) < 1e-10] = 1e-10  # 将接近零的值设为小正数
    piecewise_log_data = piecewise_log_transform(data_safe)
    axes[1, 1].plot(time_steps, piecewise_log_data, 'c-', linewidth=1, alpha=0.8)
    axes[1, 1].set_title('Piecewise Log Transform\nlog(|x|) * sign(x)', fontsize=14)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Transformed Value')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # CUMSUM变换
    cumsum_data = cumsum_transform(data)
    axes[1, 2].plot(time_steps, cumsum_data, 'orange', linewidth=1, alpha=0.8)
    axes[1, 2].set_title('Cumulative Sum Transform\ncumsum(x)', fontsize=14)
    axes[1, 2].set_xlabel('Time Steps')
    axes[1, 2].set_ylabel('Transformed Value')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # DIFF变换
    diff_data = diff_transform(data)
    axes[2, 0].plot(time_steps, diff_data, 'brown', linewidth=1, alpha=0.8)
    axes[2, 0].set_title('Difference Transform\ndiff(x)', fontsize=14)
    axes[2, 0].set_xlabel('Time Steps')
    axes[2, 0].set_ylabel('Transformed Value')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 统计信息对比
    axes[2, 1].axis('off')
    stats_text = f"""
Statistical Comparison:

Original Data:
  Mean: {np.mean(data):.3f}
  Std: {np.std(data):.3f}
  Min: {np.min(data):.3f}
  Max: {np.max(data):.3f}

Signed Log Transform:
  Mean: {np.mean(signed_log_data):.3f}
  Std: {np.std(signed_log_data):.3f}

Asinh Transform:
  Mean: {np.mean(asinh_data):.3f}
  Std: {np.std(asinh_data):.3f}

Offset Log Transform:
  Mean: {np.mean(offset_log_data):.3f}
  Std: {np.std(offset_log_data):.3f}

Piecewise Log Transform:
  Mean: {np.mean(piecewise_log_data):.3f}
  Std: {np.std(piecewise_log_data):.3f}

Cumsum Transform:
  Mean: {np.mean(cumsum_data):.3f}
  Std: {np.std(cumsum_data):.3f}

Diff Transform:
  Mean: {np.mean(diff_data):.3f}
  Std: {np.std(diff_data):.3f}
    """
    axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 隐藏多余的子图
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('transform_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 'transform_comparison.png'")
    print("\n变换方法说明:")
    print("1. 符号对数变换: 保持符号，对绝对值进行对数变换")
    print("2. asinh变换: 类似对数但可处理负值，在金融分析中常用")
    print("3. 偏移对数变换: 通过偏移使所有值为正后进行对数变换")
    print("4. 分段对数变换: 对正负值分别进行对数变换")
    print("5. CUMSUM变换: 累积和变换，用于趋势分析")
    print("6. DIFF变换: 差分变换，用于去除趋势和平稳化")

def compare_transform_properties():
    """比较不同变换的数学性质"""
    # 生成测试数据
    x = np.linspace(-10, 10, 1000)
    x = x[x != 0]  # 移除零值以避免log(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mathematical Properties of Transform Functions', fontsize=16, fontweight='bold')
    
    # 符号对数变换
    y1 = signed_log_transform(x)
    axes[0, 0].plot(x, y1, 'g-', linewidth=2)
    axes[0, 0].set_title('Signed Log Transform')
    axes[0, 0].set_xlabel('Input Value x')
    axes[0, 0].set_ylabel('sign(x) * log(1 + |x|)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # asinh变换
    y2 = asinh_transform(x)
    axes[0, 1].plot(x, y2, 'r-', linewidth=2)
    axes[0, 1].set_title('Asinh Transform')
    axes[0, 1].set_xlabel('Input Value x')
    axes[0, 1].set_ylabel('asinh(x)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 偏移对数变换（仅正值部分）
    x_pos = x[x > 0]
    offset = 1
    y3 = np.log(x_pos + offset)
    axes[1, 0].plot(x_pos, y3, 'm-', linewidth=2)
    axes[1, 0].set_title('Offset Log Transform (Positive Only)')
    axes[1, 0].set_xlabel('Input Value x')
    axes[1, 0].set_ylabel('log(x + offset)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 分段对数变换
    x_safe = x.copy()
    x_safe[np.abs(x_safe) < 1e-10] = 1e-10
    y4 = piecewise_log_transform(x_safe)
    axes[1, 1].plot(x_safe, y4, 'c-', linewidth=2)
    axes[1, 1].set_title('Piecewise Log Transform')
    axes[1, 1].set_xlabel('Input Value x')
    axes[1, 1].set_ylabel('log(|x|) * sign(x)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transform_properties.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("变换性质图已保存为 'transform_properties.png'")

if __name__ == "__main__":
    print("正在生成时间序列变换对比图...")
    plot_transformations()
    
    print("\n正在生成变换函数性质对比图...")
    compare_transform_properties()
    
    print("\n所有图表生成完成！")