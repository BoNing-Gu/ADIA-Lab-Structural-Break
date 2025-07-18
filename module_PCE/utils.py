import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_debug_zseq(z, period, key_padding_mask):
    # 取第0个样本、第0个变量的时间序列
    z_seq = z[0, 0, :].detach().cpu().numpy()            # [seq_len]
    period_seq = period[0, :].detach().cpu().numpy()     # [seq_len], values 0 or 1
    mask_seq = key_padding_mask[0, :].detach().cpu().numpy()  # [seq_len], bool

    seq_len = len(z_seq)
    t = list(range(seq_len))

    plt.figure(figsize=(8, 4))

    # 画 z 序列
    plt.plot(t, z_seq, label='z[0,0,:]', color='blue')

    # 画 period 标签作为 0/1 底色
    plt.fill_between(t, min(z_seq), max(z_seq), where=period_seq > 0.5, color='green', alpha=0.2, label='period = 1')

    # 显示 key_padding_mask 为 True 的位置
    for idx, is_pad in enumerate(mask_seq):
        if is_pad:
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.3)

    plt.title('z[0,0,:] with period and key_padding_mask')
    plt.xlabel('Time step')
    plt.ylabel('z value')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_debug_attn(tensor: torch.Tensor, title: str = "Attention Map"):
    """
    可视化 bs=0, head=0 时的 tensor 矩阵（适用于 q/k/v/output/attn_weights）
    
    Args:
        tensor (torch.Tensor): shape = [bs, n_heads, *, *]
        title (str): 热力图标题，通常写明当前变量名
    """
    # 截取 bs=0, n_heads=0 的矩阵
    data = tensor[0, 0].detach().cpu().numpy()  # shape = [*, *]

    plt.figure(figsize=(4, 4))
    sns.heatmap(data, cmap="viridis", annot=False)
    plt.title(title)
    plt.xlabel("Key/Dimension")
    plt.ylabel("Query/Time Step")
    plt.tight_layout()
    plt.show()
