import matplotlib.pyplot as plt

# 数据
submissions = [7, 8, 10, 12, 17, 18, 19]
cv_scores = [0.7875, 0.8739, 0.8883, 0.8909, 0.9059, 0.8918, 0.8945]
lb_scores = [0.7812, 0.8646, 0.8756, 0.8776, 0.8802, 0.8745, 0.8834]
gaps = [cv - lb for cv, lb in zip(cv_scores, lb_scores)]
magic_labels = ["no", "by interaction", "by interaction", "by interaction", "by feat", "by feat", "by interaction"]

# 颜色映射
color_map = {"no": "red", "by interaction": "blue", "by feat": "green"}

plt.figure(figsize=(8,6))

# 按magic类别画折线
for magic in set(magic_labels):
    x = [cv for cv, m in zip(cv_scores, magic_labels) if m == magic]
    y = [gap for gap, m in zip(gaps, magic_labels) if m == magic]
    subs = [s for s, m in zip(submissions, magic_labels) if m == magic]
    plt.plot(x, y, marker="o", linestyle="-", color=color_map[magic], label=f"Magic={magic}")
    for cv, gap, sub in zip(x, y, subs):
        plt.text(cv, gap+0.001, f"#{sub}", ha="center", fontsize=8)

plt.title("CV vs. (CV - LB) Gap", fontsize=14)
plt.xlabel("CV Score", fontsize=12)
plt.ylabel("CV - LB", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.savefig('cvlb_gap.png', dpi=600)
plt.show()


plt.figure(figsize=(9,6))

for sub, cv, lb, magic in zip(submissions, cv_scores, lb_scores, magic_labels):
    # 画线段
    plt.plot([cv, cv], [lb, cv], color=color_map[magic], linewidth=2)
    # 在上端标注CV和提交号
    plt.scatter(cv, cv, color=color_map[magic], s=50, label=f"{magic}" if f"{magic}" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(cv, cv+0.003, f"#{sub}", ha="center", fontsize=8)
    # 在下端标注LB
    plt.scatter(cv, lb, color=color_map[magic], marker="s", s=40)
    plt.text(cv, lb-0.005, f"{lb:.3f}", ha="center", fontsize=7, color="gray")

plt.xlabel("CV Score", fontsize=12)
plt.ylabel("Score (CV & LB)", fontsize=12)
plt.title("CV vs LB per Submission (Gap shown by line length)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Magic Type")
plt.tight_layout()
plt.savefig("cvlb_segment.png", dpi=600)
plt.show()