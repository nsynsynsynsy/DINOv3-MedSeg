import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 准备数据 (MSD 数据集)
# ==========================================
methods = ['MedSAM\n(Zero-shot)', 'DINOv2\n+ Decoder', 'DINOv3\n+ Decoder', 'Ours:\n2.5D DINOv3']
flops = [369.01, 9.05, 5.89, 5.89]
mdice = [60.34, 73.73, 75.01, 77.05]
trainable_params = "2.77M" # MSD 的 Ours 参数量

# ==========================================
# 2. 绘图设置
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid') # 使用干净的网格背景

# --- 绘制左轴 (ax1): FLOPs 柱状图 ---
colors = ['#ced4da', '#adb5bd', '#6c757d', '#d62728'] # 前三个用灰色系，Ours用红色
bars = ax1.bar(methods, flops, color=colors, alpha=0.8, width=0.5, label='FLOPs (G)')

# 优化左轴：对数坐标
ax1.set_yscale('log') 
ax1.set_ylabel('Computational Cost (FLOPs in G) → Lower is Better', fontsize=12, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelsize=11)
ax1.set_ylim(1, 1000) # 保持不变

# --- 绘制右轴 (ax2): mDice 折线图 ---
ax2 = ax1.twinx() # 创建共享X轴的第二个Y轴
line = ax2.plot(methods, mdice, color='#1f77b4', marker='s', markersize=10, 
                linewidth=3, label='mDice (%)', linestyle='-', zorder=10)

ax2.set_ylabel('Segmentation Performance (mDice %) → Higher is Better', fontsize=12, fontweight='bold', color='#1f77b4')
# 【关键修改】针对 MSD 数据范围 (60.34 - 77.05)，调整 Y轴范围，使折线居中好看
ax2.set_ylim(55, 85) 
ax2.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)

# ==========================================
# 3. 细节标注与美化
# ==========================================
# 给折线点添加数值标签
for i, txt in enumerate(mdice):
    ax2.annotate(f'{txt}%', (methods[i], mdice[i]), textcoords="offset points", 
                 xytext=(0, 12), ha='center', fontsize=11, color='#1f77b4', fontweight='bold')

# 专门在 Ours 的柱子上方标注“只用了 2.77M 参数”
ax1.annotate(f'Trainable:\n{trainable_params}', xy=(3, 5.89), xytext=(0, 15), 
             textcoords="offset points", ha='center', va='bottom',
             fontsize=10, color='#d62728', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#d62728'))

# 标题与网格 (已经删去括号和MSD)
plt.title('Comparison of Efficiency and Performance', fontsize=15, fontweight='bold', pad=20)
ax1.grid(True, axis='y', ls='--', alpha=0.7)
ax2.grid(False) # 关闭右轴网格，避免重叠混乱

# 合并两个轴的图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center', frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig('msd_dual_axis_chart_new_title.png')
print("✅ 图片已生成：msd_dual_axis_chart_new_title.png")
