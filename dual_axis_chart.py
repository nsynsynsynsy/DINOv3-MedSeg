import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 准备数据
# ==========================================
methods = ['MedSAM\n(Zero-shot)', 'DINOv2\n+ Decoder', 'DINOv3\n+ Decoder', 'Ours:\n2.5D DINOv3']
flops = [369.01, 18.16, 5.91, 5.91]
mdice = [64.68, 78.43, 81.58, 85.44]
trainable_params = "2.78M" # 专门标注给 Ours

# ==========================================
# 2. 绘图设置
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid') # 使用干净的网格背景

# --- 绘制左轴 (ax1): FLOPs 柱状图 ---
colors = ['#ced4da', '#adb5bd', '#6c757d', '#d62728'] # 前三个用灰色系，Ours用红色
bars = ax1.bar(methods, flops, color=colors, alpha=0.8, width=0.5, label='FLOPs (G)')

# 优化左轴：因为MedSAM太高，我们用对数坐标能看得更清楚对比
ax1.set_yscale('log') 
ax1.set_ylabel('Computational Cost (FLOPs in G) → Lower is Better', fontsize=12, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelsize=11)
ax1.set_ylim(1, 1000) # 设置范围让图形不顶格

# --- 绘制右轴 (ax2): mDice 折线图 ---
ax2 = ax1.twinx() # 关键：创建共享X轴的第二个Y轴
line = ax2.plot(methods, mdice, color='#1f77b4', marker='s', markersize=10, 
                linewidth=3, label='mDice (%)', linestyle='-', zorder=10)

ax2.set_ylabel('Segmentation Performance (mDice %) → Higher is Better', fontsize=12, fontweight='bold', color='#1f77b4')
ax2.set_ylim(60, 90) # 针对你的数据范围，设在60-90%最能体现涨点
ax2.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)

# ==========================================
# 3. 细节标注与美化
# ==========================================
# 给折线点添加数值标签
for i, txt in enumerate(mdice):
    ax2.annotate(f'{txt}%', (methods[i], mdice[i]), textcoords="offset points", 
                 xytext=(0, 12), ha='center', fontsize=11, color='#1f77b4', fontweight='bold')

# 专门在 Ours 的柱子上方标注“只用了 2.78M 参数”
ax1.annotate(f'Trainable:\n{trainable_params}', xy=(3, 5.91), xytext=(0, 15), 
             textcoords="offset points", ha='center', va='bottom',
             fontsize=10, color='#d62728', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#d62728'))

# 标题与网格
plt.title('Comparison of Efficiency and Performance', fontsize=15, fontweight='bold', pad=20)
ax1.grid(True, axis='y', ls='--', alpha=0.7)
ax2.grid(False) # 关闭右轴网格，避免重叠混乱

# 合并两个轴的图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center', frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig('dual_axis_chart.png')
print("✅ 双Y轴柱线图已生成：dual_axis_chart.png")
