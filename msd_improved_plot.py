import matplotlib.pyplot as plt

# ==========================================
# 1. MSD 数据集核心数据
# ==========================================
# 方法名称
methods = ['MedSAM\n(Zero-shot)', 'DINOv2+Decoder', 'DINOv3+Decoder', 'Ours: 2.5D DINOv3\n+Gating']
# X轴：FLOPs (G)
flops = [369.01, 9.05, 5.89, 5.89]
# Y轴：mDice (%)
mdice = [60.34, 73.73, 75.01, 77.05]
# 气泡大小 (参考 AMOS 图表比例)
sizes = [1500, 450, 450, 750] 
# 颜色配置 (复刻 AMOS 风格)
colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#d62728']

# ==========================================
# 2. 绘图初始化
# ==========================================
plt.figure(figsize=(10, 7), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid')

# 绘制气泡
plt.scatter(flops, mdice, s=sizes, c=colors, alpha=0.75, edgecolors='black', linewidth=1.5, zorder=3)

# 坐标轴设置
plt.xscale('symlog')
plt.xlim(4, 550)    # 留出左右间距
plt.ylim(55, 85)    # MSD 的数据区间在 60-77 之间，设置 55-85 最美观

# ==========================================
# 3. 复刻 AMOS 风格的标注布局
# ==========================================

# (1) MedSAM - 放在大圆圈上方，HA居中
plt.annotate(methods[0], (flops[0], mdice[0]), xytext=(0, 25), 
             textcoords="offset points", ha='center', fontsize=10)

# (2) DINOv2 - 放在气泡右侧 (ha='left')
plt.annotate(methods[1], (flops[1], mdice[1]), xytext=(15, 0), 
             textcoords="offset points", ha='left', va='center', fontsize=10)

# (3) DINOv3 - 放在气泡右侧
plt.annotate(methods[2], (flops[2], mdice[2]), xytext=(15, 0), 
             textcoords="offset points", ha='left', va='center', fontsize=10)

# (4) Ours - 放在气泡右侧，加粗显眼
plt.annotate(methods[3], (flops[3], mdice[3]), xytext=(18, 5), 
             textcoords="offset points", ha='left', va='bottom', 
             fontsize=11, fontweight='bold', color='#d62728')

# (5) 红色汇总框 - 放置在左上方
text_str = 'Best Performance: 77.05%\nTrainable Params: 2.77M\nFLOPs: 5.89G'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#d62728', lw=1.2)
plt.text(4.5, 83.5, text_str, fontsize=10.5, verticalalignment='top', 
         bbox=props, color='#d62728', fontweight='bold', zorder=5)

# (6) Ideal Zone 绿色箭头 - 放置在最顶部
plt.annotate('Ideal Zone\n(High Acc, Low Cost)', xy=(4.5, 84.5), xytext=(15, 83.5),
             arrowprops=dict(facecolor='darkseagreen', shrink=0.05, width=1, headwidth=8, alpha=0.8),
             fontsize=10, color='green', fontweight='bold', ha='left', va='top')

# ==========================================
# 4. 图表细节美化
# ==========================================
plt.xlabel('Computational Cost (GFLOPs, Log Scale) → Lower is Better', fontsize=11, fontweight='bold')
plt.ylabel('mDice (%) → Higher is Better', fontsize=11, fontweight='bold')
plt.title('Performance vs. Efficiency on MSD Dataset', fontsize=14, fontweight='bold', pad=20)

# 精准设置 X 轴刻度
plt.xticks([5.89, 9.05, 100, 369.01], ['5.89', '9.05', '100', '369.01'])
plt.grid(True, which="both", ls="-", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('MSD_Performance_Efficiency.png')
plt.show()

print("✅ MSD 气泡图已生成，布局已参考 AMOS22 样式优化。")
