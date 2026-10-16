import matplotlib.pyplot as plt

# ==========================================
# 核心数据
# ==========================================
methods = ['MedSAM\n(Zero-shot)', 'DINOv2+Decoder', 'DINOv3+Decoder', 'Ours: 2.5D DINOv3\n+Gating']
flops = [369.01, 18.16, 5.91, 5.91]
mdice = [64.68, 78.43, 81.58, 85.44]
# 重点：参数量对比。MedSAM固定大，Ours标注出来
params = [1000, 300, 300, 450] 
colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#d62728']

# ==========================================
# 绘图逻辑
# ==========================================
plt.figure(figsize=(10, 7), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid')

# 绘制散点
plt.scatter(flops, mdice, s=params, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

# 坐标轴设置
plt.xscale('symlog')
plt.xlim(4, 500)  # 稍微放宽边界，防止文字出界
plt.ylim(60, 95)  # 提高上限，给标注留出空间

# 逐个添加标注，手动避开重叠
# 1. MedSAM
plt.annotate(methods[0], (flops[0], mdice[0]), xytext=(0, 15), textcoords="offset points", ha='center', fontsize=10)
# 2. DINOv2
plt.annotate(methods[1], (flops[1], mdice[1]), xytext=(10, 5), textcoords="offset points", ha='left', fontsize=10)
# 3. DINOv3
plt.annotate(methods[2], (flops[2], mdice[2]), xytext=(15, -5), textcoords="offset points", ha='left', fontsize=10)
# 4. Ours (高亮标注)
plt.annotate(methods[3], (flops[3], mdice[3]), xytext=(15, 10), textcoords="offset points", 
             ha='left', fontsize=12, fontweight='bold', color='#d62728')

# 在左上角做一个漂亮的文本框，专门写结果
text_str = 'Best Performance: 85.44%\nTrainable Params: 2.78M\nFLOPs: 5.91G'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#d62728')
plt.text(5, 92, text_str, fontsize=11, verticalalignment='top', bbox=props, color='#d62728', fontweight='bold')

# 辅助箭头：指向理想区域
plt.annotate('Ideal Zone\n(High Acc, Low Cost)', xy=(5, 94), xytext=(20, 92),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=8, alpha=0.4),
             fontsize=10, color='green', fontweight='bold', ha='left')

# 标签和标题
plt.xlabel('Computational Cost (GFLOPs, Log Scale) → Lower is Better', fontsize=12, fontweight='bold')
plt.ylabel('mDice (%) → Higher is Better', fontsize=12, fontweight='bold')
plt.title('Performance vs. Efficiency (PEFT Comparison)', fontsize=14, fontweight='bold', pad=20)

plt.xticks([5.91, 18.16, 100, 369.01], ['5.91', '18.16', '100', '369.01'])
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig('improved_plot.png')
print("✅ 优化后的图表已保存为 improved_plot.png")
