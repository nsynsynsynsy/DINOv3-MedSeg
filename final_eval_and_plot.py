import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt

# 导入你的组件
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPT

# ==========================================
# 配置区域 (在这里切换模式)
# ==========================================
# 模式选择：
# 0 = 全部跑完并画【对比柱状图】 (最推荐，用于论文结果展示)
# 1 = 只测 Exp1_2D (Baseline) 并画雷达图
# 2 = 只测 Exp2_2.5D_NoGate 并画雷达图
# 3 = 只测 Exp3_Ours_Gate (Ours) 并画雷达图
EXPERIMENT_MODE = 0 

DATA_DIR = "/root/autodl-tmp"
REPO_DIR = "./dinov3"
DINO_CKPT = "./web_pth/dinov3_vits16_pretrain.pth"
NUM_CLASSES = 11
SAVE_DIR = "./runs/Analysis_Results" 
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Background", "Spleen", "R.Kidney", "L.Kidney", "Gallbladder", 
               "Esophagus", "Liver", "Stomach", "Aorta", "IVC", "Pancreas"]

# 完整权重列表 (请核对路径是否与你文件系统一致)
FULL_CKPT_LIST = {
    "Exp1_2D": "runs/Exp1_2D_Baseline/ckpts/best_ep088_mDice0.8158_mIoU0.7101.pth",
    "Exp2_2.5D_NoGate": "runs/Exp2_25D_NoGate/ckpts/best_ep085_mDice0.8246_mIoU0.7217.pth",
    "Exp3_Ours_Gate": "runs/Exp3_25D_WithGate_Ours/ckpts/best_ep084_mDice0.8544_mIoU0.7629.pth"
}

# 根据设定选择要跑的实验
if EXPERIMENT_MODE == 0:
    CKPT_LIST = FULL_CKPT_LIST
elif EXPERIMENT_MODE in [1, 2, 3]:
    keys = list(FULL_CKPT_LIST.keys())
    CKPT_LIST = {keys[EXPERIMENT_MODE-1]: FULL_CKPT_LIST[keys[EXPERIMENT_MODE-1]]}
else:
    raise ValueError("EXPERIMENT_MODE 必须是 0, 1, 2, 3 中的一个！")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 评价指标计算 (HD95 & Dice)
# ==========================================
def hd95_simple(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not np.any(pred) or not np.any(gt):
        return 100.0
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    dist_pred_to_gt = dt_gt[pred]
    dist_gt_to_pred = dt_pred[gt]
    return np.percentile(np.concatenate([dist_pred_to_gt, dist_gt_to_pred]), 95)

def calculate_metrics(pred, gt, num_classes):
    dice_list, hd95_list = [], []
    for c in range(num_classes):
        p, g = (pred == c), (gt == c)
        if g.sum() == 0:
            dice_list.append(np.nan)
            hd95_list.append(np.nan)
        else:
            intersection = np.logical_and(p, g).sum()
            union = p.sum() + g.sum()
            dice_list.append((2. * intersection) / union if union > 0 else 0.0)
            hd95_list.append(hd95_simple(p, g))
    return dice_list, hd95_list


# ==========================================
# 绘图函数 (论文级视觉优化)
# ==========================================
def plot_comparison_bar(df):
    """画多模型对比柱状图 (添加分隔线、Average、提高Y轴、打星号)"""
    # 确保不画 Background
    if "Background" in df.index:
        df = df.drop("Background")
        
    # 把 'Mean (mDice)' 改名为 'Average'
    if 'Mean (mDice)' in df.index:
        df = df.rename(index={'Mean (mDice)': 'Average'})
    
    categories = df.index.tolist()
    plt.figure(figsize=(15, 6))
    x = np.arange(len(categories))
    width = 0.25
    
    # 画柱子（把返回的对象存起来，方便后面找高度打星号）
    bar1 = plt.bar(x - width, df["Exp1_2D"], width, label='Exp1: 2D Baseline', color='#8FB0D3', edgecolor='gray', linewidth=0.5)
    bar2 = plt.bar(x, df["Exp2_2.5D_NoGate"], width, label='Exp2: 2.5D (No Gate)', color='#E5A9A9', edgecolor='gray', linewidth=0.5)
    bar3 = plt.bar(x + width, df["Exp3_Ours_Gate"], width, label='Exp3: Ours (Gating)', color='#7BC87B', edgecolor='gray', linewidth=0.5)

    plt.ylabel('Dice Score', fontsize=14, weight='bold')
    plt.title('Performance Comparison Across Different Organs', fontsize=16, weight='bold')
    plt.xticks(x, categories, rotation=35, fontsize=11, weight='bold')
    
    # ✅ 需求满足：调高了 Y 轴上限到 1.05，防止星号被顶破天花板
    plt.ylim(0.4, 1.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # ✅ 需求满足：添加分隔线，区分个体器官和 Average
    if 'Average' in categories:
        avg_index = categories.index('Average')
        plt.axvline(x=avg_index - 0.5, color='gray', linestyle='-.', alpha=0.8, linewidth=1.5)

    # ✅ 需求满足：在特定器官的 Ours (Exp3) 柱子上打星号 (*)
    # 你可以在这个列表里修改你想打星号的器官名称
    highlight_organs = ['Gallbladder', 'Stomach', 'Pancreas'] 
    
    for i, category in enumerate(categories):
        if category in highlight_organs:
            # 获取当前 Exp3 柱子的高度 (即 Dice 值)
            height = bar3[i].get_height()
            # 在绿色柱子正上方 +0.01 的位置画一个红色星号
            plt.text(x[i] + width, height + 0.01, '*', ha='center', va='bottom', fontsize=18, color='red', weight='bold')

    # 图例放在左上角，防止挡住右边的 Average 柱子
    plt.legend(fontsize=11, loc='upper left', frameon=True)
    
    # 移除多余边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Comparison_Dice_Paper_Ready.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 【终极论文版】带Average和星号的对比图已保存至: {save_path}")

def plot_single_radar(df, exp_name):
    """画单个模型的雷达图"""
    if "Background" in df.index:
        df = df.drop("Background")
    categories = df.index.tolist()
    values = df[exp_name].values.tolist()
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
    ax.fill(angles, values, '#3498db', alpha=0.3)
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_ylim(0, 1)
    plt.title(f'{exp_name} Segmentation Performance\n', size=16, weight='bold')
    
    save_path = os.path.join(SAVE_DIR, f"{exp_name}_radar.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 雷达图已保存至: {save_path}")

# ==========================================
# 主运行流程
# ==========================================
def load_model(name, path):
    print(f"\n[Loading] {name} from {path}")
    backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=DINO_CKPT)
    use_gate = True if "Gate" in name and "NoGate" not in name else False
    model = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=use_gate).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def run_evaluation():
    # 数据加载 (测试最后100张切片作为采样)
    dataset = FolderDataset(root=DATA_DIR, transform=ResizeAndNormalize(), input_type="2.5d")
    indices = list(range(len(dataset)))[-100:] 
    val_loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1)

    all_results_dice, all_results_hd95 = {}, {}

    for exp_name, ckpt_path in CKPT_LIST.items():
        model = load_model(exp_name, ckpt_path)
        exp_dice, exp_hd95 = [], []

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc=f"Evaluating {exp_name}"):
                inputs = inputs.to(device)
                # Exp1 特殊处理：将2.5D降级为2D测试
                if "Exp1" in exp_name:
                    mid = inputs[:, 1:2, :, :]
                    inputs_in = torch.cat([mid, mid, mid], dim=1)
                else:
                    inputs_in = inputs
                
                output = model(inputs_in)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                gt = targets[0].cpu().numpy()

                d, h = calculate_metrics(pred, gt, NUM_CLASSES)
                exp_dice.append(d)
                exp_hd95.append(h)

        all_results_dice[exp_name] = np.nanmean(np.array(exp_dice), axis=0)
        all_results_hd95[exp_name] = np.nanmean(np.array(exp_hd95), axis=0)

     # 整理数据
    df_dice = pd.DataFrame(all_results_dice, index=CLASS_NAMES)
    df_hd95 = pd.DataFrame(all_results_hd95, index=CLASS_NAMES)

    # ==========================================
    # 🚨 核心修改：剔除背景，并计算真实的平均分
    # ==========================================
    if "Background" in df_dice.index:
        df_dice = df_dice.drop("Background")
        df_hd95 = df_hd95.drop("Background")

    # 增加一行 'Mean' (所有器官的平均值)
    df_dice.loc['Mean (mDice)'] = df_dice.mean()
    df_hd95.loc['Mean (mHD95)'] = df_hd95.mean()

    print("\n" + "="*50)
    print("🏆 DICE SCORE REPORT (Higher is better) 🏆")
    print("="*50)
    print(df_dice.round(4))
    
    print("\n" + "="*50)
    print("📉 HD95 REPORT (Lower is better) 📉")
    print("="*50)
    print(df_hd95.round(2))
    print("==================================================")

    # 保存与绘图
    if EXPERIMENT_MODE == 0:
        # ✅ 把 .drop() 删掉，直接把包含 Average 的完整 df_dice 传进去！
        plot_comparison_bar(df_dice)  
        df_dice.to_csv(os.path.join(SAVE_DIR, "All_Models_Dice.csv"))
        df_hd95.to_csv(os.path.join(SAVE_DIR, "All_Models_HD95.csv"))
    else:
        current_exp = list(CKPT_LIST.keys())[0]
        plot_single_radar(df_dice.drop('Mean (mDice)'), current_exp)
        df_dice.to_csv(os.path.join(SAVE_DIR, f"{current_exp}_Results.csv"))

if __name__ == "__main__":
    run_evaluation()


