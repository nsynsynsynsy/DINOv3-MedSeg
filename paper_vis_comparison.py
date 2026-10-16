import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你原本的组件
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPT

# ==========================================
# 配置区域
# ==========================================
DATA_DIR = "/root/autodl-tmp"
REPO_DIR = "./dinov3"
DINO_CKPT = "./web_pth/dinov3_vits16_pretrain.pth"
NUM_CLASSES = 11
SAVE_DIR = "./runs/Paper_Visualizations" # 结果保存路径

# 🚨 请务必确认以下权重文件名与你 runs 文件夹下的实际文件名完全一致
CKPT_1 = "runs/Exp1_2D_Baseline/ckpts/best_ep088_mDice0.8158_mIoU0.7101.pth"
CKPT_2 = "runs/Exp2_25D_NoGate/ckpts/best_ep085_mDice0.8246_mIoU0.7217.pth"
CKPT_3 = "runs/Exp3_25D_WithGate_Ours/ckpts/best_ep084_mDice0.8544_mIoU0.7629.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_all_models():
    print("正在初始化并加载 3 组实验的模型权重...")
    # 加载 Backbone
    backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=DINO_CKPT)
    
    # 实例化三个模型
    model1 = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=False).to(device)
    model2 = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=False).to(device)
    model3 = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=True).to(device)

    # 加载权重
    model1.load_state_dict(torch.load(CKPT_1, map_location=device))
    model2.load_state_dict(torch.load(CKPT_2, map_location=device))
    model3.load_state_dict(torch.load(CKPT_3, map_location=device))

    model1.eval()
    model2.eval()
    model3.eval()
    return model1, model2, model3

def denormalize(tensor):
    """反归一化，修复设备冲突 Bug"""
    # 确保 mean 和 std 与输入 tensor 在同一设备 (CPU/GPU)
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    res = tensor * std + mean
    return res.clamp(0, 1)

def run_visualization():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model1, model2, model3 = load_all_models()

    # 使用 2.5d 加载数据以便对比
    dataset = FolderDataset(root=DATA_DIR, transform=ResizeAndNormalize(), input_type="2.5d")
    
    # 取最后 50 张（通常是验证集部分）
    indices = list(range(len(dataset)))[-50:] 
    val_sub = torch.utils.data.Subset(dataset, indices)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

    print(f"开始生成对比图，保存至: {SAVE_DIR}")
    
    with torch.no_grad():
        for i, (inputs, targets, names) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device) # (1, 3, 256, 256)
            
            # --- Exp1: 模拟 2D 输入逻辑 ---
            mid = inputs[:, 1:2, :, :] # 取 2.5D 的中间层
            inputs_2d = torch.cat([mid, mid, mid], dim=1)
            out1 = model1(inputs_2d)
            pred1 = torch.argmax(out1, dim=1).squeeze(0).cpu().numpy()

            # --- Exp2 & Exp3: 直接 2.5D 输入 ---
            out2 = model2(inputs)
            out3 = model3(inputs)
            pred2 = torch.argmax(out2, dim=1).squeeze(0).cpu().numpy()
            pred3 = torch.argmax(out3, dim=1).squeeze(0).cpu().numpy()

            # 准备绘图数据
            # 反归一化并转回 CPU numpy
            img_show = denormalize(inputs[0]).permute(1, 2, 0).detach().cpu().numpy()
            gt_show = targets[0].detach().cpu().numpy()

            # --- 绘图排版 ---
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            # 1. 原图 (中间切片)
            axes[0].imshow(img_show)
            axes[0].set_title("Original Image", fontsize=14)
            
            # 2. 真值标签
            axes[1].imshow(gt_show, cmap='jet')
            axes[1].set_title("Ground Truth", fontsize=14)
            
            # 3. 实验 1 结果
            axes[2].imshow(pred1, cmap='jet')
            axes[2].set_title("Exp1: 2D Baseline", fontsize=14)
            
            # 4. 实验 2 结果
            axes[3].imshow(pred2, cmap='jet')
            axes[3].set_title("Exp2: 2.5D (No Gate)", fontsize=14)
            
            # 5. 实验 3 结果
            axes[4].imshow(pred3, cmap='jet')
            axes[4].set_title("Exp3: Ours (Gate)", fontsize=14)

            # 移除坐标轴
            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            
            # 处理文件名（移除 .npy 后缀）
            clean_name = names[0].split('.')[0]
            save_path = os.path.join(SAVE_DIR, f"compare_{clean_name}.png")
            
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    run_visualization()
