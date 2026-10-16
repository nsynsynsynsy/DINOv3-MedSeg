import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile
from medpy import metric
import timm  # 引入 timm 库替代 torch.hub

# 导入你原有的组件
from train_segdino import set_seed, print_model_params, metrics_multiclass_torch, train_one_epoch
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPTHead

# ==============================================================
# 1. 性能指标补全函数 (FLOPs & HD95)
# ==============================================================

def print_flops(model, input_size=(1, 3, 252, 252), device="cuda"):
    """计算并在控制台打印模型的 FLOPs"""
    dummy_input = torch.randn(*input_size).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops_g = (macs * 2) / 1e9  # 1 MAC ≈ 2 FLOPs
    print(f"   ⚡ 计算量 (FLOPs): {flops_g:.2f} G")
    print("=" * 40 + "\n")
    return flops_g

def compute_hd95(pred_np, target_np, num_classes):
    """计算 95% Hausdorff Distance"""
    hd95_sum = 0.0
    valid_classes = 0
    max_dist = np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2) 
    
    for c in range(1, num_classes):
        p = (pred_np == c)
        t = (target_np == c)
        if np.sum(p) > 0 and np.sum(t) > 0:
            hd95_sum += metric.binary.hd95(p, t)
            valid_classes += 1
        elif np.sum(p) == 0 and np.sum(t) == 0:
            continue
        else:
            hd95_sum += max_dist
            valid_classes += 1
            
    return hd95_sum / max(1, valid_classes) if valid_classes > 0 else 0.0

@torch.no_grad()
def evaluate_all_metrics(model, val_loader, device, num_classes=11):
    """同时计算 mDice, mIoU, mHD95"""
    model.eval()
    dice_list, iou_list, hd95_list = [], [], []
    
    for inputs, targets, _ in tqdm(val_loader, desc="[Eval All Metrics]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        
        # 计算 Dice 和 IoU (GPU)
        dice, iou = metrics_multiclass_torch(logits, targets, num_classes)
        dice_list.append(dice)
        iou_list.append(iou)
        
        # 计算 HD95 (CPU)
        preds_np = torch.argmax(logits, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        for b in range(preds_np.shape[0]):
            hd_val = compute_hd95(preds_np[b], targets_np[b], num_classes)
            hd95_list.append(hd_val)
            
    return np.mean(dice_list), np.mean(iou_list), np.mean(hd95_list)

# ==============================================================
# 2. 对比实验模型 (Frozen DINOv2 + DPT Decoder) 使用 timm 重构
# ==============================================================

class DINOv2_DPT(nn.Module):
    def __init__(self, nclass=11, features=128, out_channels=[96, 192, 384, 768]):
        super().__init__()
        print("🌐 正在通过 timm 加载 DINOv2-vits14 权重 (自动绕过 GitHub 限制)...")
        # 🔥 修改点：加入了 dynamic_img_size=True，让模型自动适应 252x252 输入
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
        self.patch_size = 14
        self.embed_dim = 384
        
        # 定义要提取特征的层数对齐原逻辑 [2, 5, 8, 11]
        self.out_indices = [2, 5, 8, 11]
        self.intermediate_features = []

        # 使用 Hook 机制安全提取特征，不受版本影响
        def hook_fn(module, input, output):
            self.intermediate_features.append(output)

        for i in self.out_indices:
            self.backbone.blocks[i].register_forward_hook(hook_fn)
        
        # 调用你 DPT.py 里的 DPTHead
        self.head = DPTHead(
            nclass=nclass, 
            in_channels=self.embed_dim, 
            features=features, 
            out_channels=out_channels, 
            use_gating=False
        )

    def lock_backbone(self):
        print("❄️  [Frozen] DINOv2 骨干网络已锁定。")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        spatial_len = patch_h * patch_w
        
        # 1. 每次前向传播前清空列表
        self.intermediate_features = []
        
        # 2. 跑一遍骨干网络，触发 Hook 收集特征
        _ = self.backbone(x)
        
        # 3. 截取空间 Patch 特征 (抛弃 DINOv2 前面的 cls_token 和 register_tokens)
        # 确保输入给 head 的维度是 [B, patch_h*patch_w, C]
        features = [f[:, -spatial_len:, :] for f in self.intermediate_features]
        
        # 4. 进入 DPT 解码器
        out = self.head(features, patch_h, patch_w)
        return F.interpolate(out, (h, w), mode='bilinear', align_corners=True)

# ==============================================================
# 3. 主训练流程
# ==============================================================

def main():
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=11)
    args = parser.parse_args()

    exp_name = "Exp0_DINOv2_Baseline_2D"
    print(f"\n🚀 启动对比实验: {exp_name}")

    # 模型初始化
    model = DINOv2_DPT(nclass=args.num_classes).cuda()
    model.lock_backbone()

    # 打印表格所需的参数和计算量
    print_model_params(model)
    print_flops(model, input_size=(1, 3, 252, 252))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据加载 (适配 DINOv2 的 Patch 14 -> 252x252)
    dataset = FolderDataset(
        root=args.data_dir, 
        transform=ResizeAndNormalize(size=(252, 252)), 
        input_type="2d"
    )
    
    train_size = int(0.9 * len(dataset))
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    best_dice = 0.0
    ckpt_dir = f"./runs/{exp_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # 训练
        t_loss, t_dice = train_one_epoch(model, train_loader, optimizer, "cuda", args.num_classes, epoch)
        # 验证 (获取所有 3 个医学指标)
        v_dice, v_iou, v_hd95 = evaluate_all_metrics(model, val_loader, "cuda", args.num_classes)
        scheduler.step()

        print(f"Epoch {epoch} | mDice: {v_dice:.4f} ↑ | mIoU: {v_iou:.4f} ↑ | mHD95: {v_hd95:.4f} ↓")

        if v_dice > best_dice:
            for f in os.listdir(ckpt_dir):
                if f.startswith("best_ep"):
                    try: os.remove(os.path.join(ckpt_dir, f))
                    except: pass
            best_dice = v_dice
            ckpt_name = f"best_ep{epoch:03d}_mDice{v_dice:.4f}_mIoU{v_iou:.4f}_mHD95{v_hd95:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
            print(f"✅ 已保存最高分权重 (用于填表): {ckpt_name}")

if __name__ == "__main__":
    main()
