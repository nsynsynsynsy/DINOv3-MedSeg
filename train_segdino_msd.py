import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import distance_transform_edt
from thop import profile  # 用于计算 FLOPs

# 导入你的组件
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPT

# ==============================================================
# 🚨 实验控制中心
# 1 = Exp1: 2D Baseline (DINOv3 Frozen)
# 2 = Exp2: 2.5D No Gate (DINOv3 Frozen)
# 3 = Exp3: 2.5D + Feature Gating (Ours, DINOv3 Frozen)
# ==============================================================
EXPERIMENT_MODE = 1
# ==============================================================

def set_seed(seed=42):
    """保持与 AMOS22 完全一致的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_complexity(model, input_type="2.5d"):
    """计算参数量和FLOPs"""
    dummy_input = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 40)
    print("📊 模型参数量与计算量统计:")
    print(f"   🔹 总参数量 (Total Params):     {total_params / 1e6:.2f} M")
    print(f"   🟢 可训练参数 (Trainable Params): {trainable_params / 1e6:.2f} M")
    print(f"   ❄️  冻结参数 (Frozen Params):     {frozen_params / 1e6:.2f} M")
    print(f"   ⚡ 计算量 (FLOPs):              {flops / 1e9:.2f} G")
    print("=" * 40 + "\n")
    
    return total_params / 1e6, trainable_params / 1e6, flops / 1e9


# ==============================================================
# 评估逻辑 (修复了 .item() 报错)
# ==============================================================
def metrics_multiclass_torch(logits, targets, num_classes, eps=1e-6):
    """训练过程中的 Dice 和 IoU"""
    preds = torch.argmax(logits, dim=1)
    targets = targets.squeeze(1)
    dice_sum, iou_sum, valid_classes = 0.0, 0.0, 0
    for c in range(1, num_classes):  # 忽略背景类0
        p, t = (preds == c).float(), (targets == c).float()
        inter = (p * t).sum()
        union_add = p.sum() + t.sum()
        union_or = union_add - inter
        if union_add > 0:
            dice_sum += (2 * inter + eps) / (union_add + eps)
            iou_sum += (inter + eps) / (union_or + eps)
            valid_classes += 1
    
    # 使用 float() 替代 .item() 解决报错
    return float(dice_sum / max(1, valid_classes)), float(iou_sum / max(1, valid_classes))

def hd95_simple(pred, gt):
    """HD95计算"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not np.any(pred) or not np.any(gt):
        return 100.0
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    dist_pred_to_gt = dt_gt[pred]
    dist_gt_to_pred = dt_pred[gt]
    return np.percentile(np.concatenate([dist_pred_to_gt, dist_gt_to_pred]), 95)

# ==============================================================
# 训练与验证 Loop
# ==============================================================
def train_one_epoch(model, train_loader, optimizer, device, num_classes, epoch=0):
    model.train()
    total_loss, dice_list = 0.0, []
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets, _ in pbar:
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        dice, _ = metrics_multiclass_torch(logits, targets, num_classes)
        total_loss += loss.item()
        dice_list.append(dice)
        pbar.set_postfix(loss=f"{loss.item():.4f}", mDice=f"{dice:.4f}")
    return total_loss / len(train_loader), np.mean(dice_list)

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes):
    model.eval()
    dice_list, iou_list = [], []
    for inputs, targets, _ in tqdm(val_loader, desc="[Eval]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        dice, iou = metrics_multiclass_torch(logits, targets, num_classes)
        dice_list.append(dice)
        iou_list.append(iou)
    return np.mean(dice_list), np.mean(iou_list)

@torch.no_grad()
def final_test_for_table(model, val_loader, device, num_classes):
    """
    最终评估：跳过完全没有肿瘤的切片，确保 mDice 准确
    """
    model.eval()
    dices, ious, hd95s = [], [], []
    print("\n⏳ 正在进行最终全面评估 (计算 HD95 需要一点时间)...")
    for inputs, targets, _ in tqdm(val_loader, desc="[Final Test]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        
        pred_np = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        gt_np = targets.squeeze(0).cpu().numpy()
        
        # 针对肿瘤类 (Label 1)
        p, g = (pred_np == 1), (gt_np == 1)
        
        # 核心逻辑：只有真实标签有肿瘤时，才计入统计指标，防止背景切片拉低分数
        if g.sum() > 0: 
            # 算 Dice 和 IoU
            dice, iou = metrics_multiclass_torch(logits, targets, num_classes)
            dices.append(dice)
            ious.append(iou)
            
            # 算 HD95
            hd = hd95_simple(p, g)
            hd95s.append(hd)
            
    return np.mean(dices), np.mean(ious), np.mean(hd95s)

def main():
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    # 请确保路径指向你 MSD 数据集的位置
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/msd_lung_processed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=2) 
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vits16_pretrain.pth")
    args = parser.parse_args()

    configs = {
        1: ("MSD_Exp1_2D_Baseline", "2d", False),
        2: ("MSD_Exp2_25D_NoGate", "2.5d", False),
        3: ("MSD_Exp3_25D_WithGate_Ours", "2.5d", True)
    }
    exp_name, dataset_type, enable_gating = configs[EXPERIMENT_MODE]
    print(f"\n🚀 启动MSD泛化实验: {exp_name} | 数据模式: {dataset_type} | 门控: {enable_gating}")

    # 模型初始化
    backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    model = DPT(nclass=args.num_classes, backbone=backbone, use_gating=enable_gating).cuda()
    model.lock_backbone()

    # 📊 获取复杂度参数
    total_p, train_p, g_flops = get_complexity(model, dataset_type)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据加载逻辑
    dataset = FolderDataset(root=args.data_dir, transform=ResizeAndNormalize(), input_type=dataset_type)
    train_size = int(0.9 * len(dataset))

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # 单独留一个 batch=1 的 loader 用于最终测 HD95
    eval_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    best_dice = 0.0
    ckpt_dir = f"./runs_msd/{exp_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = ""

    # 训练
    for epoch in range(1, args.epochs + 1):
        t_loss, t_dice = train_one_epoch(model, train_loader, optimizer, "cuda", args.num_classes, epoch)
        v_dice, v_iou = evaluate(model, val_loader, "cuda", args.num_classes)
        scheduler.step()

        print(f"Epoch {epoch} | Train Dice: {t_dice:.4f} | Val Dice: {v_dice:.4f} | Val IoU: {v_iou:.4f}")

        if v_dice > best_dice:
            for f in os.listdir(ckpt_dir):
                if f.startswith("best_ep"):
                    try: os.remove(os.path.join(ckpt_dir, f))
                    except: pass

            best_dice = v_dice
            ckpt_name = f"best_ep{epoch:03d}_mDice{v_dice:.4f}_mIoU{v_iou:.4f}.pth"
            best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ 已保存最高分权重: {ckpt_name}")

    # ==============================================================
    # 🎉 自动生成论文对比表格
    # ==============================================================
    print("\n" + "🔥"*25)
    print("      📊 论文填表数据总结 (MSD Lung Tumor 泛化实验)")
    print("🔥"*25)
    
    # 自动加载最佳权重进行最终评估
    if best_ckpt_path == "":
        # 如果是直接运行测试，手动找目录下最新的 best
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_ep")]
        if files: best_ckpt_path = os.path.join(ckpt_dir, sorted(files)[-1])

    if os.path.exists(best_ckpt_path):
        print(f"加载最佳权重: {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        
        final_dice, final_iou, final_hd95 = final_test_for_table(model, eval_loader, "cuda", args.num_classes)
        
        print("\n================== 将此数据填入表格 ==================")
        print(f"Dataset:          MSD Lung Tumor")
        print(f"Method:           Frozen DINOv3 + {'Gating (Ours)' if enable_gating else 'Decoder'}")
        print(f"Dimension:        {dataset_type.upper()}")
        print(f"Total Params:     {total_p:.2f} M")
        print(f"Trainable Params: {train_p:.2f} M")
        print(f"FLOPs:            {g_flops:.2f} G")
        print(f"mDice:            {final_dice * 100:.2f}%")
        print(f"mIoU:             {final_iou * 100:.2f}%")
        print(f"mHD95:            {final_hd95:.2f}")
        print("======================================================")
    else:
        print("未找到权重文件，请检查训练是否正常完成。")


if __name__ == "__main__":
    main()
