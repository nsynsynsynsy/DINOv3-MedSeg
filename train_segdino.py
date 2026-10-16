import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==============================================================
# 🚨 实验控制中心
# 1 = Exp1: 2D Baseline (DINOv3 Frozen)
# 2 = Exp2: 2.5D No Gate (DINOv3 Frozen)
# 3 = Exp3: 2.5D + Feature Gating (Ours, DINOv3 Frozen)
# ==============================================================
EXPERIMENT_MODE = 3


# ==============================================================

def set_seed(seed=42):
    """
    确保论文实验的绝对可复现性，消除网络初始化的玄学波动
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_params(model):
    """打印模型的参数量，区分可训练和冻结参数"""
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # 换算为百万(M)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    frozen_params = total_params - trainable_params

    print("\n" + "=" * 40)
    print("📊 模型参数量统计 (Parameters):")
    print(f"   🔹 总参数量 (Total):     {total_params:.2f} M")
    print(f"   🟢 可训练参数 (Trainable): {trainable_params:.2f} M")
    print(f"   ❄️  冻结参数 (Frozen):      {frozen_params:.2f} M")
    print("=" * 40 + "\n")


def metrics_multiclass_torch(logits, targets, num_classes, eps=1e-6):
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
    return (dice_sum / max(1, valid_classes)).item(), (iou_sum / max(1, valid_classes)).item()


def train_one_epoch(model, train_loader, optimizer, device, num_classes=11, epoch=0):
    model.train()
    total_loss, dice_list = 0.0, []
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets, _ in pbar:
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0  # 标签清洗

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
def evaluate(model, val_loader, device, num_classes=11):
    model.eval()
    dice_list = []
    iou_list = []
    for inputs, targets, _ in tqdm(val_loader, desc="[Eval]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        # 获取两个指标
        dice, iou = metrics_multiclass_torch(logits, targets, num_classes)
        dice_list.append(dice)
        iou_list.append(iou)
    return np.mean(dice_list), np.mean(iou_list)


def main():
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vits16_pretrain.pth")
    args = parser.parse_args()

    # 根据 EXPERIMENT_MODE 自动配置
    configs = {
        1: ("Exp1_2D_Baseline", "2d", False),
        2: ("Exp2_25D_NoGate", "2.5d", False),
        3: ("Exp3_25D_WithGate_Ours", "2.5d", True)
    }
    exp_name, dataset_type, enable_gating = configs[EXPERIMENT_MODE]
    print(f"\n🚀 启动实验: {exp_name} | 数据模式: {dataset_type} | 门控: {enable_gating}")
    print(f"⚙️  当前学习率: {args.lr}")

    # 模型初始化
    backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    from dpt import DPT
    model = DPT(nclass=args.num_classes, backbone=backbone, use_gating=enable_gating).cuda()

    # 🚨 核心：冻结 Backbone
    model.lock_backbone()

    # 📊 打印参数量
    print_model_params(model)

    # 仅优化非冻结参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据加载
    from dataset import FolderDataset, ResizeAndNormalize
    dataset = FolderDataset(root=args.data_dir, transform=ResizeAndNormalize(), input_type=dataset_type)
    train_size = int(0.9 * len(dataset))

    # 严格固定数据集划分种子
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    best_dice = 0.0
    ckpt_dir = f"./runs/{exp_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_dice = train_one_epoch(model, train_loader, optimizer, "cuda", args.num_classes, epoch)
        v_dice, v_iou = evaluate(model, val_loader, "cuda", args.num_classes)
        scheduler.step()

        print(f"Epoch {epoch} | Train Dice: {t_dice:.4f} | Val Dice: {v_dice:.4f} | Val IoU: {v_iou:.4f}")

        if v_dice > best_dice:
            # 1. 清理旧的 best 文件
            for f in os.listdir(ckpt_dir):
                if f.startswith("best_ep"):
                    try:
                        os.remove(os.path.join(ckpt_dir, f))
                    except:
                        pass

            best_dice = v_dice
            # 2. 保存新的权重，文件名包含指标
            ckpt_name = f"best_ep{epoch:03d}_mDice{v_dice:.4f}_mIoU{v_iou:.4f}.pth"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ 已保存最高分权重: {ckpt_name}")


if __name__ == "__main__":
    main()
