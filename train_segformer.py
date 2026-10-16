import os
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from medpy.metric.binary import hd95 # 用于计算 HD95

# ==============================================================
# 🚨 实验控制中心
# 1 = SegFormer-B2 2D Baseline (Frozen Encoder)
# 2 = SegFormer-B2 2.5D Baseline (Frozen Encoder)
# ==============================================================
EXPERIMENT_MODE = 1 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_stats(model, input_res=(1, 3, 256, 256)):
    """计算参数量和 FLOPs (G)"""
    model.eval()
    inputs = torch.randn(input_res).cuda()
    
    # 计算 FLOPs
    flops = FlopCountAnalysis(model, inputs)
    total_flops = flops.total() / 1e9  # 转换为 GFLOPs
    
    # 计算参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    print("\n" + "=" * 40)
    print(f"📊 论文指标统计:")
    print(f"   🔸 Trainable Params: {trainable_params:.2f} M")
    print(f"   🔸 FLOPs (G):       {total_flops:.2f} G")
    print("=" * 40 + "\n")
    return trainable_params, total_flops

def calculate_mHD95(preds, targets, num_classes):
    """
    计算平均 95% 豪斯多夫距离 (mHD95)
    preds, targets: (B, H, W) torch tensors
    """
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    hd95_list = []
    for b in range(preds_np.shape[0]):
        batch_hd = []
        for c in range(1, num_classes):  # 忽略背景
            p = (preds_np[b] == c).astype(np.uint8)
            t = (targets_np[b] == c).astype(np.uint8)
            
            if np.sum(p) > 0 and np.sum(t) > 0:
                try:
                    batch_hd.append(hd95(p, t))
                except:
                    batch_hd.append(50.0) # 如果无法计算（如预测完全错误），给一个较大的惩罚值
            elif np.sum(p) == 0 and np.sum(t) == 0:
                batch_hd.append(0.0) # 预测和真值都为空，完美匹配
            else:
                batch_hd.append(50.0) # 一个有一位一个没有，给惩罚值
        hd95_list.append(np.mean(batch_hd))
    return np.mean(hd95_list)

def metrics_all_torch(logits, targets, num_classes):
    """计算 Dice 和 IoU"""
    preds = torch.argmax(logits, dim=1)
    targets = targets.squeeze(1) if targets.dim() == 4 else targets
    dice_sum, iou_sum, valid_classes = 0.0, 0.0, 0
    eps = 1e-6
    
    for c in range(1, num_classes):
        p, t = (preds == c).float(), (targets == c).float()
        inter = (p * t).sum()
        union_add = p.sum() + t.sum()
        union_or = union_add - inter
        if union_add > 0:
            dice_sum += (2 * inter + eps) / (union_add + eps)
            iou_sum += (inter + eps) / (union_or + eps)
            valid_classes += 1
            
    mDice = (dice_sum / max(1, valid_classes)).item()
    mIoU = (iou_sum / max(1, valid_classes)).item()
    return mDice, mIoU, preds

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes):
    model.eval()
    dice_list, iou_list, hd95_list = [], [], []
    
    for inputs, targets, _ in tqdm(val_loader, desc="[Eval]"):
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        
        # [修复] 去除通道维度
        if targets.dim() == 4:
            targets = targets.squeeze(1)
            
        # [修复] 评测时越界值设为 0 (背景) 或者不管，这里按你原代码逻辑
        targets[targets >= num_classes] = 0
        
        outputs = model(inputs).logits
        logits = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        
        dice, iou, preds = metrics_all_torch(logits, targets, num_classes)
        # 计算 HD95
        hd = calculate_mHD95(preds, targets, num_classes)
        
        dice_list.append(dice)
        iou_list.append(iou)
        hd95_list.append(hd)
        
    return np.mean(dice_list), np.mean(iou_list), np.mean(hd95_list)

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

    configs = {1: ("SegB2_2D", "2d"), 2: ("SegB2_25D", "2.5d")}
    exp_name, dataset_type = configs[EXPERIMENT_MODE]

    # 加载模型 (加入了 use_safetensors=True 防止你再遇到加载漏洞报错)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    ).cuda()

    # 冻结 Encoder
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    get_model_stats(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据集
    from dataset import FolderDataset, ResizeAndNormalize
    dataset = FolderDataset(root=args.data_dir, transform=ResizeAndNormalize(), input_type=dataset_type)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size], 
                                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    best_dice = 0.0
    ckpt_dir = f"./runs/{exp_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 默认 CrossEntropyLoss 的 ignore_index 就是 -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs = inputs.to("cuda")
            targets = targets.to("cuda").long()
            
            # 👇👇👇 [关键修复区域开始] 👇👇👇
            
            # 1. 消除多余的通道维度 [B, 1, H, W] -> [B, H, W]
            if targets.dim() == 4:
                targets = targets.squeeze(1)
                
            # 2. 将越界的标签(比如255, 或者是超过11的值)设为 -100 (让Loss函数忽略它)
            targets[targets >= args.num_classes] = -100
            targets[targets < 0] = -100 
            
            # 👆👆👆 [关键修复区域结束] 👆👆👆

            optimizer.zero_grad()
            outputs = model(inputs).logits
            
            # 加上 align_corners=False 可以消除 PyTorch 警告
            logits = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
            # 此时的 logits 是 [B, 11, H, W], targets 是纯净的 [B, H, W]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        # 验证并输出
        v_dice, v_iou, v_hd = evaluate(model, val_loader, "cuda", args.num_classes)
        scheduler.step()

        print(f"📊 Result -> Dice: {v_dice:.4f} | IoU: {v_iou:.4f} | HD95: {v_hd:.4f}")

        # 保存最佳权重文件
        if v_dice > best_dice:
            best_dice = v_dice
            ckpt_name = f"best_B2_Dice{v_dice:.4f}_IoU{v_iou:.4f}_HD{v_hd:.2f}.pth"
            torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
            print(f"✅ Saved weight: {ckpt_name}")

if __name__ == "__main__":
    main()
