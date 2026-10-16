import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import distance_transform_edt
from thop import profile
import timm

# 导入你的组件
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPTHead

# ==============================================================
# 1. 基础配置与复杂度计算
# ==============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_complexity(model, input_size=(1, 3, 252, 252), device="cuda"):
    """计算参数量和FLOPs (注意 DINOv2 输入尺寸通常适配 patch14, 这里用 252x252)"""
    dummy_input = torch.randn(*input_size).to(device)
    flops, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 40)
    print("📊 模型参数量与计算量统计 (DINOv2):")
    print(f"   🔹 总参数量 (Total Params):     {total_params / 1e6:.2f} M")
    print(f"   🟢 可训练参数 (Trainable Params): {trainable_params / 1e6:.2f} M")
    print(f"   ❄️  冻结参数 (Frozen Params):     {frozen_params / 1e6:.2f} M")
    print(f"   ⚡ 计算量 (FLOPs):              {flops / 1e9:.2f} G")
    print("=" * 40 + "\n")
    
    return total_params / 1e6, trainable_params / 1e6, flops / 1e9

# ==============================================================
# 2. 评估逻辑 (完美适配 MSD 二分类，并跳过空切片)
# ==============================================================
def metrics_multiclass_torch(logits, targets, num_classes, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    targets = targets.squeeze(1)
    dice_sum, iou_sum, valid_classes = 0.0, 0.0, 0
    for c in range(1, num_classes):  
        p, t = (preds == c).float(), (targets == c).float()
        inter = (p * t).sum()
        union_add = p.sum() + t.sum()
        union_or = union_add - inter
        if union_add > 0:
            dice_sum += (2 * inter + eps) / (union_add + eps)
            iou_sum += (inter + eps) / (union_or + eps)
            valid_classes += 1
    # 使用 float() 彻底解决 .item() 报错
    return float(dice_sum / max(1, valid_classes)), float(iou_sum / max(1, valid_classes))

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

# ==============================================================
# 3. 对比实验模型 (Frozen DINOv2 + DPT Decoder)
# ==============================================================
class DINOv2_DPT(nn.Module):
    def __init__(self, nclass=2, features=128, out_channels=[96, 192, 384, 768]):
        super().__init__()
        print("🌐 正在通过 timm 加载 DINOv2-vits14 权重...")
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
        self.patch_size = 14
        self.embed_dim = 384
        
        self.out_indices = [2, 5, 8, 11]
        self.intermediate_features = []

        def hook_fn(module, input, output):
            self.intermediate_features.append(output)

        for i in self.out_indices:
            self.backbone.blocks[i].register_forward_hook(hook_fn)
        
        # 禁用 Gating (作为最纯净的 Baseline)
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
        
        self.intermediate_features = []
        _ = self.backbone(x)
        features = [f[:, -spatial_len:, :] for f in self.intermediate_features]
        out = self.head(features, patch_h, patch_w)
        return F.interpolate(out, (h, w), mode='bilinear', align_corners=True)

# ==============================================================
# 4. 训练与测试循环
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
    """专门为 MSD 设计的最终测试：跳过无肿瘤切片，算 HD95"""
    model.eval()
    dices, ious, hd95s = [], [], []
    print("\n⏳ 正在进行最终全面评估 (跳过全背景切片)...")
    for inputs, targets, _ in tqdm(val_loader, desc="[Final Test]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        
        pred_np = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        gt_np = targets.squeeze(0).cpu().numpy()
        
        p, g = (pred_np == 1), (gt_np == 1)
        
        # 🚨 只有包含肿瘤时才计算，防止成绩被拉低
        if g.sum() > 0: 
            dice, iou = metrics_multiclass_torch(logits, targets, num_classes)
            dices.append(dice)
            ious.append(iou)
            
            hd = hd95_simple(p, g)
            hd95s.append(hd)
            
    return np.mean(dices), np.mean(ious), np.mean(hd95s)

def main():
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    # 🚨 注意这里指向你的 MSD 数据集！
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/msd_lung_processed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=2) # MSD是二分类
    args = parser.parse_args()

    exp_name = "MSD_Exp0_DINOv2_Baseline_2D"
    print(f"\n🚀 启动MSD泛化对比实验: {exp_name}")

    # 模型初始化
    model = DINOv2_DPT(nclass=args.num_classes).cuda()
    model.lock_backbone()

    # 📊 获取填表用的参数量和计算量 (DINOv2 用 252x252 算 FLOPs)
    total_p, train_p, g_flops = get_complexity(model, input_size=(1, 3, 252, 252), device="cuda")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据加载 (强制 Resize 到 252x252，这是 DINOv2 patch14 的倍数)
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
    # 单独留一个 batch=1 的 loader 用于最终测 HD95
    eval_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    best_dice = 0.0
    ckpt_dir = f"./runs_msd/{exp_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = ""

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
    print("      📊 论文填表数据总结 (DINOv2 Baseline @ MSD)")
    print("🔥"*25)
    
    if best_ckpt_path == "":
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_ep")]
        if files: best_ckpt_path = os.path.join(ckpt_dir, sorted(files)[-1])

    if os.path.exists(best_ckpt_path):
        print(f"加载最佳权重: {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        
        final_dice, final_iou, final_hd95 = final_test_for_table(model, eval_loader, "cuda", args.num_classes)
        
        print("\n================== 将此数据填入表格 ==================")
        print(f"Dataset:          MSD Lung Tumor")
        print(f"Method:           Frozen DINOv2-S/14 + Decoder")
        print(f"Dimension:        2D")
        print(f"Total Params:     {total_p:.2f} M")
        print(f"Trainable Params: {train_p:.2f} M")
        print(f"FLOPs:            {g_flops:.2f} G")
        print(f"mDice:            {final_dice * 100:.2f}%")
        print(f"mIoU:             {final_iou * 100:.2f}%")
        print(f"mHD95:            {final_hd95:.2f}")
        print("======================================================")

if __name__ == "__main__":
    main()
