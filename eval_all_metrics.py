import os
import torch
import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPT

# --- 严格锁死随机种子 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def calculate_hd95(pred, gt):
    pred_coords = np.argwhere(pred)
    gt_coords = np.argwhere(gt)
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 50.0 
    dist1 = directed_hausdorff(pred_coords, gt_coords)[0]
    dist2 = directed_hausdorff(gt_coords, pred_coords)[0]
    return max(dist1, dist2)

def main():
    data_dir = "/root/autodl-tmp"
    num_classes = 11
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dino_ckpt = "./web_pth/dinov3_vits16_pretrain.pth"
    # 确保权重文件路径正确
    resume_ckpt = "./runs/segdino_Medical_256/ckpts/best_ep047_mDice0.8479.pth" 

    print(f"正在加载模型并对齐 84.79% 权重...")
    backbone = torch.hub.load('./dinov3', 'dinov3_vits16', source='local', weights=dino_ckpt)
    model = DPT(nclass=num_classes, backbone=backbone).to(device)
    
    # 强制加载权重
    checkpoint = torch.load(resume_ckpt, map_location=device)
    # 如果保存的是字典格式 {"state_dict": ...}，则取 state_dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    dataset = FolderDataset(
        root=data_dir, split="", img_dir_name="images", label_dir_name="labels",
        transform=ResizeAndNormalize(size=(256, 256)),
        input_type="2d" # ### 对齐修改：显式指定2D ###
    )
    
    # ### 对齐修改：确保划分和训练时完全一致 ###
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # 使用显式 Generator 保证划分逻辑和训练时一模一样
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    print(f"🚀 开始评估 (验证集样本数: {len(val_ds)})...")
    
    total_dice_list = []
    total_iou_list = []
    total_hd95_list = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            if targets.ndim == 4:
                targets = targets.squeeze(1)
                
            invalid_mask = (targets < 0) | (targets >= num_classes)
            targets[invalid_mask] = 0

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            # 计算这一张图的 mDice 和 mIoU
            img_dice = []
            img_iou = []
            img_hd95 = []
            
            for c in range(1, num_classes):
                p = (preds == c).float()
                t = (targets == c).float()
                
                inter = (p * t).sum()
                union = p.sum() + t.sum()
                
                if union > 0:
                    # ### 对齐修改：训练脚本中 dice 计算的 eps 是 1e-6 ###
                    dice = (2 * inter + 1e-6) / (union + 1e-6)
                    iou = (inter + 1e-6) / (union - inter + 1e-6)
                    img_dice.append(dice.item())
                    img_iou.append(iou.item())
                    
                    # HD95 比较耗时，仅在有预测或有GT时计算
                    if p.sum() > 0 and t.sum() > 0:
                        hd = calculate_hd95(p[0].cpu().numpy(), t[0].cpu().numpy())
                        img_hd95.append(hd)

            if len(img_dice) > 0:
                total_dice_list.append(np.mean(img_dice))
                total_iou_list.append(np.mean(img_iou))
            if len(img_hd95) > 0:
                total_hd95_list.append(np.mean(img_hd95))

    print("\n" + "="*50)
    print(f"🏆 Final mDice : {np.mean(total_dice_list)*100:.2f} %")
    print(f"🏆 Final mIoU  : {np.mean(total_iou_list)*100:.2f} %")
    print(f"🏆 Final mHD95 : {np.mean(total_hd95_list):.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
