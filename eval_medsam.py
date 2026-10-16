import os
import random
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor
from dataset import FolderDataset

# ==============================================================
# 配置区域
# ==============================================================
DATA_DIR = "/root/autodl-tmp"          
MEDSAM_CKPT = "./medsam_vit_b.pth"     
NUM_CLASSES = 11                       
EVAL_SIZE = (256, 256)                 
# ==============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hd95_simple(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not np.any(pred) or not np.any(gt):
        return 100.0  # 对齐 DINOv3 惩罚
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    dist_pred_to_gt = dt_gt[pred]
    dist_gt_to_pred = dt_pred[gt]
    return np.percentile(np.concatenate([dist_pred_to_gt, dist_gt_to_pred]), 95)

def get_jittered_bbox(mask_2d):
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(y_indices) == 0:
        return None
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    H, W = mask_2d.shape
    
    pad_y_min, pad_y_max = random.randint(2, 5), random.randint(2, 5)
    pad_x_min, pad_x_max = random.randint(2, 5), random.randint(2, 5)
    
    y_min = max(0, y_min - pad_y_min)
    x_min = max(0, x_min - pad_x_min)
    y_max = min(H - 1, y_max + pad_y_max)
    x_max = min(W - 1, x_max + pad_x_max)
    return np.array([x_min, y_min, x_max, y_max])

def calculate_model_stats(model, device):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    flops_g = 0.0
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
        flops, _ = profile(model.image_encoder, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except ImportError:
        pass
    return trainable_params, total_params, flops_g

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 正在加载 MedSAM 大模型...")
    sam_model = sam_model_registry["vit_b"]()
    ckpt = torch.load(MEDSAM_CKPT, map_location="cpu")
    sam_model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    sam_model = sam_model.to(device)
    sam_model.eval()
    
    trainable_p, total_p, flops = calculate_model_stats(sam_model, device)
    predictor = SamPredictor(sam_model)
    
    # 🔥 1. 使用 lambda 原样返回，完美兼容 Dataset
    dataset = FolderDataset(root=DATA_DIR, input_type="2.5d", transform=lambda x, y: (x, y))
    
    # 🔥 2. 强制对齐 DINOv3 的数据划分：取最后 100 张！
    indices = list(range(len(dataset)))[-100:] 
    val_ds = torch.utils.data.Subset(dataset, indices)
    
    print(f"📊 评测数据集大小锁定为: {len(val_ds)} 张 (完美对齐 DINOv3)")
    
    dice_list, iou_list, hd95_list = [], [], []
    
    for i in tqdm(range(len(val_ds)), desc="[MedSAM Zero-shot Eval]"):
        img_np, mask_np, _ = val_ds[i] 
        img_rgb = np.clip(np.transpose(img_np, (1, 2, 0)), 0, 255).astype(np.uint8)
        
        predictor.set_image(img_rgb)
        mask_256 = cv2.resize(mask_np, EVAL_SIZE, interpolation=cv2.INTER_NEAREST)
        
        for class_id in range(1, NUM_CLASSES):
            class_mask_original = (mask_np == class_id).astype(np.uint8)
            gt_binary = (mask_256 == class_id).astype(np.uint8)
            
            # 🔥 对齐 DINOv3 逻辑：如果本张图没有这个器官，记为 NaN 并忽略！
            if gt_binary.sum() == 0:
                dice_list.append(np.nan)
                iou_list.append(np.nan)
                hd95_list.append(np.nan)
                continue
                
            box = get_jittered_bbox(class_mask_original)
            if box is not None:
                masks, _, _ = predictor.predict(
                    point_coords=None, point_labels=None, box=box, multimask_output=False
                )
                pred_binary = cv2.resize(masks[0].astype(np.uint8), EVAL_SIZE, interpolation=cv2.INTER_NEAREST)
            else:
                pred_binary = np.zeros_like(gt_binary)
                
            inter = np.logical_and(pred_binary, gt_binary).sum()
            union_add = pred_binary.sum() + gt_binary.sum()
            union_or = union_add - inter
            
            dice_list.append((2. * inter) / union_add if union_add > 0 else 0.0)
            iou_list.append(inter / union_or if union_or > 0 else 0.0)
            hd95_list.append(hd95_simple(pred_binary, gt_binary))

    # 忽略 NaN 求均值
    final_dice = np.nanmean(dice_list) * 100
    final_iou = np.nanmean(iou_list) * 100
    final_hd95 = np.nanmean(hd95_list)
    
    print("\n" + "★"*60)
    print("🏆 【MedSAM (Zero-shot) 公平重评结果 (请填入主表)】 🏆")
    print("-" * 60)
    print(f"FLOPs (G): {flops:.2f} G")
    print(f"mDice (↑): {final_dice:.2f} %")
    print(f"mIoU  (↑): {final_iou:.2f} %")
    print(f"mHD95 (↓): {final_hd95:.2f}")
    print("★"*60)

if __name__ == "__main__":
    main()
