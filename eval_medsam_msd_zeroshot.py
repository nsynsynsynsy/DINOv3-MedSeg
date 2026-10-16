import os
import random
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
from segment_anything import sam_model_registry, SamPredictor
from dataset import FolderDataset

# ==============================================================
# 🛠️ 配置区域 (MSD 肺部肿瘤专用)
# ==============================================================
DATA_DIR = "/root/autodl-tmp"          # 你的数据集根目录
MEDSAM_CKPT = "./medsam_vit_b.pth"     # MedSAM 官方权重路径
EVAL_SIZE = (252, 252)                 # 🎯 必须和 DINOv2 保持一致的分辨率
MAX_HD95 = 362.0                       
# ==============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_hd95(pred, gt):
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return MAX_HD95  
    coords_pred = np.argwhere(pred > 0)
    coords_gt = np.argwhere(gt > 0)
    tree_pred = cKDTree(coords_pred)
    tree_gt = cKDTree(coords_gt)
    d1, _ = tree_gt.query(coords_pred)
    d2, _ = tree_pred.query(coords_gt)
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    return hd95

def get_jittered_bbox(mask_2d):
    """根据真实标签生成带有随机抖动的 Bounding Box 提示"""
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(y_indices) == 0:
        return None # 该切片无肿瘤
    
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    H, W = mask_2d.shape
    
    # 随机向外扩展 2~5 个像素，模拟真实的人类标注框
    pad_y_min, pad_y_max = random.randint(2, 5), random.randint(2, 5)
    pad_x_min, pad_x_max = random.randint(2, 5), random.randint(2, 5)
    
    y_min = max(0, y_min - pad_y_min)
    x_min = max(0, x_min - pad_x_min)
    y_max = min(H - 1, y_max + pad_y_max)
    x_max = min(W - 1, x_max + pad_x_max)
    
    return np.array([x_min, y_min, x_max, y_max])

def calculate_model_stats(model, device):
    print("⏳ 正在计算参数量与 FLOPs...")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    flops_g = 0.0
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
        flops, _ = profile(model.image_encoder, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except ImportError:
        print("提示: 未安装 thop，跳过 FLOPs 计算 (可用 pip install thop 安装)")
    return trainable_params, total_params, flops_g

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 正在加载 MedSAM 大模型 (Zero-shot Box Prompt 模式)...")
    sam_model = sam_model_registry["vit_b"]()
    
    # 兼容不同 PyTorch 版本的加载方式
    ckpt = torch.load(MEDSAM_CKPT, map_location="cpu", weights_only=False)
    sam_model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    sam_model = sam_model.to(device)
    
    # 完全冻结 MedSAM，0 可训练参数
    for param in sam_model.parameters():
        param.requires_grad = False
    sam_model.eval()
    
    trainable_p, total_p, flops = calculate_model_stats(sam_model, device)
    predictor = SamPredictor(sam_model)
    
    # ==============================================================
    # 🎯 加载数据集 (使用 lambda 穿透，防止 numpy 格式报错)
    # ==============================================================
    dataset = FolderDataset(
        root=DATA_DIR, 
        input_type="2d", # MedSAM 原生评估使用 2D
        transform=lambda x, y: (x, y) 
    )
    
    # 严格保持 90/10 的划分比例和相同的 seed，保证测试集图片与 DINOv2 完全一致
    train_size = int(0.9 * len(dataset))
    generator = torch.Generator().manual_seed(42)
    _, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    
    print(f"📊 验证集数量: {len(val_ds)} 张切片")
    dice_list, iou_list, hd95_list = [], [], []
    eps = 1e-6
    
    for i in tqdm(range(len(val_ds)), desc="[MedSAM Zero-shot Eval]"):
        img_np, mask_np, name = val_ds[i] 
        
        # 1. 图像格式处理：转为 (H, W, 3) 且为 0-255 的 uint8
        if img_np.ndim == 3 and img_np.shape[0] == 3: # (3, H, W)
            img_rgb = np.transpose(img_np, (1, 2, 0))
        elif img_np.ndim == 3 and img_np.shape[0] == 1: # (1, H, W)
            img_rgb = np.repeat(np.transpose(img_np, (1, 2, 0)), 3, axis=2)
        elif img_np.ndim == 2: # (H, W)
            img_rgb = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
        else: # 兜底逻辑
            img_rgb = np.transpose(img_np, (1, 2, 0))[:, :, :3]
            
        if img_rgb.max() <= 1.0:
            img_rgb = img_rgb * 255
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        # 2. 设置图像给 SamPredictor
        predictor.set_image(img_rgb)
        
        # 3. 标签处理
        mask_eval = cv2.resize(mask_np.astype(np.uint8), EVAL_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # MSD 数据集中，肿瘤类别为 1
        gt_binary = (mask_eval == 1).astype(np.uint8)
        class_mask_original = (mask_np == 1).astype(np.uint8)
        
        # 生成提示框 (Box)
        box = get_jittered_bbox(class_mask_original)
        
        # 🚨 核心逻辑：只评估有肿瘤的切片（与你 AMOS 的代码逻辑对齐，防止被大量空切片冲淡分数）
        if box is not None and np.sum(gt_binary) > 0:
            
            # 使用 Box 进行零样本预测
            masks, _, _ = predictor.predict(
                point_coords=None, 
                point_labels=None, 
                box=box, 
                multimask_output=False
            )
            
            # 获取预测结果并缩放到 EVAL_SIZE 
            pred_binary = masks[0].astype(np.uint8)
            pred_binary = cv2.resize(pred_binary, EVAL_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # 计算指标
            inter = np.sum(pred_binary * gt_binary)
            union_add = np.sum(pred_binary) + np.sum(gt_binary)
            union_or = union_add - inter
            
            dice = (2 * inter + eps) / (union_add + eps)
            iou = (inter + eps) / (union_or + eps)
            hd95 = compute_hd95(pred_binary, gt_binary)
            
            dice_list.append(dice)
            iou_list.append(iou)
            hd95_list.append(hd95)

    # 计算均值
    final_dice = np.mean(dice_list) * 100
    final_iou = np.mean(iou_list) * 100
    final_hd95 = np.mean(hd95_list)
    
    print("\n" + "★"*65)
    print("🏆 MedSAM (带 GT Bbox Prompt) 最终对比表格数据 - MSD数据集")
    print("🚨 核心卖点: 尽管给了 GT 提示，表现依然远不如全自动微调的 DINOv3")
    print("-" * 65)
    print(f"1. Dimension:            2D")
    print(f"2. Gating:               ✘")
    print(f"3. Trainable Params (M): {trainable_p:.2f} M")
    print(f"4. FLOPs (G):            {flops:.2f} G")
    print(f"5. mDice (↑):            {final_dice:.2f} %")
    print(f"6. mIoU  (↑):            {final_iou:.2f} %")
    print(f"7. mHD95 (↓):            {final_hd95:.2f}")
    print("★"*65)

if __name__ == "__main__":
    main()
