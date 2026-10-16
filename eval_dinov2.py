import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from thop import profile
import timm
from scipy.ndimage import distance_transform_edt

# 导入你原有的组件
from train_segdino import set_seed, print_model_params
from dataset import FolderDataset, ResizeAndNormalize
from dpt import DPTHead

# ==============================================================
# 1. 强制对齐 DINOv3 的 HD95 和评价指标逻辑
# ==============================================================
def print_flops(model, input_size=(1, 3, 252, 252), device="cuda"):
    dummy_input = torch.randn(*input_size).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops_g = (macs * 2) / 1e9  
    print(f"   ⚡ 计算量 (FLOPs): {flops_g:.2f} G")
    print("=" * 40 + "\n")
    return flops_g

def hd95_simple(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not np.any(pred) or not np.any(gt):
        return 100.0  # 对齐 DINOv3 的 100.0 惩罚
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    dist_pred_to_gt = dt_gt[pred]
    dist_gt_to_pred = dt_pred[gt]
    return np.percentile(np.concatenate([dist_pred_to_gt, dist_gt_to_pred]), 95)

@torch.no_grad()
def evaluate_all_metrics(model, val_loader, device, num_classes=11):
    model.eval()
    dice_list, iou_list, hd95_list = [], [], []
    
    for inputs, targets, _ in tqdm(val_loader, desc="[Evaluating DINOv2]"):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets[targets >= num_classes] = 0
        logits = model(inputs)
        
        preds_np = torch.argmax(logits, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        for b in range(preds_np.shape[0]):
            p = preds_np[b]
            g = targets_np[b]
            
            # 遍历前景类 (1~10)
            for c in range(1, num_classes):
                pc = (p == c)
                gc = (g == c)
                
                # 对齐 DINOv3：如果真实标签中没有这个器官，直接记为 NaN 并跳过
                if gc.sum() == 0:
                    dice_list.append(np.nan)
                    iou_list.append(np.nan)
                    hd95_list.append(np.nan)
                else:
                    inter = np.logical_and(pc, gc).sum()
                    union_add = pc.sum() + gc.sum()
                    union_or = union_add - inter
                    
                    dice_list.append((2. * inter) / union_add if union_add > 0 else 0.0)
                    iou_list.append(inter / union_or if union_or > 0 else 0.0)
                    hd95_list.append(hd95_simple(pc, gc))
            
    return np.nanmean(dice_list), np.nanmean(iou_list), np.nanmean(hd95_list)

# ==============================================================
# 2. DINOv2 模型定义 (不变)
# ==============================================================
class DINOv2_DPT(nn.Module):
    def __init__(self, nclass=11, features=128, out_channels=[96, 192, 384, 768]):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, dynamic_img_size=True)
        self.patch_size = 14
        self.embed_dim = 384
        self.out_indices = [2, 5, 8, 11]
        self.intermediate_features = []

        def hook_fn(module, input, output):
            self.intermediate_features.append(output)

        for i in self.out_indices:
            self.backbone.blocks[i].register_forward_hook(hook_fn)
        
        self.head = DPTHead(nclass=nclass, in_channels=self.embed_dim, features=features, out_channels=out_channels, use_gating=False)

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
# 3. 推理验证主程序
# ==============================================================
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🔥 1. 这里填入你当时训练好的 DINOv2 最佳权重的真实路径！！！
    CKPT_PATH = "./runs/Exp0_DINOv2_Baseline_2D/ckpts/best_ep075_mDice0.8168_mIoU0.7118_mHD9548.7242.pth"
    DATA_DIR = "/root/autodl-tmp"
    NUM_CLASSES = 11

    print("🚀 启动 DINOv2 公平对齐验证...")
    model = DINOv2_DPT(nclass=NUM_CLASSES).to(device)
        # 1. 先把权重加载到内存中
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    
    # 2. 过滤掉 thop 库产生的 total_ops 和 total_params 垃圾键值
    cleaned_state_dict = {
        k: v for k, v in checkpoint.items() 
        if "total_ops" not in k and "total_params" not in k
    }
    
    # 3. 将干净的权重加载到模型中
    model.load_state_dict(cleaned_state_dict)

    print_flops(model, input_size=(1, 3, 252, 252))

    # 🔥 2. 强制对齐 DINOv3 的数据划分：取最后 100 张
    dataset = FolderDataset(root=DATA_DIR, transform=ResizeAndNormalize(size=(252, 252)), input_type="2d")
    indices = list(range(len(dataset)))[-100:] 
    val_ds = torch.utils.data.Subset(dataset, indices)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    print(f"📊 评测数据集大小锁定为: {len(val_ds)} 张 (完美对齐 DINOv3)")
    
    # 开始评估
    v_dice, v_iou, v_hd95 = evaluate_all_metrics(model, val_loader, device, NUM_CLASSES)
    
    print("\n" + "★"*60)
    print("🏆 【DINOv2 公平重评结果 (请填入主表)】 🏆")
    print("-" * 60)
    print(f"mDice (↑): {v_dice * 100:.2f} %")
    print(f"mIoU  (↑): {v_iou * 100:.2f} %")
    print(f"mHD95 (↓): {v_hd95:.2f}  <--- 现在的 HD95 绝对合理了！")
    print("★"*60)

if __name__ == "__main__":
    main()
