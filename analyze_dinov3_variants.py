import torch
import torch.nn as nn
import os
from thop import profile
import copy

# ==============================================================
# 🚨 实验一键控制中心
# 1 = Exp1: 2D Baseline (DINOv3 Frozen + DPT)
# 2 = Exp2: 2.5D No Gate (DINOv3 Frozen + DPT)
# 3 = Exp3: 2.5D + Feature Gating (Ours)
# ==============================================================
EXPERIMENT_MODE = 1 
# ==============================================================

# 配置基础路径
REPO_DIR = "./dinov3"
DINO_CKPT = "./web_pth/dinov3_vits16_pretrain.pth"
NUM_CLASSES = 11
INPUT_SIZE = (1, 3, 256, 256) # 统一 256x256 评估，确保公平对比

def get_model_variant(mode):
    """根据模式一键构建不同的模型变体"""
    from dpt import DPT
    
    # 加载 Backbone
    backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=DINO_CKPT)
    
    if mode == 1:
        # Exp1: 2D Baseline (不使用 Gating)
        print("🛠️  正在构建: Exp1 - 2D Baseline")
        model = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=False)
    
    elif mode == 2:
        # Exp2: 2.5D No Gate (架构上和 2D DPT 类似，但逻辑上处理 2.5D 输入)
        print("🛠️  正在构建: Exp2 - 2.5D No Gate")
        model = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=False)
        
    elif mode == 3:
        # Exp3: 2.5D + Feature Gating (我们的方法)
        print("🛠️  正在构建: Exp3 - 2.5D + Feature Gating (Ours)")
        model = DPT(nclass=NUM_CLASSES, backbone=backbone, use_gating=True)
    
    else:
        raise ValueError("Unknown EXPERIMENT_MODE!")

    # 统一冻结 Backbone
    model.lock_backbone()
    return model

def run_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 获取模型
    model = get_model_variant(EXPERIMENT_MODE).to(device)
    model.eval()

    # 2. 计算参数量 (M)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    # 3. 计算 FLOPs (G)
    # 特别说明：对于 2.5D，虽然它代表 3 个切片，但在输入 Tensor 层面依然是 (1, 3, 256, 256)
    dummy_input = torch.randn(*INPUT_SIZE).to(device)
    
    # 使用 thop 分析，并清理可能产生的垃圾键值
    # 这里用 copy.deepcopy 是为了防止 profile 修改原模型结构
    temp_model = copy.deepcopy(model)
    macs, params = profile(temp_model, inputs=(dummy_input, ), verbose=False)
    
    # 转换为 GFLOPs
    # 注意：通常业界 1 MACs = 2 FLOPs，但 thop 的 profile 结果习惯上直接对应 G 
    flops = macs / 1e9  

    print("\n" + "═"*50)
    print(f"📊 实验模式 {EXPERIMENT_MODE} 性能分析报告")
    print("-" * 50)
    configs = {1: "2D Baseline", 2: "2.5D No Gate", 3: "2.5D + Gating (Ours)"}
    print(f"🔹 实验名称: {configs[EXPERIMENT_MODE]}")
    print(f"🔹 输入尺寸: {INPUT_SIZE[2]}x{INPUT_SIZE[3]}")
    print(f"🟢 可训练参数 (Trainable): {trainable_params:.2f} M")
    print(f"📦 总参数量 (Total):      {total_params:.2f} M")
    print(f"⚡ 计算量 (FLOPs):        {flops:.2f} G")
    print("═"*50 + "\n")

if __name__ == "__main__":
    # 防止递归深度报错
    import sys
    sys.setrecursionlimit(3000)
    run_analysis()
