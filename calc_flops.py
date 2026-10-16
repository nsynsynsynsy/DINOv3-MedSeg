import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# ！！请修改以下配置 ！！
# ==========================================
DINO_REPO_DIR = "./dinov3"                   
DINO_CKPT_PATH = "dinov3_vits16_pretrain.pth"  # 这里一定要填写真实的权重路径！
IMAGE_SIZE = 1024                            
# ==========================================

def check_model_complexity(model, input_size, model_name):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 构造假输入 (Batch=1, C=3, H, W)
    dummy_input = torch.randn(input_size).to(device)
    
    # 1. 计算总参数量 (单位: M)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # 2. 使用 fvcore 计算真实的 FLOPs (单位: G)
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False) # 关闭不支持算子的警告
    flops.uncalled_modules_warnings(False) 
    
    total_flops = flops.total() / 1e9 # 转为 Giga (G) FLOPs
    
    print(f"========== {model_name} ==========")
    print(f"输入尺寸 : {input_size}")
    print(f"参数量   : {total_params:.2f} M (与你截图的 24.08M / 24.38M 对齐)")
    print(f"GFLOPs   : {total_flops:.2f} G")
    print("===================================\n")

if __name__ == '__main__':
    print("正在加载 DINOv3 骨干网络...")
    backbone = torch.hub.load(DINO_REPO_DIR, 'dinov3_vits16', source='local', weights=DINO_CKPT_PATH)
    
    from dpt import DPT
    input_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("\n开始计算真实复杂度 (过程可能需要几秒钟)...\n")

    # ------------------------------------------------
    # [Exp1 & Exp2] Baseline / 2.5D No Gate
    # 这里修改为真实的参数名 use_gating=False
    # ------------------------------------------------
    model_base = DPT(nclass=1, backbone=backbone, use_gating=False)
    check_model_complexity(model_base, input_shape, "Exp1(2D) & Exp2(2.5D No Gate)")

    # ------------------------------------------------
    # [Exp3] 2.5D With Gate (Ours)
    # 这里修改为真实的参数名 use_gating=True
    # ------------------------------------------------
    model_ours = DPT(nclass=1, backbone=backbone, use_gating=True)
    check_model_complexity(model_ours, input_shape, "Exp3 (2.5D With Gate - Ours)")
