import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

# ================= 创新点：自适应层级特征门控 (Ours) =================
class FeatureGatingModule(nn.Module):
    """
    针对 DINOv3 提取出的多层特征进行自适应权重分配。
    核心思想：通过全局上下文感知，动态调整 2.5D 输入下各层特征的贡献度。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, out_channels=[256, 512, 1024, 1024], use_gating=False):
        super(DPTHead, self).__init__()
        self.use_gating = use_gating

        # 如果开启门控，为 DINOv3 的 4 个输出层分别建立门控模块
        if self.use_gating:
            self.gating_layers = nn.ModuleList([
                FeatureGatingModule(in_channels) for _ in range(4)
            ])

        # 适配投影层：将 DINOv3 的 embed_dim 映射到解码器通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1) 
            for out_channel in out_channels
        ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.output_conv = nn.Conv2d(features * 4, nclass, kernel_size=1)
        self.proj = nn.ConvTranspose2d(features, features, 4, stride=4, bias=False)

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            # 1. 转换维度: [B, N, C] -> [B, C, H, W]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            # 2. 【核心创新】执行门控特征筛选
            if self.use_gating:
                x = self.gating_layers[i](x)
            
            # 3. 投影到解码器空间
            x = self.projects[i](x)
            out.append(x)

        # 4. DPT 特征融合路径
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.proj(self.scratch.layer1_rn(layer_1))
        target_hw = layer_1_rn.shape[-2:]
        
        layer_2_up = F.interpolate(self.scratch.layer2_rn(layer_2), size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(self.scratch.layer3_rn(layer_3), size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(self.scratch.layer4_rn(layer_4), size=target_hw, mode="bilinear", align_corners=True)
        
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        return self.scratch.output_conv(fused)

class DPT(nn.Module):
    def __init__(self, encoder_size='small', nclass=11, features=128, out_channels=[96, 192, 384, 768], backbone=None, use_gating=False):
        super(DPT, self).__init__()
        # DINOv3 提取 3, 6, 9, 12 层特征
        self.intermediate_layer_idx = {'small': [2, 5, 8, 11], 'base': [2, 5, 8, 11]}
        self.encoder_size = encoder_size
        self.backbone = backbone
        
        embed_dim = self.backbone.embed_dim
        self.head = DPTHead(nclass, embed_dim, features, out_channels=out_channels, use_gating=use_gating)

    def lock_backbone(self):
        """严格冻结 DINOv3 参数，确保消融实验的公平性"""
        print("❄️  [Frozen] DINOv3 骨干网络已锁定，仅训练解码器和门控模块。")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // 16, w // 16
        features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_size])
        out = self.head(features, patch_h, patch_w)
        return F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
