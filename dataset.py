import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ResizeAndNormalize:
    def __init__(self, size=(256, 256)):
        self.size = size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, img_np, mask_np):
        img_t = torch.from_numpy(img_np).float() / 255.0
        img_t = TF.resize(img_t, self.size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        img_t = TF.normalize(img_t, self.mean, self.std)

        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        mask_t = TF.resize(mask_t, self.size, interpolation=InterpolationMode.NEAREST)
        return img_t, mask_t.squeeze(0).long()


class FolderDataset(data.Dataset):
    def __init__(self, root, input_type="2.5d", transform=None):
        self.img_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.input_type = input_type
        self.transform = transform
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_np = np.load(os.path.join(self.img_dir, name))  # (3, H, W)
        mask_np = np.load(os.path.join(self.label_dir, name))  # (H, W)

        if self.input_type == "2d":
            # 2D 实验只取中间层并复制三份，模拟 2D 输入
            mid = img_np[1]
            img_np = np.stack([mid, mid, mid], axis=0)

        if self.transform:
            img_t, mask_t = self.transform(img_np, mask_np)
        return img_t, mask_t, name
