import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def preprocess():
    # 原始路径
    src_root = "/root/autodl-tmp/Task06_Lung/Task06_Lung"
    # 处理后的存储路径
    save_root = "/root/autodl-tmp/msd_lung_processed"
    
    os.makedirs(os.path.join(save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "labels"), exist_ok=True)

    img_dir = os.path.join(src_root, "imagesTr")
    mask_dir = os.path.join(src_root, "labelsTr")
    files = [f for f in os.listdir(img_dir) if f.endswith(".nii.gz") and not f.startswith("._")]

    for f in tqdm(files):
        img = nib.load(os.path.join(img_dir, f)).get_fdata()
        mask = nib.load(os.path.join(mask_dir, f)).get_fdata()

        # 窗口化：针对肺部 (-1000, 400)
        img = np.clip(img, -1000, 400)
        img = (img - (-1000)) / (400 - (-1000)) * 255.0
        img = img.astype(np.uint8)

        # 转换为 2.5D (每 3 层一组)
        for i in range(1, img.shape[2] - 1):
            # 过滤大部分全黑切片以节省训练时间
            if np.sum(mask[:, :, i]) == 0 and np.random.random() > 0.05:
                continue
                
            slice_25d = img[:, :, i-1:i+2].transpose(2, 0, 1) # (3, H, W)
            label = mask[:, :, i]
            
            save_name = f"{f.split('.')[0]}_s{i:03d}.npy"
            np.save(os.path.join(save_root, "images", save_name), slice_25d)
            np.save(os.path.join(save_root, "labels", save_name), label)

if __name__ == "__main__":
    preprocess()

