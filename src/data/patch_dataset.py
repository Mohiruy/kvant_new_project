from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def read_gray(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m

def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class PatchSegDataset(Dataset):
    """
    Returns random patches from (image + aux_mask) and GT mask.
    aux_mask path: outputs/.../same_name.png
    """
    def __init__(self, images_dir, masks_gt_dir, aux_dir, patch=128, patches_per_image=8, seed=123):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_gt_dir)
        self.aux_dir = Path(aux_dir) if aux_dir else None
        self.patch = patch
        self.patches_per_image = patches_per_image
        self.rng = random.Random(seed)

        self.names = sorted([p.name for p in self.images_dir.glob("*.png")])
        assert len(self.names) > 0, "images_dir bo‘sh"

        # total length = images * patches_per_image
        self.N = len(self.names) * self.patches_per_image

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]

        img = read_rgb(self.images_dir / name)          # H,W,3
        gt  = read_gray(self.masks_dir / name)          # H,W

        if self.aux_dir is not None:
            aux = read_gray(self.aux_dir / name)        # H,W
        else:
            aux = np.zeros(gt.shape, dtype=np.uint8)

        # binarize masks to {0,1}
        gt = (gt > 127).astype(np.float32)
        aux = (aux > 127).astype(np.float32)

        H, W = gt.shape
        p = self.patch

        if H <= p or W <= p:
            # pad if too small
            pad_h = max(0, p - H)
            pad_w = max(0, p - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            gt  = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            aux = cv2.copyMakeBorder(aux,0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            H, W = gt.shape

        y = self.rng.randint(0, H - p)
        x = self.rng.randint(0, W - p)

        img_p = img[y:y+p, x:x+p, :]              # p,p,3
        gt_p  = gt[y:y+p, x:x+p]                  # p,p
        aux_p = aux[y:y+p, x:x+p]                 # p,p

        # normalize
        img_p = img_p.astype(np.float32) / 255.0
        aux_p = aux_p.astype(np.float32)[..., None]  # p,p,1

        # concat channels: 3 + 1 = 4
        x_in = np.concatenate([img_p, aux_p], axis=2)  # p,p,4

        # to torch: [C,H,W]
        x_in = torch.from_numpy(x_in).permute(2,0,1).float()
        gt_p = torch.from_numpy(gt_p[None, ...]).float()  # [1,H,W]

        return x_in, gt_p