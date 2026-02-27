from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_mask_01(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 127).astype(np.float32)
    return m


def to_tensor_img(img_rgb: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(img_rgb.astype(np.float32) / 255.0)
    return torch.from_numpy(x).permute(2, 0, 1).float()


def to_tensor_mask(mask01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask01)[None, ...]).float()


def resize_img_mask(img_rgb: np.ndarray, mask01: np.ndarray, size: int):
    img_r = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    m_r = cv2.resize(mask01, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, m_r


def _mask_name_from_image(name: str, mask_ext: str = ".png") -> str:
    # "samoyed_64.jpg" -> "samoyed_64.png"
    return Path(name).stem + mask_ext


class SegDataset(Dataset):
    """
    Full-image dataset: returns (RGB + aux) resized to (resize, resize).
    """
    def __init__(self, root_dir: str, split_json: str, split: str, aux_dir: str, resize: int = 256):
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self.gt_dir = self.root / "masks_gt"
        self.aux_dir = Path(aux_dir)
        self.resize = int(resize)

        import json
        with open(split_json, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.files = splits[split]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]  # image file name (jpg)
        img = read_image_rgb(str(self.images_dir / name))

        mname = _mask_name_from_image(name, ".png")
        gt = read_mask_01(str(self.gt_dir / mname))

        aux_path = self.aux_dir / mname
        if aux_path.exists():
            aux = read_mask_01(str(aux_path))
        else:
            aux = np.zeros(gt.shape, dtype=np.float32)

        # align sizes to image
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if aux.shape[:2] != img.shape[:2]:
            aux = cv2.resize(aux, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img, gt = resize_img_mask(img, gt, self.resize)
        aux = cv2.resize(aux, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)

        x = torch.cat([to_tensor_img(img), to_tensor_mask(aux)], dim=0)  # 4,H,W
        y = to_tensor_mask(gt)
        return x, y, name


class PatchSegDataset(Dataset):
    """
    Patch-based dataset: returns random patches (CPU/RAM-friendly).
    x has 4 channels (RGB + aux).
    """
    def __init__(self, root_dir: str, split_json: str, split: str, aux_dir: str,
                 patch: int = 128, patches_per_image: int = 8, seed: int = 42, aug: bool = True):
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self.gt_dir = self.root / "masks_gt"
        self.aux_dir = Path(aux_dir)
        self.patch = int(patch)
        self.ppi = int(patches_per_image)
        self.aug = bool(aug)
        self.rng = random.Random(int(seed))

        import json
        with open(split_json, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.files = splits[split]
        if len(self.files) == 0:
            raise ValueError(f"Split '{split}' bo'sh. splits.json ni tekshiring.")

    def __len__(self):
        return len(self.files) * self.ppi

    def _pad(self, img, gt, aux):
        H, W = gt.shape[:2]
        p = self.patch
        if H >= p and W >= p:
            return img, gt, aux
        pad_h = max(0, p - H)
        pad_w = max(0, p - W)
        border = cv2.BORDER_REFLECT_101
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, border)
        gt  = cv2.copyMakeBorder(gt,  0, pad_h, 0, pad_w, border)
        aux = cv2.copyMakeBorder(aux, 0, pad_h, 0, pad_w, border)
        return img, gt, aux

    def _augment(self, img, gt, aux):
        if self.rng.random() < 0.5:
            img = img[:, ::-1, :]
            gt  = gt[:, ::-1]
            aux = aux[:, ::-1]
        if self.rng.random() < 0.5:
            alpha = 0.9 + 0.2 * self.rng.random()
            beta  = -10 + 20 * self.rng.random()
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        return img, gt, aux

    def __getitem__(self, idx):
        name = self.files[idx % len(self.files)]  # image jpg
        img = read_image_rgb(str(self.images_dir / name))

        mname = _mask_name_from_image(name, ".png")
        gt = read_mask_01(str(self.gt_dir / mname))

        aux_path = self.aux_dir / mname
        if aux_path.exists():
            aux = read_mask_01(str(aux_path))
        else:
            aux = np.zeros(gt.shape, dtype=np.float32)

        # align sizes
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if aux.shape[:2] != img.shape[:2]:
            aux = cv2.resize(aux, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img, gt, aux = self._pad(img, gt, aux)

        if self.aug:
            img, gt, aux = self._augment(img, gt, aux)

        H, W = gt.shape[:2]
        p = self.patch
        y0 = self.rng.randint(0, H - p)
        x0 = self.rng.randint(0, W - p)

        img_p = img[y0:y0+p, x0:x0+p, :]
        gt_p  = gt[y0:y0+p, x0:x0+p]
        aux_p = aux[y0:y0+p, x0:x0+p]

        x = torch.cat([to_tensor_img(img_p), to_tensor_mask(aux_p)], dim=0)  # 4,p,p
        y = to_tensor_mask(gt_p)
        return x, y, name