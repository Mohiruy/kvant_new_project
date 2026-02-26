import os
import random
import numpy as np
import cv2
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_image_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def read_mask_01(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m01 = (m > 127).astype(np.uint8)
    return m01

def resize_img_mask(img, mask01, size: int):
    if size is None:
        return img, mask01
    img2 = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m2   = cv2.resize(mask01, (size, size), interpolation=cv2.INTER_NEAREST)
    return img2, m2

def bgr_to_tensor(img_bgr):
    # (H,W,3) uint8 -> (3,H,W) float32 [0..1]
    x = img_bgr.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))
    return torch.from_numpy(x)

def mask01_to_tensor(mask01):
    # (H,W) 0/1 -> (1,H,W) float32
    y = mask01.astype(np.float32)[None, :, :]
    return torch.from_numpy(y)
