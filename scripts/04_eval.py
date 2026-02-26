import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils import set_seed
from src.models.unet import UNet
from src.datasets import SegDataset, PatchSegDataset


def dice_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-6):
    p = (pred01 > 0.5).astype(np.uint8)
    g = (gt01 > 0.5).astype(np.uint8)
    inter = (p & g).sum()
    union = (p | g).sum()
    dice = (2.0 * inter + eps) / (p.sum() + g.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def load_model_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # If checkpoint dict has 'model', use it; otherwise assume it's already a state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--aux_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)

    # full-image eval
    ap.add_argument("--resize", type=int, default=256)

    # patch eval
    ap.add_argument("--patch", type=int, default=0, help="0 -> full image eval, >0 -> patch eval")
    ap.add_argument("--ppi", type=int, default=4, help="patches per image for eval (if patch>0)")

    # model
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.patch and args.patch > 0:
        ds = PatchSegDataset(
            root_dir=args.data_dir, split_json=args.splits, split=args.split,
            aux_dir=args.aux_dir,
            patch=args.patch, patches_per_image=args.ppi, seed=args.seed,
            aug=False
        )
        print(f"[EVAL] Patch mode: patch={args.patch} ppi={args.ppi} split={args.split}")
    else:
        ds = SegDataset(
            root_dir=args.data_dir, split_json=args.splits, split=args.split,
            aux_dir=args.aux_dir, resize=args.resize
        )
        print(f"[EVAL] Full-image mode: resize={args.resize} split={args.split}")

    model = UNet(in_channels=4, out_channels=1, base=args.base)
    sd = load_model_ckpt(args.ckpt)
    model.load_state_dict(sd)
    model.eval()

    rows = []
    with torch.no_grad():
        for x, y, name in tqdm(ds, total=len(ds), desc="eval"):
            # dataset returns x:[4,H,W], y:[1,H,W]
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if y.ndim == 3:
                y = y.unsqueeze(0)

            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs >= args.thr).float()

            pred01 = pred[0, 0].cpu().numpy()
            gt01 = y[0, 0].cpu().numpy()

            d, i = dice_iou(pred01, gt01)
            rows.append({"name": name, "dice": d, "iou": i})

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print("Mean dice=%.4f  Mean iou=%.4f" % (df["dice"].mean(), df["iou"].mean()))


if __name__ == "__main__":
    main()
