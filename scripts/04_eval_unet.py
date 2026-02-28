import argparse
import json
from pathlib import Path
import time

import torch
import pandas as pd
from tqdm import tqdm

from src.models.unet import UNet
from src.datasets import SegDataset


def dice_iou_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5, eps: float = 1e-7):
    """
    logits: (B,1,H,W)
    y:      (B,1,H,W) float {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()

    inter = (pred * y).sum(dim=(1, 2, 3))
    union = (pred + y - pred * y).sum(dim=(1, 2, 3))

    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best.pt/last.pt")
    ap.add_argument("--aux_dir", required=True, help="outputs/classical_masks or outputs/quantum_masks")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out_csv", default="outputs/eval.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # dataset
    split_json = "data/splits.json"
    ds = SegDataset("data", split_json, args.split, args.aux_dir, resize=args.resize)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # model
    model = UNet(in_channels=4, out_channels=1, base=32).to(device)
    ckpt = torch.load(args.model, map_location=device)
    # support both {"model": state_dict} and raw state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    for x, y, names in tqdm(dl, desc=f"eval-{args.split}"):
        x = x.to(device)
        y = y.to(device)

        t0 = time.perf_counter()
        logits = model(x)
        dt = (time.perf_counter() - t0) * 1000.0  # ms for the batch

        dice, iou = dice_iou_from_logits(logits, y, thr=args.thr)
        ms_per_img = dt / x.shape[0]

        for n, d, j in zip(names, dice.detach().cpu().tolist(), iou.detach().cpu().tolist()):
            rows.append({"file": n, "dice": d, "iou": j, "inference_ms": ms_per_img})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print("Saved:", out_csv)
    print("Mean Dice:", float(df["dice"].mean()))
    print("Mean IoU :", float(df["iou"].mean()))


if __name__ == "__main__":
    main()