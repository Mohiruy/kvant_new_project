import argparse
from pathlib import Path
import torch

from src.utils import set_seed
from src.models.unet import UNet
from src.train import train_model
from src.datasets import SegDataset, PatchSegDataset


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--aux_dir", required=True)
    ap.add_argument("--run_name", required=True)

    # full-image mode
    ap.add_argument("--resize", type=int, default=256)

    # patch mode (NEW)
    ap.add_argument("--patch", type=int, default=0, help="0 -> full image, >0 -> patch training (masalan 128)")
    ap.add_argument("--ppi", type=int, default=8, help="train: patches per image")
    ap.add_argument("--val_ppi", type=int, default=2, help="val: patches per image")
    ap.add_argument("--no_aug", action="store_true", help="patch augmentationni ochirish")

    # train
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels (CPU uchun 16 tavsiya)")
    args = ap.parse_args()

    set_seed(args.seed)

    # dataset
    if args.patch and args.patch > 0:
        train_ds = PatchSegDataset(
            root_dir=args.data_dir, split_json=args.splits, split="train",
            aux_dir=args.aux_dir,
            patch=args.patch, patches_per_image=args.ppi, seed=args.seed,
            aug=(not args.no_aug)
        )
        val_ds = PatchSegDataset(
            root_dir=args.data_dir, split_json=args.splits, split="val",
            aux_dir=args.aux_dir,
            patch=args.patch, patches_per_image=args.val_ppi, seed=args.seed + 999,
            aug=False
        )
        print(f"[DATA] Patch mode: patch={args.patch}, ppi={args.ppi}, val_ppi={args.val_ppi}")
    else:
        train_ds = SegDataset(
            root_dir=args.data_dir, split_json=args.splits, split="train",
            aux_dir=args.aux_dir, resize=args.resize
        )
        val_ds = SegDataset(
            root_dir=args.data_dir, split_json=args.splits, split="val",
            aux_dir=args.aux_dir, resize=args.resize
        )
        print(f"[DATA] Full-image mode: resize={args.resize}")

    # model: RGB+aux = 4 kanalli
    model = UNet(in_channels=4, out_channels=1, base=args.base)

    out_dir = Path("outputs/models") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        out_dir=str(out_dir),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr)

    print("Best:", best)


if __name__ == "__main__":
    main()





