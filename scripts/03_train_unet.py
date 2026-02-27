import argparse, json
from pathlib import Path
from src.utils import set_seed
from src.datasets import SegDataset
from src.models.unet import UNet
from src.train import train_model
from src.data.patch_dataset import PatchSegDataset
from src.losses import DiceBCELoss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_dir", required=True)  # outputs/classical_masks ёки outputs/quantum_masks
    ap.add_argument("--run_name", required=True) # classic_unet ёки quantum_unet
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(42)
    splits = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))

    train_ds = SegDataset("data", "data/splits.json", "train", args.aux_dir, resize=args.resize)
    val_ds   = SegDataset("data", "data/splits.json", "val",   args.aux_dir, resize=args.resize)

    model = UNet(in_channels=4, out_channels=1, base=32)
    out_dir = Path("outputs/models") / args.run_name
    best = train_model(model, train_ds, val_ds, str(out_dir),
                       epochs=args.epochs, lr=args.lr, batch=args.batch)
    print("Best:", best)

if __name__ == "__main__":
    main()
