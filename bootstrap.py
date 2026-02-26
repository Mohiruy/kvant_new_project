from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent

def write(rel_path: str, content: str):
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

def main():
    # --- папкалар ---
    for d in [
        "data/images", "data/masks_gt",
        "outputs/classical_masks", "outputs/quantum_masks", "outputs/models",
        "src", "src/models", "src/masks",
        "scripts",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    # --- requirements.txt ---
    write("requirements.txt", """
    numpy
    pandas
    tqdm
    opencv-python
    scikit-learn
    torch
    torchvision
    """)

    # --- src/__init__.py ---
    write("src/__init__.py", "")
    write("src/models/__init__.py", "")
    write("src/masks/__init__.py", "")

    # --- utils.py ---
    write("src/utils.py", """
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
    """)

    # --- classical masks (kmeans/otsu) ---
    write("src/masks/classical.py", """
    import cv2
    import numpy as np

    def _clean(mask255):
        k = np.ones((3,3), np.uint8)
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, k, iterations=1)
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, k, iterations=2)
        return mask255

    def otsu_mask(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _clean(th)

    def kmeans_mask(img_bgr, k=2, use_lab=True):
        img = cv2.medianBlur(img_bgr, 3)
        if use_lab:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        Z = x.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(img.shape[:2])

        if use_lab:
            # L канали кичик бўлган кластерни объект деб фараз (қоронғироқ объект)
            obj = int(np.argmin(centers[:,0]))
        else:
            # HSV: S катта бўлган кластер объект (оддий фараз)
            obj = int(np.argmax(centers[:,1]))

        m = (labels == obj).astype(np.uint8) * 255
        return _clean(m)
    """)

    # --- quantum mask placeholder ---
    write("src/masks/quantum.py", """
    import numpy as np

    def quantum_mask(img_bgr) -> np.ndarray:
        \"""
        СИЗНИНГ КВАНТ СЕГМЕНТАЦИЯНГИЗ ШУ ЕРГА ҚЎЙИЛАДИ.

        КИРИШ:  img_bgr (H,W,3) uint8
        ЧИҚИШ:  mask255 (H,W) uint8 0/255

        Ҳозирча мисол учун NotImplemented чиқаради.
        \"""
        raise NotImplementedError("quantum_mask() функциясини ўзингизнинг квант коди билан тўлдиринг.")
    """)

    # --- simple U-Net ---
    write("src/models/unet.py", """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            dy = x2.size(2) - x1.size(2)
            dx = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=4, out_channels=1, base=32):
            super().__init__()
            self.inc = DoubleConv(in_channels, base)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16))

            self.up1 = Up(base*16, base*8)
            self.up2 = Up(base*8,  base*4)
            self.up3 = Up(base*4,  base*2)
            self.up4 = Up(base*2,  base)
            self.outc = nn.Conv2d(base, out_channels, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x,  x3)
            x = self.up3(x,  x2)
            x = self.up4(x,  x1)
            return self.outc(x)
    """)

    # --- dataset: (RGB + auxmask) -> gt ---
    write("src/datasets.py", """
    from pathlib import Path
    import cv2
    import torch
    from torch.utils.data import Dataset
    from .utils import read_image_bgr, read_mask_01, resize_img_mask, bgr_to_tensor, mask01_to_tensor

    class SegDataset(Dataset):
        def __init__(self, data_dir: str, files: list, aux_dir: str, resize: int = 256):
            self.data = Path(data_dir)
            self.images = self.data / "images"
            self.gts    = self.data / "masks_gt"
            self.aux    = Path(aux_dir)
            self.files = files
            self.resize = resize

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            name = self.files[i]
            img = read_image_bgr(str(self.images / name))
            gt  = read_mask_01(str(self.gts / name))
            aux = read_mask_01(str(self.aux / name))

            img, gt  = resize_img_mask(img, gt, self.resize)
            img, aux = resize_img_mask(img, aux, self.resize)

            x_img = bgr_to_tensor(img)      # (3,H,W)
            x_aux = mask01_to_tensor(aux)   # (1,H,W)
            y     = mask01_to_tensor(gt)    # (1,H,W)
            x = torch.cat([x_img, x_aux], dim=0)  # (4,H,W)
            return x, y, name
    """)

    # --- loss ---
    write("src/losses.py", """
    import torch
    import torch.nn as nn

    class DiceLoss(nn.Module):
        def __init__(self, eps=1e-6):
            super().__init__()
            self.eps = eps

        def forward(self, logits, targets):
            p = torch.sigmoid(logits)
            p = p.view(p.size(0), -1)
            t = targets.view(targets.size(0), -1)
            num = 2*(p*t).sum(dim=1) + self.eps
            den = p.sum(dim=1) + t.sum(dim=1) + self.eps
            return 1 - (num/den).mean()

    class BCEDice(nn.Module):
        def __init__(self, w=0.5):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()
            self.w = w

        def forward(self, logits, targets):
            return self.w*self.bce(logits, targets) + (1-self.w)*self.dice(logits, targets)
    """)

    # --- train loop ---
    write("src/train.py", """
    from pathlib import Path
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from .losses import BCEDice

    def train_model(model, train_ds, val_ds, out_dir: str, epochs=20, lr=1e-3, batch=8):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = BCEDice(0.5)

        best = -1.0
        for ep in range(1, epochs+1):
            model.train()
            tr = 0.0
            for x, y, _ in tqdm(train_loader, desc=f"train {ep}/{epochs}"):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                tr += float(loss.item())
            tr /= max(1, len(train_loader))

            model.eval()
            dices = []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    p = (torch.sigmoid(logits) > 0.5).float()
                    dice = (2*(p*y).sum() + 1e-9) / (p.sum() + y.sum() + 1e-9)
                    dices.append(float(dice.item()))
            val_dice = float(np.mean(dices)) if dices else 0.0

            ckpt = {"model": model.state_dict(), "val_dice": val_dice, "epoch": ep}
            torch.save(ckpt, out/"last.pt")
            if val_dice > best:
                best = val_dice
                torch.save(ckpt, out/"best.pt")

            print(f"[ep {ep}] train_loss={tr:.4f} val_dice={val_dice:.4f} best={best:.4f}")

        return str(out/"best.pt")
    """)

    # --- scripts ---
    write("scripts/00_make_splits.py", """
    import json, random
    from pathlib import Path

    def main():
        data = Path("data/images")
        files = sorted([p.name for p in data.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        if not files:
            raise SystemExit("data/images ичига тасвир қўйинг.")
        random.seed(42)
        random.shuffle(files)
        n = len(files)
        n_train = int(0.7*n)
        n_val   = int(0.15*n)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:],
        }
        Path("data/splits.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Saved data/splits.json:", {k: len(v) for k,v in splits.items()})

    if __name__ == "__main__":
        main()
    """)

    write("scripts/01_make_classical_masks.py", """
    import argparse
    from pathlib import Path
    import cv2
    from tqdm import tqdm
    from src.masks.classical import kmeans_mask, otsu_mask

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--method", choices=["kmeans","otsu"], default="kmeans")
        ap.add_argument("--out_dir", default="outputs/classical_masks")
        args = ap.parse_args()

        images = Path("data/images")
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in images.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        for p in tqdm(files, desc=f"classical {args.method}"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if args.method == "kmeans":
                m = kmeans_mask(img, k=2, use_lab=True)
            else:
                m = otsu_mask(img)
            cv2.imwrite(str(out/p.name), m)

        print("Saved:", out)

    if __name__ == "__main__":
        main()
    """)

    write("scripts/02_make_quantum_masks.py", """
    from pathlib import Path
    import cv2
    from tqdm import tqdm
    from src.masks.quantum import quantum_mask

    def main():
        images = Path("data/images")
        out = Path("outputs/quantum_masks")
        out.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in images.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        for p in tqdm(files, desc="quantum masks"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            m = quantum_mask(img)  # сиз бу функцияни тўлдирасиз
            cv2.imwrite(str(out/p.name), m)

        print("Saved:", out)

    if __name__ == "__main__":
        main()
    """)

    write("scripts/03_train_unet.py", """
    import argparse, json
    from pathlib import Path
    from src.utils import set_seed
    from src.datasets import SegDataset
    from src.models.unet import UNet
    from src.train import train_model

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

        train_ds = SegDataset("data", splits["train"], args.aux_dir, resize=args.resize)
        val_ds   = SegDataset("data", splits["val"],   args.aux_dir, resize=args.resize)

        model = UNet(in_channels=4, out_channels=1, base=32)
        out_dir = Path("outputs/models") / args.run_name
        best = train_model(model, train_ds, val_ds, str(out_dir),
                           epochs=args.epochs, lr=args.lr, batch=args.batch)
        print("Best:", best)

    if __name__ == "__main__":
        main()
    """)

    write("scripts/04_eval.py", """
    import argparse, json, time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.datasets import SegDataset
    from src.models.unet import UNet

    def dice_iou(pred01, gt01):
        inter = (pred01 & gt01).sum()
        union = (pred01 | gt01).sum()
        dice = (2*inter + 1e-9) / (pred01.sum() + gt01.sum() + 1e-9)
        iou  = (inter + 1e-9) / (union + 1e-9)
        return float(dice), float(iou)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--aux_dir", required=True)
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--split", choices=["val","test"], default="test")
        ap.add_argument("--resize", type=int, default=256)
        ap.add_argument("--out_csv", required=True)
        args = ap.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        splits = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))
        files = splits[args.split]

        ds = SegDataset("data", files, args.aux_dir, resize=args.resize)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        model = UNet(in_channels=4, out_channels=1, base=32).to(device)
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        model.eval()

        rows = []
        with torch.no_grad():
            for x, y, name in loader:
                x, y = x.to(device), y.to(device)
                t0 = time.perf_counter()
                logits = model(x)
                t1 = time.perf_counter()

                prob = torch.sigmoid(logits).cpu().numpy()[0,0]
                pred01 = (prob > 0.5).astype(np.uint8)
                gt01   = (y.cpu().numpy()[0,0] > 0.5).astype(np.uint8)
                d, j = dice_iou(pred01, gt01)

                rows.append({"file": name[0], "dice": d, "iou": j, "inference_ms": (t1-t0)*1000.0})

        df = pd.DataFrame(rows)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False, encoding="utf-8")
        print("Saved:", args.out_csv)
        print("Mean dice=%.4f  Mean iou=%.4f  Mean ms=%.2f" % (df.dice.mean(), df.iou.mean(), df.inference_ms.mean()))

    if __name__ == "__main__":
        main()
    """)

    write("README.txt", """
    ҚИСҚА ЙЎРИҚНОМА

    1) python bootstrap.py
    2) pip install -r requirements.txt
    3) data/images ва data/masks_gt га файлларни қўйинг (номлари бир хил)
    4) python scripts/00_make_splits.py
    5) python scripts/01_make_classical_masks.py --method kmeans
    6) python scripts/03_train_unet.py --aux_dir outputs/classical_masks --run_name classic_unet
    7) python scripts/04_eval.py --aux_dir outputs/classical_masks --ckpt outputs/models/classic_unet/best.pt --out_csv outputs/classic.csv

    КВАНТ УЧУН:
    - src/masks/quantum.py да quantum_mask() ни ўз кодиңиз билан тўлдиринг
    - python scripts/02_make_quantum_masks.py
    - python scripts/03_train_unet.py --aux_dir outputs/quantum_masks --run_name quantum_unet
    - python scripts/04_eval.py --aux_dir outputs/quantum_masks --ckpt outputs/models/quantum_unet/best.pt --out_csv outputs/quantum.csv
    """)

    print("Тайёр! Энди қуйидагини бажаринг:")
    print("  1) python bootstrap.py")
    print("  2) pip install -r requirements.txt")

if __name__ == "__main__":
    main()