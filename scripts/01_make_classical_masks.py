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
