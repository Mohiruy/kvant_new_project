import argparse
from pathlib import Path
import random
import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def find_label(label_dir: Path, stem: str):
    # label одатда PNG бўлади
    for ext in [".png", ".jpg", ".jpeg"]:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_img", required=True)     # masalan D:\dataset\train\img
    ap.add_argument("--src_label", required=True)   # masalan D:\dataset\train\label
    ap.add_argument("--dst_images", default=r"data\images")
    ap.add_argument("--dst_masks", default=r"data\masks_gt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    # Танланган синф ранги (RGB)
    ap.add_argument("--rgb", default="128,64,128", help="масалан road: 128,64,128 (RGB)")
    ap.add_argument("--tol", type=int, default=5, help="ранг толеранси (±tol)")
    args = ap.parse_args()

    src_img = Path(args.src_img)
    src_lab = Path(args.src_label)
    dst_images = Path(args.dst_images); dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks  = Path(args.dst_masks);  dst_masks.mkdir(parents=True, exist_ok=True)

    r,g,b = [int(x) for x in args.rgb.split(",")]
    target_bgr = np.array([b,g,r], dtype=np.int16)

    rng = random.Random(args.seed)
    imgs = [p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS]
    rng.shuffle(imgs)

    saved = 0
    for p in imgs:
        stem = p.stem
        lab_path = find_label(src_lab, stem)
        if lab_path is None:
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(lab_path), cv2.IMREAD_COLOR)  # BGR
        if img is None or lab is None:
            continue

        # label'дан бинар маска: target рангига яқин пикселлар
        diff = np.abs(lab.astype(np.int16) - target_bgr[None, None, :])
        mask01 = (diff.max(axis=2) <= args.tol).astype(np.uint8)
        mask255 = mask01 * 255

        # кичик тозалаш
        k = np.ones((3,3), np.uint8)
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, k, iterations=1)
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, k, iterations=2)

        out_name = f"{stem}.png"
        cv2.imwrite(str(dst_images / out_name), img)
        cv2.imwrite(str(dst_masks  / out_name), mask255)

        saved += 1
        if saved >= args.n:
            break

    print("Saved pairs:", saved)
    print("->", dst_images)
    print("->", dst_masks)

if __name__ == "__main__":
    main()