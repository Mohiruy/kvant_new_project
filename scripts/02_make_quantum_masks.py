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
