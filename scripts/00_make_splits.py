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
