import json, random
from pathlib import Path

random.seed(42)

img_dir = Path("data/images")
names = sorted([p.name for p in img_dir.glob("*.jpg") if not p.name.startswith("._")])

random.shuffle(names)

n = len(names)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

splits = {
    "train": names[:n_train],
    "val":   names[n_train:n_train + n_val],
    "test":  names[n_train + n_val:],
}

Path("data").mkdir(exist_ok=True)
Path("data/splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

print("Splits ready:", {k: len(v) for k, v in splits.items()})
print("Example:", splits["train"][:5])