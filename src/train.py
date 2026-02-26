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
