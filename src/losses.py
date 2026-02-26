import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(probs, targets, eps=1e-7):
    # probs, targets: [B,1,H,W] in {0,1}
    probs = probs.contiguous()
    targets = targets.contiguous()
    inter = (probs * targets).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: [B,1,H,W], targets: [B,1,H,W] float 0/1
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, targets)
        dice_loss = 1.0 - dice
        return self.bce_weight*bce + (1.0-self.bce_weight)*dice_loss
    
    # ---- Added for compatibility: BCEDice loss ----
import torch
import torch.nn as nn
import torch.nn.functional as F

def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    logits: [B,1,H,W] (raw)
    targets: [B,1,H,W] (0/1 float)
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class BCEDice(nn.Module):
    """
    BCEWithLogits + Dice loss
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = self.bce(logits, targets)
        dice = _dice_loss_from_logits(logits, targets, smooth=self.smooth)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice