from pathlib import Path
import json
import numpy as np
import cv2
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

TRAIN_RESIZE = 24
PRED_RESIZE  = 32
POS_PER_IMG  = 2
NEG_PER_IMG  = 2
MAX_TRAIN_IMAGES = 12
SEED = 42

_MODEL = None  # (svc, psi_train)

def _load_train_list():
    sp = Path("data/splits.json")
    if sp.exists():
        splits = json.loads(sp.read_text(encoding="utf-8"))
        return splits.get("train", [])
    imgs = sorted([p.name for p in Path("data/images").iterdir()
                   if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    return imgs

def _pseudo_mask01(img_rgb_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    seg = cv2.bitwise_or(binary, edges)
    return (seg > 0).astype(np.uint8)

def _sample_pixels(img_rgb_u8, mask01, rng):
    pos = np.argwhere(mask01 == 1)
    neg = np.argwhere(mask01 == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    npos = min(POS_PER_IMG, len(pos))
    nneg = min(NEG_PER_IMG, len(neg))
    pos_sel = pos[rng.choice(len(pos), size=npos, replace=False)]
    neg_sel = neg[rng.choice(len(neg), size=nneg, replace=False)]
    X_pos = img_rgb_u8[pos_sel[:,0], pos_sel[:,1], :].astype(np.float32) / 255.0
    X_neg = img_rgb_u8[neg_sel[:,0], neg_sel[:,1], :].astype(np.float32) / 255.0
    y_pos = np.ones((len(X_pos),), dtype=np.int32)
    y_neg = np.zeros((len(X_neg),), dtype=np.int32)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    return X, y

def _statevector_from_x(x3: np.ndarray) -> np.ndarray:
    qc = QuantumCircuit(3)
    for i, xi in enumerate(x3):
        qc.ry(np.pi * float(xi), i)
        qc.rz(np.pi * float(xi), i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return Statevector.from_instruction(qc).data

def _states_for_X(X: np.ndarray) -> np.ndarray:
    states = np.zeros((len(X), 8), dtype=np.complex128)
    for i in range(len(X)):
        states[i, :] = _statevector_from_x(X[i])
    return states

def _kernel(psiA: np.ndarray, psiB: np.ndarray) -> np.ndarray:
    inner = psiA @ np.conjugate(psiB).T
    return np.abs(inner) ** 2

def _train_once():
    rng = np.random.default_rng(SEED)
    train_files = _load_train_list()
    X_all, y_all = [], []
    used = 0
    for name in train_files:
        img_bgr = cv2.imread(str(Path("data/images")/name))
        if img_bgr is None:
            continue
        img_s = cv2.resize(img_bgr, (TRAIN_RESIZE, TRAIN_RESIZE), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        pm = _pseudo_mask01(img_rgb)
        X, y = _sample_pixels(img_rgb, pm, rng)
        if X is None:
            continue
        X_all.append(X); y_all.append(y)
        used += 1
        if used >= MAX_TRAIN_IMAGES:
            break
    Xtr = np.vstack(X_all).astype(np.float32)
    ytr = np.concatenate(y_all).astype(np.int32)
    print(f"[quantum-fast] Train samples: {len(ytr)} (from {used} images)")
    psi_tr = _states_for_X(Xtr)
    Ktr = _kernel(psi_tr, psi_tr)
    svc = SVC(kernel="precomputed")
    svc.fit(Ktr, ytr)
    print("[quantum-fast] Model tayyor.")
    return svc, psi_tr

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _train_once()
    return _MODEL

def quantum_mask(img_bgr: np.ndarray) -> np.ndarray:
    svc, psi_tr = _get_model()
    H0, W0 = img_bgr.shape[:2]
    img_s = cv2.resize(img_bgr, (PRED_RESIZE, PRED_RESIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    psi_te = _states_for_X(X)
    Kte = _kernel(psi_te, psi_tr)
    yhat = svc.predict(Kte).astype(np.uint8)
    mask_small = (yhat.reshape(PRED_RESIZE, PRED_RESIZE) * 255).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, k, iterations=1)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.resize(mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)