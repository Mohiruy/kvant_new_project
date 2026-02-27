import os
import cv2
import numpy as np

SRC = "data/trimaps"
DST = "data/masks_gt"

os.makedirs(DST, exist_ok=True)

for fname in os.listdir(SRC):
    if not fname.endswith(".png"):
        continue
    if fname.startswith("._"):
        continue

    m = cv2.imread(os.path.join(SRC, fname), cv2.IMREAD_GRAYSCALE)
    binary = np.where(m == 3, 0, 1).astype(np.uint8)

    cv2.imwrite(os.path.join(DST, fname), binary * 255)

print("Masks ready.")