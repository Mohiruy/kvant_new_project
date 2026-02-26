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
