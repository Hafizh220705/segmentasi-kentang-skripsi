"""Dataset dan DataLoader PyTorch untuk segmentasi multilabel 4 kanal.

Membaca cache dari src/preprocess.py (citra dan mask sudah 512 x 512),
menerapkan augmentasi geometrik (train saja) dan normalisasi ImageNet.

Keluaran satu sampel (dict):
    image : float32 tensor (3, S, S), sudah dinormalisasi
    mask  : float32 tensor (4, S, S), nilai 0 atau 1, satu kanal per kelas
    split, file : string, untuk penelusuran
"""
import random
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from . import config as C
from .preprocess import cache_paths

SCAB = 0

# Transformasi
def build_transforms(train):
    """Pipeline albumentations.

    Train: hanya augmentasi GEOMETRIK. Warna tidak diubah karena Greening dikenali
    dari warna hijau dan citra sudah Green-Boosted oleh penyusun dataset.
    Albumentations menerapkan transformasi geometrik yang sama ke citra dan mask.
    Val/test: tanpa augmentasi, hanya normalisasi.
    """
    steps = []
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=0,
                     interpolation=cv2.INTER_LINEAR,       # citra: halus
                     mask_interpolation=cv2.INTER_NEAREST, # mask: tetap 0/1
                     border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
        ]
    steps += [
        A.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD),   # input RGB 0-255
        ToTensorV2(transpose_mask=True),   # mask (H, W, 4) -> (4, H, W)
    ]
    return A.Compose(steps)

#  Dataset
def split_files(split, manifest_path=C.MANIFEST_PATH):
    """Daftar nama file yang dipakai (keep == True) untuk satu split, terurut."""
    m = pd.read_csv(manifest_path)
    files = m.loc[(m["split"] == split) & (m["keep"]), "file"]
    return sorted(files.tolist())

def load_cached(split, stem, ignore_giant_scab=C.IGNORE_GIANT_SCAB):
    """Membaca citra RGB uint8 (S, S, 3) dan mask uint8 (S, S, 4) dari cache."""
    img_path, mask_path = cache_paths(split, stem)
    buf = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cache citra tidak ada atau rusak: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # normalisasi ImageNet mengharapkan RGB

    with np.load(mask_path) as z:
        mask = z["mask"].copy()
        if ignore_giant_scab:
            mask[..., SCAB] = z["scab_nogiant"]
    return img, mask

class PotatoDefectDataset(Dataset):
    """Dataset segmentasi multilabel cacat kentang.

    split             : "train", "val", atau "test"
    train             : True untuk augmentasi (hanya untuk split train)
    ignore_giant_scab : saklar eksperimen, lihat config.IGNORE_GIANT_SCAB
    files             : opsional, daftar file tertentu (misal untuk evaluasi subset)
    """

    def __init__(self, split, train=False, ignore_giant_scab=C.IGNORE_GIANT_SCAB, files=None):
        if train and split != "train":
            raise ValueError("Augmentasi hanya boleh untuk split train.")
        self.split = split
        self.files = list(files) if files is not None else split_files(split)
        self.ignore_giant_scab = ignore_giant_scab
        self.transform = build_transforms(train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx]
        img, mask = load_cached(self.split, stem, self.ignore_giant_scab)
        out = self.transform(image=img, mask=mask)
        return {
            "image": out["image"].float(),
            "mask": out["mask"].float(),   # BCEWithLogitsLoss butuh float
            "split": self.split,
            "file": stem,
        }

# DataLoader
def seed_everything(seed=C.SEED):
    """Seed Python, NumPy, dan PyTorch (CPU dan GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _worker_init(worker_id):
    """Seed setiap worker berbeda tetapi tetap reprodusibel (hanya berlaku jika num_workers > 0)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

def build_loader(dataset, shuffle, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS,
                 seed=C.SEED):
    """DataLoader dengan generator ber-seed, supaya urutan acak bisa diulang."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=_worker_init,
                      generator=g, pin_memory=torch.cuda.is_available(),
                      drop_last=False)

def denormalize(image_tensor):
    """Kebalikan normalisasi ImageNet. (3, S, S) tensor -> (S, S, 3) uint8 RGB, untuk visualisasi."""
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)