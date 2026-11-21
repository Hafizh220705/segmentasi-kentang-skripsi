"""Preprocessing sekali jalan: letterbox citra dan mask ke IMG_SIZE, lalu simpan ke cache.

Alasan cache: citra asli sekitar 1.700 px. Membaca dan me-resize ulang setiap
epoch lambat dan membuang waktu GPU. Hasil cache juga menjamin semua eksperimen
memakai input yang identik.

Alur per citra:
    citra asli (H, W, 3) --letterbox--> (S, S, 3)        disimpan .png
    label poligon --rasterisasi resolusi asli--> (4, H, W)
                  --letterbox nearest--> (4, S, S)        disimpan .npz
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C
from .data import foreground_mask, parse_label_file, rasterize_instance, read_image_bgr

SCAB = 0

# Geometri letterbox
def letterbox_geometry(height, width, size=C.IMG_SIZE):
    """Menghitung skala dan posisi citra di kanvas persegi `size`.

    Sisi terpanjang diubah menjadi `size`, rasio aspek dipertahankan,
    lalu citra diletakkan di tengah. Sisa kanvas diisi 0 (hitam).

    Return dict: scale, new_w, new_h, pad_x, pad_y
    pad_x/pad_y = jumlah piksel kosong di kiri/atas.
    Disimpan di meta supaya prediksi bisa dikembalikan ke resolusi asli (prototipe).
    """
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return {"scale": scale, "new_w": new_w, "new_h": new_h,
            "pad_x": (size - new_w) // 2, "pad_y": (size - new_h) // 2}

def letterbox(array, geom, size=C.IMG_SIZE, interpolation=cv2.INTER_AREA):
    """Resize `array` (H, W) atau (H, W, C) sesuai geom, lalu tempel di kanvas (size, size).

    interpolation:
        INTER_AREA    untuk citra (mengecilkan tanpa aliasing)
        INTER_NEAREST untuk mask (nilai tetap 0/1, tidak ada pecahan)
    """
    resized = cv2.resize(array, (geom["new_w"], geom["new_h"]),   # cv2: (lebar, tinggi)
                         interpolation=interpolation)
    if resized.ndim == 2 and array.ndim == 3:                     # cv2 membuang dim C=1
        resized = resized[..., None]
    canvas = np.zeros((size, size) + array.shape[2:], dtype=array.dtype)
    y0, x0 = geom["pad_y"], geom["pad_x"]
    canvas[y0:y0 + geom["new_h"], x0:x0 + geom["new_w"]] = resized
    return canvas

# Mask
def build_masks(instances, height, width, fg_area, giant_threshold=C.GIANT_COVERAGE):
    """Mask multilabel resolusi asli, dua versi.

    Return:
        mask         : uint8 (H, W, 4)  semua instance (target utama)
        scab_nogiant : uint8 (H, W)     kanal Scab tanpa instance coverage >= ambang
                                        (dipakai hanya jika IGNORE_GIANT_SCAB = True)
    Disusun (H, W, C) karena cv2.resize dan albumentations memakai urutan itu.
    Logika OR per kanal sama dengan data.build_multilabel_mask.
    """
    mask = np.zeros((height, width, C.NUM_CLASSES), dtype=bool)
    scab_nogiant = np.zeros((height, width), dtype=bool)
    for inst in instances:
        m = rasterize_instance(inst, height, width)
        mask[..., inst.class_id] |= m
        if inst.class_id == SCAB:
            coverage = m.sum() / fg_area if fg_area > 0 else 0.0
            if coverage < giant_threshold:
                scab_nogiant |= m
    return mask.astype(np.uint8), scab_nogiant.astype(np.uint8)

# Simpan
def save_png(path, img_bgr):
    """Menyimpan PNG lossless. imencode + tofile aman untuk path non-ASCII di Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError(f"Gagal encode PNG: {path}")
    buf.tofile(str(path))

def cache_paths(split, stem):
    """Path cache citra (.png) dan mask (.npz) untuk satu file."""
    return (C.PROC_IMG_DIR / split / f"{stem}.png",
            C.PROC_MASK_DIR / split / f"{stem}.npz")

def process_one(split, stem, size=C.IMG_SIZE):
    """Memproses satu citra: letterbox citra dan mask, simpan, kembalikan baris meta."""
    img = read_image_bgr(C.IMAGE_DIR / split / f"{stem}.png")
    h, w = img.shape[:2]
    instances, issues = parse_label_file(C.LABEL_DIR / split / f"{stem}.txt")
    if issues:
        raise ValueError(f"{split}/{stem}: label tidak valid {issues}")

    fg_area = int(foreground_mask(img).sum())
    mask, scab_ng = build_masks(instances, h, w, fg_area)

    geom = letterbox_geometry(h, w, size)
    img_lb = letterbox(img, geom, size, cv2.INTER_AREA)
    mask_lb = letterbox(mask, geom, size, cv2.INTER_NEAREST)
    scab_ng_lb = letterbox(scab_ng, geom, size, cv2.INTER_NEAREST)

    img_path, mask_path = cache_paths(split, stem)
    save_png(img_path, img_lb)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, mask=mask_lb, scab_nogiant=scab_ng_lb)

    row = {"split": split, "file": stem, "orig_h": h, "orig_w": w, **geom}
    for c in range(C.NUM_CLASSES):
        # QC: kelas yang ada di resolusi asli tetapi hilang setelah resize
        row[f"px_orig_{c}"] = int(mask[..., c].sum())
        row[f"px_lb_{c}"] = int(mask_lb[..., c].sum())
    return row

def build_cache(manifest, size=C.IMG_SIZE, overwrite=False):
    """Membuat cache untuk semua citra dengan keep == True di manifest.

    Sebuah citra dilewati hanya jika file cache DAN baris meta-nya sudah ada,
    sehingga proses yang terputus bisa dilanjutkan tanpa mengulang dari awal.
    overwrite=True memproses ulang semuanya (misalnya setelah IMG_SIZE diubah).
    Return DataFrame meta (hanya citra keep), juga disimpan ke C.PROC_META_PATH.
    """
    old = None
    if C.PROC_META_PATH.exists() and not overwrite:
        old = pd.read_csv(C.PROC_META_PATH)
    done = set() if old is None else set(zip(old["split"], old["file"]))

    todo = manifest[manifest["keep"]]
    rows = []
    for r in tqdm(todo.itertuples(), total=len(todo), desc=f"cache {size}px"):
        img_path, mask_path = cache_paths(r.split, r.file)
        if (r.split, r.file) in done and img_path.exists() and mask_path.exists():
            continue
        rows.append(process_one(r.split, r.file, size))

    meta = pd.DataFrame(rows)
    if old is not None:
        keys = set(zip(meta["split"], meta["file"])) if len(meta) else set()
        old = old[[k not in keys for k in zip(old["split"], old["file"])]]
        meta = pd.concat([old, meta], ignore_index=True)

    # Hanya simpan citra yang masih keep (jika manifest berubah, baris lama dibuang)
    keep_keys = set(zip(todo["split"], todo["file"]))
    meta = meta[[k in keep_keys for k in zip(meta["split"], meta["file"])]]
    meta = meta.sort_values(["split", "file"]).reset_index(drop=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(C.PROC_META_PATH, index=False)
    return meta

def class_loss_report(meta):
    """QC: jumlah citra yang kelasnya hilang total setelah letterbox, per kelas per split."""
    rows = []
    for split in C.SPLITS:
        s = meta[meta["split"] == split]
        for c, name in enumerate(C.CLASS_NAMES):
            present = s[f"px_orig_{c}"] > 0
            lost = present & (s[f"px_lb_{c}"] == 0)
            rows.append({"split": split, "kelas": name,
                         "citra_berisi_kelas": int(present.sum()),
                         "citra_kelas_hilang": int(lost.sum())})
    return pd.DataFrame(rows)