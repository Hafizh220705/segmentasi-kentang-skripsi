"""Uji Modul 2: letterbox, mask, dan augmentasi. Jalankan dari root proyek: python -m pytest"""

import random
import cv2
import numpy as np
from src.data import Instance
from src.dataset import build_transforms, denormalize
from src.preprocess import build_masks, letterbox, letterbox_geometry

def square(class_id, x0, y0, x1, y1):
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return Instance(class_id, pts, line_no=1)

# Letterbox
def test_geometri_letterbox_citra_lebar():
    g = letterbox_geometry(height=1000, width=2000, size=512)
    assert (g["new_w"], g["new_h"]) == (512, 256)    # sisi terpanjang = 512
    assert (g["pad_x"], g["pad_y"]) == (0, 128)      # sisa dibagi rata atas-bawah

def test_letterbox_padding_hitam_dan_isi_di_tengah():
    img = np.full((100, 200, 3), 200, dtype=np.uint8)
    g = letterbox_geometry(100, 200, size=64)
    out = letterbox(img, g, size=64)
    assert out.shape == (64, 64, 3)
    assert out[:g["pad_y"]].max() == 0                      # padding atas hitam
    assert out[g["pad_y"] + g["new_h"]:].max() == 0         # padding bawah hitam
    assert out[32, 32].min() == 200                         # tengah berisi citra

def test_letterbox_mask_tetap_biner():
    mask = np.zeros((300, 500, 4), dtype=np.uint8)
    mask[50:200, 100:300, 0] = 1
    g = letterbox_geometry(300, 500, size=128)
    out = letterbox(mask, g, size=128, interpolation=cv2.INTER_NEAREST)
    assert out.shape == (128, 128, 4)
    assert set(np.unique(out)) <= {0, 1}                    # nearest: tidak ada pecahan

#  Mask
def test_build_masks_overlap_dan_scab_raksasa():
    h, w = 100, 100
    insts = [square(0, 0.0, 0.0, 1.0, 1.0),     # Scab menutupi seluruh citra (raksasa)
             square(0, 0.1, 0.1, 0.2, 0.2),     # Scab kecil
             square(3, 0.5, 0.5, 0.7, 0.7)]     # Mechanical Damage di dalam Scab raksasa
    mask, scab_ng = build_masks(insts, h, w, fg_area=h * w, giant_threshold=0.9)
    assert mask.shape == (h, w, 4)
    assert (mask[..., 0] & mask[..., 3]).sum() > 0          # overlap aktif di kedua kanal
    assert scab_ng.sum() < mask[..., 0].sum()               # versi tanpa raksasa lebih kecil
    assert scab_ng[15, 15] == 1 and scab_ng[60, 60] == 0    # Scab kecil tetap, raksasa hilang

#  Augmentasi
def test_augmentasi_sinkron_citra_dan_mask():
    """Area merah di citra harus tetap tepat berada di area mask setelah augmentasi."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64, 4), dtype=np.uint8)
    img[10:30, 5:20] = (255, 0, 0)
    mask[10:30, 5:20, 2] = 1
    tf = build_transforms(train=True)
    for seed in range(20):                                  # coba berbagai augmentasi acak
        np.random.seed(seed)
        random.seed(seed)
        out = tf(image=img, mask=mask)
        assert out["image"].shape == (3, 64, 64)
        assert out["mask"].shape == (4, 64, 64)             # (C, H, W)
        m = out["mask"].numpy()
        assert set(np.unique(m)) <= {0, 1}
        red = denormalize(out["image"])[..., 0] > 127
        assert (red == m[2].astype(bool)).mean() > 0.98     # posisi tetap sinkron

def test_transform_val_tidak_mengubah_posisi():
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32, 4), dtype=np.uint8)
    mask[5:10, 5:10, 1] = 1
    out = build_transforms(train=False)(image=img, mask=mask)
    assert np.array_equal(out["mask"].numpy()[1], mask[..., 1])
    assert np.abs(denormalize(out["image"]).astype(int) - img).max() <= 1