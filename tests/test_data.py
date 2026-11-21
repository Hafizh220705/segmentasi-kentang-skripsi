"""Uji fungsi inti data.py. Jalankan dari root proyek: python -m pytest"""

import numpy as np
from src.data import Instance, build_multilabel_mask, green_boost, rasterize_instance

def square(class_id, x0, y0, x1, y1):
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return Instance(class_id, pts, line_no=1)

def test_luas_persegi():
    m = rasterize_instance(square(0, 0.25, 0.25, 0.75, 0.75), 100, 100)
    # fillPoly memasukkan tepi, jadi 51 x 51 = 2601 piksel
    assert 2500 <= m.sum() <= 2601

def test_overlap_aktif_di_kedua_kanal():
    insts = [square(0, 0.1, 0.1, 0.6, 0.6), square(3, 0.4, 0.4, 0.9, 0.9)]
    mask = build_multilabel_mask(insts, 100, 100)
    assert mask.shape == (4, 100, 100)
    assert (mask[0] & mask[3]).sum() > 0          # irisan ada di kedua kanal
    assert mask[1].sum() == 0 and mask[2].sum() == 0

def test_urutan_x_y_pada_citra_tidak_persegi():
    # Citra lebar (H=50, W=100). Poligon hanya di setengah kiri (x 0-0.5).
    m = rasterize_instance(square(0, 0.0, 0.0, 0.5, 1.0), 50, 100)
    assert m[:, 60:].sum() == 0                   # kolom kanan harus kosong
    assert m[:, :50].sum() > 0

def test_green_boost():
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    img[0, 0] = (0, 180, 0)      # hijau (BGR), harus berubah
    img[0, 1] = (40, 80, 140)    # cokelat (BGR), tidak boleh berubah
    img[0, 2] = (0, 0, 0)        # hitam, tidak boleh berubah
    out = green_boost(img)
    assert not np.array_equal(out[0, 0], img[0, 0])
    assert np.array_equal(out[0, 1], img[0, 1])
    assert np.array_equal(out[0, 2], img[0, 2])