"""
Fungsi data inti: membaca citra, parsing label YOLO-seg, dan membuat mask multilabel.
Fungsi di modul ini dipakai ulang di audit, training, evaluasi, dan prototipe,
sehingga semua tahap memakai konversi koordinat dan rasterisasi yang sama. 
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from . import config as C

_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)")

@dataclass
class Instance:
    """Satu poligon anotasi.

    class_id : int, 0-3 sesuai C.CLASS_NAMES
    points   : float array (N, 2), koordinat ternormalisasi (x, y) di [0, 1]
    line_no  : nomor baris pada file label (untuk penelusuran)
    """
    class_id: int
    points: np.ndarray
    line_no: int

@dataclass
class LabelIssue:
    """Catatan baris label yang gagal validasi."""
    file: str
    line_no: int
    issue: str
    detail: str

# File
def list_pairs(split):
    """Memasangkan citra dan label berdasarkan nama file.

    Return:
        pairs     : list (img_path, lbl_path) yang punya pasangan
        img_only  : list stem citra tanpa label
        lbl_only  : list stem label tanpa citra
    """
    images = {p.stem: p for p in sorted((C.IMAGE_DIR / split).glob("*.png"))}
    labels = {p.stem: p for p in sorted((C.LABEL_DIR / split).glob("*.txt"))}
    common = sorted(images.keys() & labels.keys())
    pairs = [(images[s], labels[s]) for s in common]
    img_only = sorted(images.keys() - labels.keys())
    lbl_only = sorted(labels.keys() - images.keys())
    return pairs, img_only, lbl_only

def read_image_bgr(path):
    """Membaca citra sebagai array uint8 (H, W, 3) berurutan BGR.

    Memakai np.fromfile + cv2.imdecode, bukan cv2.imread, karena cv2.imread
    di Windows gagal membaca path yang mengandung karakter non-ASCII.
    Raise ValueError jika citra tidak bisa didekode.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Citra tidak bisa didekode: {path}")
    return img

#  Label
def _polygon_area(points):
    """Luas poligon dengan rumus shoelace (satuan sama dengan koordinat input)."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def parse_label_line(line, line_no, file):
    """Memvalidasi dan mengurai satu baris label YOLO-seg.

    Format: class_id x1 y1 x2 y2 ... (minimal 3 titik).
    Return (Instance, None) jika valid, atau (None, LabelIssue) jika tidak.
    """
    parts = line.split()

    def fail(issue):
        return None, LabelIssue(file, line_no, issue, line.strip()[:120])

    if not _INT_RE.fullmatch(parts[0]):
        return fail("class_id_bukan_integer")
    class_id = int(parts[0])
    if not 0 <= class_id < C.NUM_CLASSES:
        return fail("class_id_di_luar_rentang")

    coords = parts[1:]
    if len(coords) == 4:
        return fail("format_bbox_bukan_poligon")
    if len(coords) % 2 != 0:
        return fail("jumlah_koordinat_ganjil")
    if len(coords) < 6:
        return fail("kurang_dari_3_titik")

    try:
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    except ValueError:
        return fail("koordinat_bukan_angka")
    if not np.isfinite(points).all():
        return fail("koordinat_tidak_hingga")
    if (points < -C.COORD_EPS).any() or (points > 1 + C.COORD_EPS).any():
        return fail("koordinat_di_luar_0_1")

    points = np.clip(points, 0.0, 1.0)  # rapikan sisa pembulatan kecil
    if _polygon_area(points) <= 1e-12:
        return fail("luas_poligon_nol")
    return Instance(class_id, points, line_no), None

def parse_label_file(path):
    """Mengurai satu file label.

    Return:
        instances : list Instance yang valid
        issues    : list LabelIssue (baris tidak valid TIDAK dibuang diam-diam)
    File kosong menghasilkan dua list kosong.
    """
    path = Path(path)
    instances, issues = [], []
    text = path.read_text(encoding="utf-8-sig")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        inst, issue = parse_label_line(line, line_no, path.stem)
        if inst is not None:
            instances.append(inst)
        else:
            issues.append(issue)
    return instances, issues

#  Mask
def polygon_to_pixels(points, width, height):
    """Mengubah koordinat ternormalisasi menjadi koordinat piksel int32 (N, 2).

    Konvensi: x dikali LEBAR, y dikali TINGGI (sama dengan konvensi YOLO).
    Hasil di-clip ke [0, width-1] dan [0, height-1].
    """
    px = np.empty_like(points)
    px[:, 0] = points[:, 0] * width    # x -> kolom
    px[:, 1] = points[:, 1] * height   # y -> baris
    px = np.rint(px).astype(np.int32)  # cv2.fillPoly wajib int32
    px[:, 0] = np.clip(px[:, 0], 0, width - 1)
    px[:, 1] = np.clip(px[:, 1], 0, height - 1)
    return px

def rasterize_instance(instance, height, width):
    """Menggambar satu poligon menjadi mask boolean (H, W)."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [polygon_to_pixels(instance.points, width, height)], 1)
    return canvas.astype(bool)

def build_multilabel_mask(instances, height, width):
    """Membuat mask multilabel uint8 (NUM_CLASSES, H, W).

    Setiap instance digabung (OR) ke kanal kelasnya sendiri, sehingga piksel
    yang berada di poligon dua kelas bernilai 1 di KEDUA kanal.
    Jangan diganti dengan mask satu kanal berisi indeks kelas, karena cara itu
    menimpa overlap dan mengubah formulasi menjadi multikelas.
    """
    mask = np.zeros((C.NUM_CLASSES, height, width), dtype=bool)
    for inst in instances:
        mask[inst.class_id] |= rasterize_instance(inst, height, width)
    return mask.astype(np.uint8)

def foreground_mask(img_bgr, threshold=C.FG_THRESHOLD):
    """Mask boolean (H, W) area kentang: piksel yang tidak hitam.

    Ambang dibuat kecil karena Rot bisa berwarna cokelat gelap sampai hitam.
    Periksa visual pada beberapa citra Rot sebelum mengubah ambang.
    """
    return img_bgr.max(axis=2) > threshold

#  Inferensi
def green_boost(img_bgr, params=C.GREEN_BOOST):
    """Menerapkan transformasi Green-Boost sesuai deskripsi dataset.

    HANYA untuk citra baru saat inferensi. Citra dataset sudah Green-Boosted,
    jadi fungsi ini tidak boleh diterapkan lagi pada citra dataset.
    Catatan: cara pembulatan oleh penyusun dataset tidak didokumentasikan.
    Di sini nilai dipotong (truncate) saat kembali ke uint8.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]   # view, bukan salinan
    sel = ((h >= params["h_min"]) & (h <= params["h_max"])
           & (s >= params["s_min"]) & (v >= params["v_min"]))
    # Dihitung dalam float lalu di-clip, supaya uint8 tidak overflow.
    s[sel] = np.clip(s[sel] * params["s_mul"], 0, 255)
    v[sel] = np.clip(v[sel] * params["v_mul"], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)