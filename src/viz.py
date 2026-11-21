"""Visualisasi audit: overlay mask, perbandingan berdampingan, dan mask area kentang.

Setiap grid diberi judul tujuan dan legenda warna, supaya gambar bisa dibaca
tanpa melihat kode atau keterangan tambahan.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from . import config as C
from .data import build_multilabel_mask, foreground_mask, parse_label_file, read_image_bgr

# Data
def load_sample(split, stem):
    """Memuat citra BGR dan mask multilabel (NUM_CLASSES, H, W) untuk satu file."""
    img = read_image_bgr(C.IMAGE_DIR / split / f"{stem}.png")
    instances, _ = parse_label_file(C.LABEL_DIR / split / f"{stem}.txt")
    h, w = img.shape[:2]
    return img, build_multilabel_mask(instances, h, w)

def to_rgb(img_bgr):
    """BGR (OpenCV) -> RGB (matplotlib)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Overlay
def overlay(img_bgr, mask, alpha=0.35):
    """Menumpuk mask berwarna dan konturnya di atas citra. Return array RGB uint8.

    Kontur digambar supaya area overlap tetap terlihat, karena warna yang
    bertumpuk sulit dibedakan jika hanya mengandalkan isi.
    """
    out = to_rgb(img_bgr).astype(np.float32)
    for c in range(C.NUM_CLASSES):
        sel = mask[c].astype(bool)
        if sel.any():
            color = np.array(C.CLASS_COLORS_RGB[c], dtype=np.float32)
            out[sel] = (1 - alpha) * out[sel] + alpha * color
    out = out.astype(np.uint8)

    thickness = max(2, min(out.shape[:2]) // 300)
    for c in range(C.NUM_CLASSES):
        contours, _ = cv2.findContours(mask[c].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, C.CLASS_COLORS_RGB[c], thickness)
    return out

# Tata letak
def _legend_handles(class_ids=None):
    ids = range(C.NUM_CLASSES) if class_ids is None else class_ids
    return [Patch(facecolor=np.array(C.CLASS_COLORS_RGB[c]) / 255, edgecolor="black",
                  label=f"{c} - {C.CLASS_NAMES[c]}") for c in ids]

def _finish(fig, title, legend_ids="all"):
    """Menambahkan judul dan legenda di atas grid dengan ruang tetap (dalam inci).

    legend_ids: "all" untuk semua kelas, list id kelas, atau None tanpa legenda.
    """
    height = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.12 / height)
    if legend_ids is not None:
        ids = None if legend_ids == "all" else legend_ids
        fig.legend(handles=_legend_handles(ids), loc="center",
                   bbox_to_anchor=(0.5, 1 - 0.5 / height),
                   ncol=C.NUM_CLASSES, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))  # sisakan 0.85 inci di atas
    return fig

def _present(mask):
    names = [C.CLASS_NAMES[c] for c in range(C.NUM_CLASSES) if mask[c].any()]
    return ", ".join(names) or "tanpa label"

# Grid
def show_grid(items, title="Audit visual", ncols=3, size=5):
    """Grid overlay semua kelas.

    items: list tuple (split, stem, keterangan_tambahan)
    """
    if not items:
        print("Tidak ada citra untuk ditampilkan.")
        return None
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows),
                             squeeze=False)
    for ax, (split, stem, extra) in zip(axes.flat, items):
        img, mask = load_sample(split, stem)
        ax.imshow(overlay(img, mask))
        extra_line = f"\n{extra}" if extra else ""
        ax.set_title(f"{split}/{stem}{extra_line}\nKelas: {_present(mask)}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    return _finish(fig, title)

def show_side_by_side(items, title, class_id=None, size=4):
    """Per baris: citra asli (kiri) dan overlay mask (kanan).

    class_id: jika diisi (0-3), hanya kanal kelas itu yang ditampilkan.
    """
    if not items:
        print("Tidak ada citra untuk ditampilkan.")
        return None
    label = C.CLASS_NAMES[class_id] if class_id is not None else "semua kelas"
    fig, axes = plt.subplots(len(items), 2, figsize=(size * 2, size * len(items)),
                             squeeze=False)
    for (ax_raw, ax_ov), (split, stem, extra) in zip(axes, items):
        img, mask = load_sample(split, stem)
        if class_id is not None:
            only = np.zeros_like(mask)
            only[class_id] = mask[class_id]
            mask = only
        extra_line = f" | {extra}" if extra else ""
        ax_raw.imshow(to_rgb(img))
        ax_raw.set_title(f"CITRA ASLI\n{split}/{stem}{extra_line}", fontsize=9)
        ax_ov.imshow(overlay(img, mask))
        ax_ov.set_title(f"MASK {label.upper()}\n{split}/{stem}", fontsize=9)
        ax_raw.axis("off")
        ax_ov.axis("off")
    ids = [class_id] if class_id is not None else "all"
    return _finish(fig, title, legend_ids=ids)

def show_pairs(pairs_df, title, size=4):
    """Per baris: dua citra hampir identik lintas split, tanpa mask."""
    if len(pairs_df) == 0:
        print("Tidak ada pasangan untuk ditampilkan.")
        return None
    fig, axes = plt.subplots(len(pairs_df), 2, figsize=(size * 2, size * len(pairs_df)),
                             squeeze=False)
    for (ax_a, ax_b), r in zip(axes, pairs_df.itertuples()):
        for ax, tag, split, stem in ((ax_a, "A", r.split_a, r.file_a),
                                     (ax_b, "B", r.split_b, r.file_b)):
            ax.imshow(to_rgb(read_image_bgr(C.IMAGE_DIR / split / f"{stem}.png")))
            ax.set_title(f"CITRA {tag}: {split.upper()}/{stem}\njarak pHash = {r.distance}",
                         fontsize=9)
            ax.axis("off")
    return _finish(fig, title, legend_ids=None)

def show_foreground(items, title, size=4):
    """Per baris: citra asli (kiri) dan mask area kentang (kanan, putih = kentang).

    Dipakai untuk mengecek FG_THRESHOLD: area busuk yang gelap seharusnya
    tetap putih, bukan berlubang hitam.
    """
    if not items:
        print("Tidak ada citra untuk ditampilkan.")
        return None
    fig, axes = plt.subplots(len(items), 2, figsize=(size * 2, size * len(items)),
                             squeeze=False)
    for (ax_raw, ax_fg), (split, stem, extra) in zip(axes, items):
        img = read_image_bgr(C.IMAGE_DIR / split / f"{stem}.png")
        fg = foreground_mask(img)
        extra_line = f" | {extra}" if extra else ""
        ax_raw.imshow(to_rgb(img))
        ax_raw.set_title(f"CITRA ASLI\n{split}/{stem}{extra_line}", fontsize=9)
        ax_fg.imshow(fg, cmap="gray")
        ax_fg.set_title(f"AREA KENTANG (putih)\nambang = {C.FG_THRESHOLD}", fontsize=9)
        ax_raw.axis("off")
        ax_fg.axis("off")
    return _finish(fig, title, legend_ids=None)

def neighbor_items(images_df, stem, k=3):
    """Item untuk show_grid: citra bernomor stem-k sampai stem+k, lintas split."""
    num = images_df.loc[images_df["file"] == stem, "num"].iloc[0]
    sub = images_df[(images_df["num"] >= num - k) & (images_df["num"] <= num + k)]
    return [(r.split, r.file, "<< ACUAN" if r.file == stem else "")
            for r in sub.sort_values("num").itertuples()]