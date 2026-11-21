"""Audit Defect Dataset: statistik validasi, cakupan instance, overlap, dan risiko kebocoran."""

import re
import imagehash
import numpy as np
import pandas as pd
from itertools import combinations
from PIL import Image
from tqdm import tqdm
from . import config as C
from .data import (foreground_mask, list_pairs, parse_label_file,
                   rasterize_instance, read_image_bgr)

_NUM_RE = re.compile(r"(\d+)")
CLASS_PAIRS = list(combinations(range(C.NUM_CLASSES), 2))

def file_number(stem):
    """Mengambil angka dari nama file, misalnya 'c_0076' -> 76. None jika tidak ada."""
    m = _NUM_RE.search(stem)
    return int(m.group(1)) if m else None

def touches_all_edges(points, tol=0.02):
    """True jika poligon menyentuh keempat tepi citra (indikasi kontur seluruh umbi)."""
    x, y = points[:, 0], points[:, 1]
    return bool(x.min() <= tol and x.max() >= 1 - tol
                and y.min() <= tol and y.max() >= 1 - tol)

def run_audit(giant_threshold=C.GIANT_COVERAGE):
    """Menjalankan audit seluruh split.

    Return tiga DataFrame:
        images_df    : satu baris per citra (ukuran, area kentang, jumlah instance,
                       overlap piksel per pasangan kelas versi 'all' dan 'nogiant')
        instances_df : satu baris per instance (luas, coverage, flag raksasa)
        issues_df    : semua masalah file dan baris label
    """
    image_rows, inst_rows, issue_rows = [], [], []

    for split in C.SPLITS:
        pairs, img_only, lbl_only = list_pairs(split)
        for s in img_only:
            issue_rows.append({"split": split, "file": s, "line_no": 0,
                               "issue": "citra_tanpa_label", "detail": ""})
        for s in lbl_only:
            issue_rows.append({"split": split, "file": s, "line_no": 0,
                               "issue": "label_tanpa_citra", "detail": ""})

        for img_path, lbl_path in tqdm(pairs, desc=f"audit {split}"):
            stem = img_path.stem
            try:
                img = read_image_bgr(img_path)
            except ValueError as exc:
                issue_rows.append({"split": split, "file": stem, "line_no": 0,
                                   "issue": "citra_rusak", "detail": str(exc)})
                continue

            h, w = img.shape[:2]
            fg_area = int(foreground_mask(img).sum())
            instances, issues = parse_label_file(lbl_path)
            for it in issues:
                issue_rows.append({"split": split, **vars(it)})

            mask_all = np.zeros((C.NUM_CLASSES, h, w), dtype=bool)
            mask_nogiant = np.zeros_like(mask_all)
            n_per_class = [0] * C.NUM_CLASSES

            for inst in instances:
                m = rasterize_instance(inst, h, w)
                area = int(m.sum())
                coverage = area / fg_area if fg_area > 0 else np.nan
                is_giant = bool(fg_area > 0 and coverage >= giant_threshold)

                inst_rows.append({
                    "split": split, "file": stem, "line_no": inst.line_no,
                    "class_id": inst.class_id,
                    "class_name": C.CLASS_NAMES[inst.class_id],
                    "area_px": area, "coverage": coverage,
                    "touches_all_edges": touches_all_edges(inst.points),
                    "is_giant": is_giant,
                })
                mask_all[inst.class_id] |= m
                if not is_giant:
                    mask_nogiant[inst.class_id] |= m
                n_per_class[inst.class_id] += 1

            row = {"split": split, "file": stem, "num": file_number(stem),
                   "h": h, "w": w, "fg_area": fg_area,
                   "n_instances": len(instances), "n_invalid_lines": len(issues)}
            for c in range(C.NUM_CLASSES):
                row[f"n_{c}"] = n_per_class[c]
            for i, j in CLASS_PAIRS:
                row[f"ov_{i}{j}_all"] = int((mask_all[i] & mask_all[j]).sum())
                row[f"ov_{i}{j}_nogiant"] = int((mask_nogiant[i] & mask_nogiant[j]).sum())
            image_rows.append(row)

    return (pd.DataFrame(image_rows), pd.DataFrame(inst_rows),
            pd.DataFrame(issue_rows, columns=["split", "file", "line_no", "issue", "detail"]))

# Analisis A
def compare_official(images_df, instances_df):
    """Analisis A: jumlah citra dan instance per kelas dibandingkan angka resmi."""
    rows = []
    for split in C.SPLITS:
        rows.append({"split": split, "item": "citra",
                     "observed": int((images_df["split"] == split).sum()),
                     "official": C.OFFICIAL_IMAGES[split]})
        for c, name in enumerate(C.CLASS_NAMES):
            sel = (instances_df["split"] == split) & (instances_df["class_id"] == c)
            rows.append({"split": split, "item": name,
                         "observed": int(sel.sum()),
                         "official": C.OFFICIAL_INSTANCES[split][c]})
    df = pd.DataFrame(rows)
    df["match"] = df["observed"] == df["official"]
    return df

# Analisis B dan C
def size_summary(images_df):
    """Analisis B: statistik ukuran dan rasio aspek citra per split."""
    df = images_df.assign(aspect=images_df["w"] / images_df["h"])
    return df.groupby("split")[["h", "w", "aspect"]].describe().T

def empty_label_summary(images_df):
    """Analisis C: jumlah citra tanpa instance valid per split."""
    return (images_df.assign(kosong=images_df["n_instances"] == 0)
            .groupby("split")["kosong"].sum().rename("citra_tanpa_instance"))

# Analisis D
def coverage_summary(instances_df, thresholds=(0.5, 0.9)):
    """Analisis D: distribusi coverage per kelas dan jumlah instance di atas ambang."""
    g = instances_df.groupby("class_name")
    out = g["coverage"].describe(percentiles=[0.5, 0.9, 0.99])
    for t in thresholds:
        out[f"n_cov>={t}"] = g["coverage"].apply(lambda s, t=t: int((s >= t).sum()))
    out["n_sentuh_4_tepi"] = g["touches_all_edges"].sum()
    return out

def giant_images_summary(instances_df):
    """Jumlah citra per split yang memiliki minimal satu instance raksasa, per kelas."""
    giant = instances_df[instances_df["is_giant"]]
    return (giant.groupby(["split", "class_name"])["file"].nunique()
            .rename("jumlah_citra").reset_index())

# Analisis E
def overlap_summary(images_df, version="all"):
    """Analisis E: overlap per pasangan kelas.

    version: 'all' (semua instance) atau 'nogiant' (tanpa instance raksasa).
    """
    rows = []
    for i, j in CLASS_PAIRS:
        col = f"ov_{i}{j}_{version}"
        rows.append({"pasangan": f"{C.CLASS_NAMES[i]} & {C.CLASS_NAMES[j]}",
                     "jumlah_citra": int((images_df[col] > 0).sum()),
                     "total_piksel": int(images_df[col].sum())})
    return pd.DataFrame(rows)

# Analisis F
def split_adjacency(images_df):
    """Analisis F: proporsi nomor file berurutan yang berada di split berbeda.

    Jika penempatan split acak per citra, nilai harapannya 1 - sum(p_split^2),
    dengan p_split adalah proporsi tiap split.
    """
    df = images_df.dropna(subset=["num"]).sort_values("num")
    nums = df["num"].to_numpy()
    splits = df["split"].to_numpy()
    consecutive = np.diff(nums) == 1
    diff_split = splits[1:] != splits[:-1]
    p = df["split"].value_counts(normalize=True)
    return {
        "pasangan_berurutan": int(consecutive.sum()),
        "proporsi_beda_split_teramati": float(diff_split[consecutive].mean()),
        "proporsi_harapan_jika_acak": float(1 - (p ** 2).sum()),
        "jumlah_celah_nomor": int((np.diff(nums) > 1).sum()),
        "nomor_duplikat": int(df["num"].duplicated().sum()),
    }

# Analisis G
def near_duplicates(images_df, max_distance=5):
    """Analisis G: pasangan citra lintas split dengan jarak pHash <= max_distance.

    Hanya menangkap citra yang hampir identik. Sisi berbeda dari umbi yang sama
    TIDAK akan terdeteksi dengan cara ini.
    """
    hashes = []
    for r in tqdm(images_df.itertuples(), total=len(images_df), desc="pHash"):
        path = C.IMAGE_DIR / r.split / f"{r.file}.png"
        with Image.open(path) as im:
            hashes.append((r.split, r.file, imagehash.phash(im)))

    rows = []
    for (sa, fa, ha), (sb, fb, hb) in combinations(hashes, 2):
        if sa == sb:
            continue
        d = ha - hb
        if d <= max_distance:
            rows.append({"split_a": sa, "file_a": fa, "split_b": sb,
                         "file_b": fb, "distance": int(d)})
    return pd.DataFrame(rows, columns=["split_a", "file_a", "split_b", "file_b", "distance"])

# Manifest
def build_manifest(images_df):
    """Menandai citra yang dipakai dalam penelitian.

    Citra tanpa instance dikeluarkan dari ketiga split
    (arahan dosen pembimbing, 16 November 2026).
    Return DataFrame: split, file, keep (bool), reason.
    """
    m = images_df[["split", "file", "n_instances"]].copy()
    m["keep"] = m["n_instances"] > 0
    m["reason"] = np.where(m["keep"], "", "tanpa_anotasi")
    return m.drop(columns="n_instances")

def split_counts(manifest):
    """Jumlah citra per split sebelum dan sesudah penghapusan."""
    g = manifest.groupby("split")["keep"]
    out = pd.DataFrame({"sebelum": g.size(), "dipakai": g.sum()})
    out["dihapus"] = out["sebelum"] - out["dipakai"]
    return out.reindex(list(C.SPLITS))

def kept_images(images_df, manifest):
    """images_df yang hanya berisi citra dengan keep == True."""
    m = images_df.merge(manifest[["split", "file", "keep"]], on=["split", "file"])
    return m[m["keep"]].drop(columns="keep")

# Audit 4
def images_per_class(images_df):
    """Audit 4: jumlah citra dan instance per kelas per split.

    Penting untuk kelas langka (Greening): sedikitnya citra di val
    membuat penyetelan threshold di val tidak stabil.
    """
    rows = []
    for split in C.SPLITS:
        s = images_df[images_df["split"] == split]
        for c, name in enumerate(C.CLASS_NAMES):
            rows.append({"split": split, "kelas": name,
                         "jumlah_citra": int((s[f"n_{c}"] > 0).sum()),
                         "jumlah_instance": int(s[f"n_{c}"].sum())})
    return pd.DataFrame(rows)

# Audit 3
def lesion_size_after_resize(instances_df, images_df,
                             sizes=C.RESIZE_CANDIDATES, tiny_px=C.TINY_LESION_PX):
    """Audit 3: perkiraan luas lesi (piksel) setelah resize letterbox.

    Letterbox: sisi terpanjang citra diubah menjadi `size`, rasio aspek dipertahankan.
    Luas berubah sebanding kuadrat faktor skala, jadi
        luas_baru = luas_asli * (size / sisi_terpanjang) ** 2
    Return tabel per resolusi dan kelas: median luas dan persentase lesi < tiny_px.
    """
    df = instances_df.merge(images_df[["split", "file", "h", "w"]], on=["split", "file"])
    long_side = df[["h", "w"]].max(axis=1)
    rows = []
    for size in sizes:
        area = df["area_px"] * (size / long_side) ** 2
        for name, grp in area.groupby(df["class_name"]):
            rows.append({"resolusi": size, "kelas": name,
                         "median_px": round(float(grp.median()), 1),
                         f"persen_<{tiny_px}px": round(100 * float((grp < tiny_px).mean()), 1),
                         f"jumlah_<{tiny_px}px": int((grp < tiny_px).sum())})
    return pd.DataFrame(rows)

# Duplikat terkonfirmasi
def apply_duplicates(manifest, review_df, label="foto_sama"):
    """Menandai citra duplikat terkonfirmasi lintas split di manifest.

    Aturan: dari setiap pasangan yang dinilai `label`, yang dikeluarkan adalah
    citra di split "lebih awal" (train lebih dulu, lalu val), sehingga test
    tetap utuh dan sebanding dengan baseline.

    Menambah kolom:
        is_duplicate : bool, citra dikeluarkan karena duplikat
        keep         : diperbarui (keep lama AND bukan duplikat)
        reason       : diisi "duplikat_lintas_split" bila dikeluarkan karena duplikat
    """
    order = {s: i for i, s in enumerate(C.SPLITS)}   # train=0, val=1, test=2
    confirmed = review_df[review_df["penilaian"] == label]

    drop = set()
    for r in confirmed.itertuples():
        a, b = (r.split_a, r.file_a), (r.split_b, r.file_b)
        drop.add(a if order[a[0]] < order[b[0]] else b)

    m = manifest.copy()
    m["is_duplicate"] = [(s, f) in drop for s, f in zip(m["split"], m["file"])]
    newly = m["is_duplicate"] & m["keep"]
    m.loc[newly, "reason"] = "duplikat_lintas_split"
    m["keep"] = m["keep"] & ~m["is_duplicate"]
    return m