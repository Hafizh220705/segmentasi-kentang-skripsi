"""Konstanta proyek: path, kelas, angka resmi dataset, dan parameter preprocessing."""
from pathlib import Path 

# Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "defect_dataset"
IMAGE_DIR = RAW_DIR / "images"
LABEL_DIR = RAW_DIR / "labels"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

# Class / Kelas / Label
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("Scab", "Rot", "Greening", "Mechanical Damage") # indeks = class_id
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLORS_RGB = (
    (0, 255, 0),    # Scab
    (0, 255, 255),  # Rot
    (255, 165, 0),  # Greening
    (255, 0, 165),  # Mechanical Damage 
)

# Angka resmi (Tasnia, 2026)
OFFICIAL_IMAGES = {"train": 1281, "val": 176, "test": 267}
# Urutan sesuai CLASS_NAMES: Scab, Rot, Greening, Mechanical Damage.
# Perhatikan: deskripsi dataset menulis Mechanical Damage sebelum Greening.
OFFICIAL_INSTANCES = {
    "train": (3188, 903, 79, 562),
    "val": (548, 175, 17, 108),
    "test": (654, 190, 29, 123),
}

# Preprocessing
# Parameter Green-Boost dari deskripsi dataset (hue OpenCV 0-179).
GREEN_BOOST = {"h_min": 35, "h_max": 85, "s_min": 40, "v_min": 40,
               "s_mul": 1.5, "v_mul": 1.3}
FG_THRESHOLD = 5  # Piksel dengan nilai kanal maks > ambang dianggap kentang
COORD_EPS = 1e-6  # Toleransi korodinat sedikit di luar [0, 1]

# Audit
GIANT_COVERAGE = 0.9 # Ambang audit, BUKAN  keputusan final penanganan data
SEED = 42

# Data olahan
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

# Audit resolusi
# Kandidat ukuran sisi terpanjang setelah resize (letterbox).
RESIZE_CANDIDATES = (512, 768, 1024)
# Lesi dengan luas di bawah ini (sekitar 4 x 4 piksel) dianggap hampir hilang.
# Heuristik audit, bukan aturan pemrosesan data.
TINY_LESION_PX = 16

# Resolusi input model (sisi terpanjang setelah letterbox). Dasar: audit ukuran lesi.
IMG_SIZE = 512

# Cache hasil preprocessing
PROC_IMG_DIR = PROCESSED_DIR / f"images_{IMG_SIZE}"
PROC_MASK_DIR = PROCESSED_DIR / f"masks_{IMG_SIZE}"
PROC_META_PATH = PROCESSED_DIR / f"meta_{IMG_SIZE}.csv"

# Normalisasi ImageNet (encoder ResNet34 dilatih dengan statistik ini)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Saklar eksperimen. Default False = anotasi dipakai apa adanya.
# True = instance Scab dengan coverage >= GIANT_COVERAGE tidak dipakai sebagai target.
IGNORE_GIANT_SCAB = False

# DataLoader. num_workers=0 paling aman untuk Jupyter di Windows.
BATCH_SIZE = 4
NUM_WORKERS = 0