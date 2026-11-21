"""Metrik segmentasi multilabel tingkat dataset.

Setiap kanal (kelas) dievaluasi sebagai segmentasi biner yang berdiri sendiri.
TP, FP, FN, TN dijumlahkan dari SELURUH piksel seluruh citra lebih dulu,
baru metrik dihitung (agregasi tingkat dataset). Cara ini tidak dipengaruhi
citra yang tidak memuat kelas tersebut.

Kanal tidak digabung lewat argmax, karena satu piksel boleh positif di dua kanal.
"""

import numpy as np
import torch

from . import config as C

class MultilabelConfusion:
    """Akumulator TP/FP/FN/TN per kelas.

    Pemakaian:
        cm = MultilabelConfusion()
        for batch: cm.update(logits, target)
        hasil = cm.compute()
    """

    def __init__(self, num_classes=C.NUM_CLASSES, thresholds=0.5):
        self.num_classes = num_classes
        t = np.broadcast_to(np.asarray(thresholds, dtype=np.float32), (num_classes,))
        self.thresholds = torch.tensor(t.copy())          # satu threshold per kanal
        self.reset()

    def reset(self):
        # int64: jumlah piksel seluruh dataset bisa melebihi batas int32
        self.tp = torch.zeros(self.num_classes, dtype=torch.int64)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)
        self.tn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, logits, target):
        """logits, target: tensor (B, C, H, W). target bernilai 0/1."""
        prob = torch.sigmoid(logits.float())
        thr = self.thresholds.to(prob.device).view(1, -1, 1, 1)
        pred = prob >= thr
        gt = target > 0.5
        dims = (0, 2, 3)                                   # jumlahkan semua kecuali kanal
        self.tp += (pred & gt).sum(dims).cpu()
        self.fp += (pred & ~gt).sum(dims).cpu()
        self.fn += (~pred & gt).sum(dims).cpu()
        self.tn += (~pred & ~gt).sum(dims).cpu()

    def compute(self):
        """Return dict berisi array per kelas dan rata-ratanya.

        Penyebut 0 menghasilkan NaN (bukan 0 atau 1), supaya kasus tanpa data
        tidak tampil sebagai nilai sempurna atau buruk palsu.
        """
        tp, fp, fn = (x.double().numpy() for x in (self.tp, self.fp, self.fn))

        def div(a, b):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(b > 0, a / b, np.nan)

        iou = div(tp, tp + fp + fn)
        dice = div(2 * tp, 2 * tp + fp + fn)
        precision = div(tp, tp + fp)
        recall = div(tp, tp + fn)
        return {
            "iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "miou": float(np.nanmean(iou)), "mdice": float(np.nanmean(dice)),
            "tp": self.tp.numpy().copy(), "fp": self.fp.numpy().copy(),
            "fn": self.fn.numpy().copy(), "tn": self.tn.numpy().copy(),
        }

def metrics_to_row(result, prefix=""):
    """Meratakan hasil compute() menjadi dict satu baris (untuk log CSV)."""
    row = {f"{prefix}miou": result["miou"], f"{prefix}mdice": result["mdice"]}
    for c, name in enumerate(C.CLASS_NAMES):
        key = name.lower().replace(" ", "_")
        for m in ("iou", "dice", "precision", "recall"):
            row[f"{prefix}{m}_{key}"] = float(result[m][c])
    return row