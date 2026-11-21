"""Uji Modul 3: metrik, loss, dan bentuk keluaran model. Jalankan: python -m pytest"""

import math
import numpy as np
import torch
from src.metrics import MultilabelConfusion
from src.model import BCEDiceLoss, build_model

BIG = 20.0   # logit besar: sigmoid(20) ~ 1, sigmoid(-20) ~ 0

def logits_from(mask):
    """Mask 0/1 -> logit yang pasti menghasilkan prediksi sama persis."""
    return torch.where(mask > 0.5, torch.tensor(BIG), torch.tensor(-BIG))

def test_metrik_prediksi_sempurna():
    y = torch.zeros(2, 4, 8, 8)
    y[:, 0, :4, :4] = 1
    y[:, 3, 2:6, 2:6] = 1            # overlap dengan kanal 0
    cm = MultilabelConfusion()
    cm.update(logits_from(y), y)
    r = cm.compute()
    assert r["iou"][0] == 1.0 and r["iou"][3] == 1.0
    assert math.isnan(r["iou"][1])   # kanal tanpa GT dan tanpa prediksi -> NaN, bukan 1

def test_metrik_angka_diketahui():
    # GT: 4 piksel positif. Prediksi: 2 benar + 2 salah. TP=2, FP=2, FN=2
    y = torch.zeros(1, 4, 4, 4)
    y[0, 1, 0, :4] = 1
    p = torch.zeros(1, 4, 4, 4)
    p[0, 1, 0, :2] = 1
    p[0, 1, 1, :2] = 1
    cm = MultilabelConfusion()
    cm.update(logits_from(p), y)
    r = cm.compute()
    assert np.isclose(r["iou"][1], 2 / 6)
    assert np.isclose(r["dice"][1], 4 / 8)
    assert np.isclose(r["precision"][1], 0.5) and np.isclose(r["recall"][1], 0.5)

def test_metrik_akumulasi_tingkat_dataset():
    """Dua batch dijumlah dulu, baru dihitung (bukan rata-rata IoU per batch)."""
    y1 = torch.zeros(1, 4, 4, 4); y1[0, 0, :2, :2] = 1          # 4 piksel, tebakan benar
    y2 = torch.zeros(1, 4, 4, 4); y2[0, 0, 0, 0] = 1            # 1 piksel, terlewat
    cm = MultilabelConfusion()
    cm.update(logits_from(y1), y1)
    cm.update(logits_from(torch.zeros_like(y2)), y2)
    assert np.isclose(cm.compute()["iou"][0], 4 / 5)            # TP=4, FN=1

def test_threshold_per_kanal():
    y = torch.zeros(1, 4, 2, 2); y[0, :, 0, 0] = 1
    logits = torch.full((1, 4, 2, 2), -BIG)
    logits[0, :, 0, 0] = 0.0                                    # prob 0.5 tepat
    cm = MultilabelConfusion(thresholds=[0.5, 0.5, 0.6, 0.5])
    cm.update(logits, y)
    tp = cm.compute()["tp"]
    assert tp.tolist() == [1, 1, 0, 1]                          # kanal 2 butuh >= 0.6

def test_loss_sempurna_lebih_kecil_dari_acak():
    y = torch.zeros(2, 4, 16, 16); y[:, 2, 4:12, 4:12] = 1
    loss = BCEDiceLoss()
    perfect = loss(logits_from(y), y).item()
    random_ = loss(torch.randn_like(y), y).item()
    assert perfect < 0.05 < random_

def test_bentuk_keluaran_model():
    model = build_model(encoder_weights=None).eval()            # tanpa unduh bobot
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)                          # 4 kanal, ukuran sama

def test_dice_mengabaikan_kanal_kosong():
    """Kanal tanpa piksel positif di batch tidak ikut Dice, tetapi tetap kena BCE."""
    y = torch.zeros(2, 4, 16, 16); y[:, 0, 4:12, 4:12] = 1     # hanya kanal 0 yang ada
    logits = logits_from(y)
    dice_only = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
    base = dice_only(logits, y).item()
    logits_fp = logits.clone(); logits_fp[:, 2] = BIG          # kanal 2 (kosong) prediksi positif
    assert abs(dice_only(logits_fp, y).item() - base) < 1e-6   # Dice tidak berubah
    bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
    assert bce_only(logits_fp, y).item() > bce_only(logits, y).item()   # BCE menghukum

def test_loss_batch_tanpa_cacat_tidak_error():
    y = torch.zeros(1, 4, 8, 8)
    loss = BCEDiceLoss()(torch.randn(1, 4, 8, 8, requires_grad=True), y)
    loss.backward()
    assert torch.isfinite(loss)