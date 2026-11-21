"""Model dan loss untuk segmentasi multilabel."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from . import config as C

def build_model(encoder_name="resnet34", encoder_weights="imagenet", num_classes=C.NUM_CLASSES):
    """U-Net dengan encoder ResNet34 pralatih ImageNet.

    Output: logit (B, num_classes, H, W), TANPA aktivasi.
    Sigmoid diterapkan di dalam loss (BCEWithLogits, lebih stabil secara numerik)
    dan saat evaluasi atau inferensi. Satu kanal per kelas, saling independen.
    """
    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                    in_channels=3, classes=num_classes, activation=None)

class BCEDiceLoss(nn.Module):
    """Gabungan Binary Cross-Entropy dan Soft Dice per kanal.

    BCE   : dihitung per piksel per kanal. Mendorong probabilitas setiap piksel benar.
    Dice  : dihitung per kanal di seluruh batch, lalu dirata-rata HANYA atas kanal
            yang memiliki piksel positif di batch tersebut.
            Tidak didominasi piksel latar, sehingga membantu kelas dengan piksel sedikit.
    loss = bce_weight * BCE + dice_weight * (1 - mean Dice kanal yang ada)

    Kenapa kanal kosong dikeluarkan dari Dice: jika sebuah kelas tidak ada di batch,
    Dice-nya hanya bisa dimaksimalkan dengan menekan seluruh prediksi kelas itu ke nol.
    Untuk kelas langka (Greening ada di sekitar 6% citra train), hal itu terjadi di
    sebagian besar batch dan menekan model agar tidak pernah memprediksi kelas tersebut.
    Penalti prediksi salah pada kanal kosong tetap diberikan oleh BCE.
    Perilaku ini sama dengan DiceLoss mode multilabel di segmentation_models_pytorch.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth   # mencegah pembagian nol

    def forward(self, logits, target):
        logits = logits.float()                 # hitung loss dalam float32 walau memakai AMP
        target = target.float()
        bce = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        dims = (0, 2, 3)                        # jumlahkan batch dan piksel, pisahkan kanal
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        present = target.sum(dims) > 0          # kanal yang punya piksel positif di batch
        if present.any():
            dice_loss = 1 - dice[present].mean()
        else:
            dice_loss = logits.sum() * 0.0      # batch tanpa cacat: hanya BCE yang bekerja

        return self.bce_weight * bce + self.dice_weight * dice_loss