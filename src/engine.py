"""Satu epoch training dan evaluasi. Dipakai oleh scripts/train.py."""

import time
import torch
from tqdm import tqdm
from .metrics import MultilabelConfusion

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    amp=True, accum_steps=1, max_batches=None):
    """Melatih model satu epoch. Return rata-rata loss.

    accum_steps > 1: gradient accumulation. Gradien dari beberapa batch kecil
    dijumlahkan sebelum optimizer.step(), sehingga efeknya setara batch besar
    tanpa memori GPU tambahan.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))

    for i, batch in enumerate(tqdm(loader, total=n_batches, desc="train", leave=False)):
        if i >= n_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        loss = loss_fn(logits, y)                       # loss dihitung float32

        scaler.scale(loss / accum_steps).backward()
        last = (i + 1) == n_batches
        if (i + 1) % accum_steps == 0 or last:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp=True, thresholds=0.5, max_batches=None):
    """Evaluasi tanpa augmentasi. Return (rata-rata loss, hasil MultilabelConfusion.compute())."""
    model.eval()
    cm = MultilabelConfusion(thresholds=thresholds)
    total, n = 0.0, 0
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))

    for i, batch in enumerate(tqdm(loader, total=n_batches, desc="eval", leave=False)):
        if i >= n_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        total += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        cm.update(logits, y)
    return total / max(n, 1), cm.compute()

class Timer:
    """Pengukur waktu sederhana untuk log per epoch."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0