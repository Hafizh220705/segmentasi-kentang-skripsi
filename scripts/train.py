"""Training U-Net multilabel.

Pemakaian (dari root proyek, environment .venv aktif):
    python scripts/train.py --config configs/baseline.yaml --smoke     # uji memori + alur
    python scripts/train.py --config configs/baseline.yaml --overfit 8 # uji model bisa belajar
    python scripts/train.py --config configs/baseline.yaml             # training penuh

Hasil disimpan di outputs/runs/<run_name>/:
    config.yaml     salinan konfigurasi yang dipakai
    env.json        versi library, GPU, jumlah data
    log.csv         loss dan metrik validasi per epoch
    best.pt         checkpoint dengan val mIoU tertinggi
    last.pt         checkpoint epoch terakhir (untuk melanjutkan atau audit)

Test set TIDAK disentuh di script ini. Evaluasi test ada di Modul 4.
"""

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from src import config as C
from src.dataset import PotatoDefectDataset, build_loader, seed_everything
from src.engine import Timer, evaluate, train_one_epoch
from src.metrics import metrics_to_row
from src.model import BCEDiceLoss, build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--run-name", default=None, help="menimpa run_name di config")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 10 batch train, 5 batch val: uji memori dan alur")
    p.add_argument("--overfit", type=int, default=0,
                   help="latih dan evaluasi pada N citra train tanpa augmentasi")
    p.add_argument("--lr", type=float, default=None, help="menimpa lr di config")
    p.add_argument("--overfit-class", type=int, default=None,
                   help="pilih N citra train dengan piksel kelas ini terbanyak (0-3)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or cfg["run_name"]
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
        run_name += f"_lr{args.lr:g}"
    if args.smoke:
        run_name += "_smoke"
        cfg["train"]["max_epochs"] = 1
    if args.overfit:
        run_name += f"_overfit{args.overfit}"
        if args.overfit_class is not None:                       # BARU
            run_name += f"_kelas{args.overfit_class}"            # BARU
        cfg["train"]["max_epochs"] = 100
        cfg["train"]["early_stopping_patience"] = 1000   # jangan berhenti dini
        cfg["scheduler"]["patience"] = 1000              # jangan turunkan lr saat uji menghafal

    run_dir = C.PROJECT_ROOT / "outputs" / "runs" / run_name
    if run_dir.exists() and (run_dir / "best.pt").exists() and not (args.smoke or args.overfit):
        raise FileExistsError(f"{run_dir} sudah berisi hasil. Pakai --run-name lain.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- Reproducibility
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("PERINGATAN: GPU tidak terdeteksi, training berjalan di CPU (sangat lambat).")

    # -------------------------------------------------------------- Data
    d = cfg["data"]
    if args.overfit:
        if args.overfit_class is not None:                       # BARU: pilih citra kaya kelas
            meta = pd.read_csv(C.PROC_META_PATH)
            top = meta[meta["split"] == "train"].nlargest(
                args.overfit, f"px_lb_{args.overfit_class}")
            files = top["file"].tolist()
            print("Citra overfit:", files)
        else:
            files = PotatoDefectDataset("train", train=False).files[:args.overfit]
        train_ds = PotatoDefectDataset("train", train=False, files=files,
                                       ignore_giant_scab=d["ignore_giant_scab"])
        val_ds = train_ds                                  # sengaja sama: uji menghafal
    else:
        train_ds = PotatoDefectDataset("train", train=True,
                                       ignore_giant_scab=d["ignore_giant_scab"])
        val_ds = PotatoDefectDataset("val", train=False,
                                     ignore_giant_scab=d["ignore_giant_scab"])
    train_loader = build_loader(train_ds, shuffle=True, batch_size=d["batch_size"],
                                num_workers=d["num_workers"], seed=cfg["seed"])
    val_loader = build_loader(val_ds, shuffle=False, batch_size=d["batch_size"],
                              num_workers=d["num_workers"], seed=cfg["seed"])

    # -------------------------------------------------------------- Model dan optimasi
    model = build_model(cfg["model"]["encoder_name"], cfg["model"]["encoder_weights"]).to(device)
    loss_fn = BCEDiceLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                                  weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"])
    t = cfg["train"]
    use_amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------------------------------------------------- Catat lingkungan
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = {
        "waktu_mulai": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                          if device.type == "cuda" else None),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "img_size": C.IMG_SIZE, "mode": "smoke" if args.smoke else
                                       (f"overfit{args.overfit}" if args.overfit else "full"),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))

    # -------------------------------------------------------------- Loop
    max_train = 10 if args.smoke else None
    max_val = 5 if args.smoke else None
    best, bad_epochs, log = -np.inf, 0, []

    for epoch in range(1, t["max_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Timer() as tm:
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                         device, use_amp, t["accum_steps"], max_train)
            val_loss, res = evaluate(model, val_loader, loss_fn, device, use_amp,
                                     t["threshold"], max_val)

        val_miou = res["miou"]
        scheduler.step(val_miou)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": train_loss, "val_loss": val_loss, "detik": round(tm.seconds, 1),
               "gpu_peak_gb": (round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                               if device.type == "cuda" else None)}
        row.update(metrics_to_row(res, prefix="val_"))
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)

        ckpt = {"epoch": epoch, "model": model.state_dict(), "config": cfg,
                "val_miou": val_miou, "val_iou": res["iou"].tolist()}
        torch.save(ckpt, run_dir / "last.pt")
        improved = val_miou > best
        if improved:
            best, bad_epochs = val_miou, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad_epochs += 1

        iou_txt = " ".join(f"{n[:4]}={v:.3f}" for n, v in zip(C.CLASS_NAMES, res["iou"]))
        print(f"[{epoch:02d}] loss {train_loss:.4f}/{val_loss:.4f} | val mIoU {val_miou:.4f} "
              f"({iou_txt}) | {tm.seconds:.0f}s | GPU {row['gpu_peak_gb']} GB"
              f"{' *' if improved else ''}")

        if bad_epochs >= t["early_stopping_patience"]:
            print(f"Early stopping: val mIoU tidak membaik {bad_epochs} epoch.")
            break

    print(f"Selesai. Val mIoU terbaik = {best:.4f}. Hasil di {run_dir}")

if __name__ == "__main__":
    main()