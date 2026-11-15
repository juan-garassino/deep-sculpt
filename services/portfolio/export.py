"""Export trainer checkpoints to TorchScript generators for the portfolio serve.

    PYTHONPATH=. .venv/bin/python services/portfolio/export.py \
        --run <dir with config.json + checkpoints/> --epochs 17_best,75 --out services/portfolio/weights

Writes generator_e<N>.ts (raw generator, not EMA: EMA samples look blurry) and
meta.json (noise_dim, num_classes, palette, per-epoch training metrics). The serve
then needs only torch + numpy, not the deepsculpt package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepsculpt.core.data.transforms.palette import CLASS_PALETTE
from deepsculpt.core.latent.loader import load_generator


def _epoch_metrics(run: Path) -> dict[int, dict]:
    path = run / "logs" / "epoch_metrics.jsonl"
    rows: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row.get("epoch", -1))] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--epochs", default="75")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="", help="run label, e.g. gan-shodhan-color-001/gan_skip_20260712_053147")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = _epoch_metrics(args.run)
    meta: dict = {"run": args.name or args.run.name, "epochs": {}, "palette": {str(k): v for k, v in CLASS_PALETTE.items()}}

    for tag in args.epochs.split(","):
        epoch = int(tag.split("_")[0])
        lg = load_generator(args.run / "checkpoints" / f"checkpoint_epoch_{tag}.pth", "cpu", prefer_ema=False)
        gen = lg.generator.eval()
        z = torch.randn(2, lg.noise_dim, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            traced = torch.jit.trace(gen, z[:1], check_trace=False)
            ref = gen(z).argmax(1)
            got = traced(z).argmax(1)
        agree = float((ref == got).float().mean())
        if agree < 0.999:
            raise SystemExit(f"epoch {epoch}: traced generator disagrees with eager ({agree:.4f})")
        traced.save(str(args.out / f"generator_e{epoch}.ts"))
        meta["noise_dim"] = lg.noise_dim
        meta["void_dim"] = lg.void_dim
        meta["num_classes"] = int(lg.config.get("num_classes", 13))
        meta["epochs"][str(epoch)] = {"checkpoint": f"checkpoint_epoch_{tag}.pth", "agreement": agree,
                                      "training": metrics.get(epoch, {})}
        print(f"epoch {epoch}: traced ok (agreement {agree:.4f})")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
