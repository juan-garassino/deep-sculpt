"""deep-sculpt portfolio serve: sample a 64^3 sculpture from the GAN on CPU.

Contract (the ML gateway + the art² Work section volume viewer):
  GET  /health                         liveness + which epochs are loaded
  GET  /api/metrics                    training metrics of the default epoch
  GET  /api/volume?seed=&epoch=        one sculpture: {dims, volumes{sculpture: b64 u8}, palette, ...}
  POST /predict {seed?, epoch?}        same payload (the gateway's :predict route)

Volumes go out as (D,H,W) C-order uint8 class ids (0 empty, 1..12 elements) with D
vertical and d=0 at the TOP — deep-sculpt's own arrays are (X,Y,Z) with Z up.

The model comes from the MLflow registry, not the image: at boot the serve resolves
`<DEEPSCULPT_MODEL>@champion` (+ @challenger) on MLFLOW_TRACKING_URI and downloads each
version's TorchScript file (the tracker's artifact proxy; files over 30 MiB, or a gs://
`source`, straight from GCS with the runtime SA). Promote a version in MLflow and restart the
serve — no rebuild. Without MLFLOW_TRACKING_URI it reads a local weights dir (export.py output)
for offline dev.

Boot never blocks on the tracker: registry.py (the 027-ml-workspace serves' module, copied
verbatim) loads in a background thread with retries (exponential backoff + jitter, ~3 min per
call; a failed load is reported on /health and retried), and its gate answers /health with 503
{"status": "loading"} and holds other requests until the generators are loaded and warmed.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACKER = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")  # e.g. https://garassino-mlflow-…run.app
MODEL = os.environ.get("DEEPSCULPT_MODEL", "deepsculpt-gan")  # registered model name
ALIASES = os.environ.get("DEEPSCULPT_ALIASES", "champion,challenger").split(",")  # first = default
WEIGHTS = Path(os.environ.get("DEEPSCULPT_WEIGHTS", Path(__file__).parent / "weights"))  # e.g. /tmp/weights
MAX_SEED = 2**31 - 1
SOURCES: dict[int, dict] = {}  # epoch -> {alias, version, file}; champion's epoch first
META: dict = {}  # the generators' meta.json (noise_dim, palette, per-epoch training stats)
EPOCHS: list[int] = []
DEFAULT: dict = {"epoch": None}  # champion's epoch (first alias), or the first local one


def _fetch(src: str, fname: str, dest: Path) -> None:
    """One file of a model version: proxy, backing bucket (large) or a gcs-direct source."""
    if src.startswith("gs://") and not src.startswith(registry._destination()):
        loc = f"{src.rstrip('/')}/{fname}"
        registry.download(loc, dest, registry.gcs_size(loc))
        return
    base = registry.proxy_path(src)
    size = dict(registry.list_files(base)).get(fname)
    registry.download(registry.where(f"{base}/{fname}", size), dest, size, fallback=f"{base}/{fname}")


def _from_registry() -> dict:
    """Download every alias's generator into WEIGHTS; return {epoch: {alias, version, file}}."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    loaded: dict[int, dict] = {}
    for alias in ALIASES:
        mv = registry.resolve(MODEL, alias)
        epoch = int({t["key"]: t["value"] for t in mv.get("tags", [])}["epoch"])
        fname = f"generator_e{epoch}.ts"
        _fetch(mv["source"], fname, WEIGHTS / fname)
        if not (WEIGHTS / "meta.json").exists():
            _fetch(mv["source"], "meta.json", WEIGHTS / "meta.json")
        loaded.setdefault(epoch, {"alias": alias, "version": mv["version"], "file": fname})
        logger.info("%s@%s -> v%s (epoch %s)", MODEL, alias, mv["version"], epoch)
    return loaded


def _load() -> None:
    """Registry (or local weights) -> SOURCES/META, then warm TorchScript — off the request path."""
    if TRACKER:
        sources = _from_registry()
    else:
        sources = {int(e): {"alias": "local", "version": None, "file": f"generator_e{e}.ts"}
                   for e in sorted(json.loads((WEIGHTS / "meta.json").read_text())["epochs"], key=int, reverse=True)}
    META.update(json.loads((WEIGHTS / "meta.json").read_text()))
    SOURCES.update(sources)
    EPOCHS[:] = sorted(sources)
    DEFAULT["epoch"] = next(iter(sources))
    for _ in range(3):  # TorchScript optimizes over its first runs; pay that at boot, not on a visitor
        sculpt(0, DEFAULT["epoch"])


torch.set_num_threads(max(1, os.cpu_count() or 1))
app = FastAPI(title="deepsculpt-serve")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
registry.install(app)  # /health 503 "loading" + requests held until the generators are loaded


class SculptRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    epoch: int | None = None


@lru_cache(maxsize=None)
def generator(epoch: int) -> torch.jit.ScriptModule:
    if epoch not in EPOCHS:
        raise KeyError(epoch)
    gen = torch.jit.load(str(WEIGHTS / SOURCES[epoch]["file"]), map_location="cpu")
    gen.eval()
    logger.info("loaded generator epoch %s", epoch)
    return gen


def sculpt(seed: int | None, epoch: int | None) -> dict:
    epoch = DEFAULT["epoch"] if epoch is None else epoch
    seed = secrets.randbelow(MAX_SEED) if seed is None else seed
    try:
        gen = generator(epoch)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"epoch must be one of {EPOCHS}")
    t0 = time.perf_counter()
    z = torch.randn(1, META["noise_dim"], generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        vol = gen(z).argmax(1)[0].to(torch.uint8).numpy()  # (X, Y, Z), Z up
    ms = (time.perf_counter() - t0) * 1000
    dhw = np.ascontiguousarray(np.transpose(vol, (2, 0, 1))[::-1])  # (Z top→bottom, X, Y)
    return {
        "dims": list(dhw.shape),
        "volumes": {"sculpture": base64.b64encode(dhw.tobytes()).decode("ascii")},
        "palette": META["palette"],
        "seed": seed,
        "epoch": epoch,
        "occupancy": round(float((dhw > 0).mean()), 4),
        "elements": int(len(np.unique(dhw[dhw > 0]))),
        "sample_ms": round(ms, 1),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "default_epoch": DEFAULT["epoch"],
            "epochs": {e: {k: v for k, v in src.items() if k != "file"} for e, src in SOURCES.items()}}


@app.get("/api/metrics")
def metrics() -> dict:
    tr = META["epochs"][str(DEFAULT["epoch"])].get("training", {})
    keys = ("fake_occupancy", "real_occupancy", "occupancy_gap", "disc_real_acc", "disc_fake_acc", "gen_loss", "disc_loss")
    return {"epoch": DEFAULT["epoch"], "model_version": SOURCES[DEFAULT["epoch"]]["version"],
            **{k: tr[k] for k in keys if k in tr}}


@app.get("/api/volume")
def volume(seed: int | None = None, epoch: int | None = None) -> dict:
    if seed is not None and not 0 <= seed <= MAX_SEED:
        raise HTTPException(status_code=400, detail="seed out of range")
    return sculpt(seed, epoch)


@app.post("/predict")
def predict(req: SculptRequest) -> dict:
    return sculpt(req.seed, req.epoch)


# Last: _load warms through sculpt(). With a tracker it runs in a daemon thread (retried until it
# works); offline (local weights) inline, so an import still fails loudly on a broken weights dir.
if TRACKER:
    registry.run_in_background(_load)
else:
    _load()
    registry.start({}, {})  # no tracker: marks the gate ready
