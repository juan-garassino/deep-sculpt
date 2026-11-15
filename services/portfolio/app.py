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
version's TorchScript file through the tracker's artifact proxy. Promote a version in
MLflow and restart the serve — no rebuild. Without MLFLOW_TRACKING_URI it reads a local
weights dir (export.py output) for offline dev.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACKER = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")  # e.g. https://garassino-mlflow-…run.app
MODEL = os.environ.get("DEEPSCULPT_MODEL", "deepsculpt-gan")  # registered model name
ALIASES = os.environ.get("DEEPSCULPT_ALIASES", "champion,challenger").split(",")  # first = default
WEIGHTS = Path(os.environ.get("DEEPSCULPT_WEIGHTS", Path(__file__).parent / "weights"))  # e.g. /tmp/weights
MAX_SEED = 2**31 - 1
ARTIFACTS = "mlflow-artifacts:/"


def _tracker_get(path: str, query: dict | None = None) -> bytes:
    url = f"{TRACKER}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(url)
    user, pw = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if user and pw:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode())
    with urllib.request.urlopen(req, timeout=180) as r:  # the tracker scales to zero; first call is slow
        return r.read()


def _from_registry() -> dict:
    """Download every alias's generator into WEIGHTS; return {epoch: {alias, version, file}}."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    loaded: dict[int, dict] = {}
    for alias in ALIASES:
        mv = json.loads(_tracker_get("/api/2.0/mlflow/registered-models/alias", {"name": MODEL, "alias": alias}))["model_version"]
        epoch = int({t["key"]: t["value"] for t in mv.get("tags", [])}["epoch"])
        src = mv["source"]
        if not src.startswith(ARTIFACTS):
            raise RuntimeError(f"{MODEL} v{mv['version']}: unsupported source {src}")
        base = f"/api/2.0/mlflow-artifacts/artifacts/{src[len(ARTIFACTS):]}"
        fname = f"generator_e{epoch}.ts"
        (WEIGHTS / fname).write_bytes(_tracker_get(f"{base}/{fname}"))
        if not (WEIGHTS / "meta.json").exists():
            (WEIGHTS / "meta.json").write_bytes(_tracker_get(f"{base}/meta.json"))
        loaded.setdefault(epoch, {"alias": alias, "version": mv["version"], "file": fname})
        logger.info("%s@%s -> v%s (epoch %s)", MODEL, alias, mv["version"], epoch)
    return loaded


torch.set_num_threads(max(1, os.cpu_count() or 1))
if TRACKER:
    SOURCES = _from_registry()
else:
    SOURCES = {int(e): {"alias": "local", "version": None, "file": f"generator_e{e}.ts"}
               for e in sorted(json.loads((WEIGHTS / "meta.json").read_text())["epochs"], key=int, reverse=True)}
META = json.loads((WEIGHTS / "meta.json").read_text())
EPOCHS = sorted(SOURCES)
DEFAULT_EPOCH = next(iter(SOURCES))  # champion's epoch (first alias), or the first local one

app = FastAPI(title="deepsculpt-serve")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


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
    epoch = DEFAULT_EPOCH if epoch is None else epoch
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
    return {"status": "ok", "model": MODEL, "default_epoch": DEFAULT_EPOCH,
            "epochs": {e: {k: v for k, v in src.items() if k != "file"} for e, src in SOURCES.items()}}


@app.get("/api/metrics")
def metrics() -> dict:
    tr = META["epochs"][str(DEFAULT_EPOCH)].get("training", {})
    keys = ("fake_occupancy", "real_occupancy", "occupancy_gap", "disc_real_acc", "disc_fake_acc", "gen_loss", "disc_loss")
    return {"epoch": DEFAULT_EPOCH, "model_version": SOURCES[DEFAULT_EPOCH]["version"], **{k: tr[k] for k in keys if k in tr}}


@app.get("/api/volume")
def volume(seed: int | None = None, epoch: int | None = None) -> dict:
    if seed is not None and not 0 <= seed <= MAX_SEED:
        raise HTTPException(status_code=400, detail="seed out of range")
    return sculpt(seed, epoch)


@app.post("/predict")
def predict(req: SculptRequest) -> dict:
    return sculpt(req.seed, req.epoch)


for _ in range(3):  # TorchScript optimizes over its first runs; pay that at boot, not on a visitor
    sculpt(0, DEFAULT_EPOCH)
