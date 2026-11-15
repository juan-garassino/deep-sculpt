"""Serve-side MLflow registry client: the serve loads `<model>@champion`, nothing else.

At boot the app calls `start(MODELS, ENV)`. With MLFLOW_TRACKING_URI set it
points the app's env vars (MODEL_PATH, EVAL_DIR, ...) at their local paths under REGISTRY_DIR
right away, then a background thread runs `materialize`:
  1. resolves each registered model's alias (default `champion`) on the tracker,
  2. downloads that version's single `.keras` to the local path the app expects,
  3. downloads the `serve/` subtree of each champion's RUN (metadata such as meta.json, the
     eval metrics, held-out sample inputs/evidence) into the same local root — first model
     listed wins on a name clash,
  4. runs the `on_ready(fn)` hooks (apps that build state from those files), then flips
     `ready()`.
(`run_in_background(load)` is the same loop around any loader, for serves whose artifacts are
not one .keras per model — deep-sculpt's TorchScript files reuse this module that way.)
The HTTP server is up the whole time: `install(app)` adds a gate in front of the app. Until
ready, /health answers 503 {"status": "loading"} (after holding up to REGISTRY_HEALTH_WAIT_S)
and every other request is held up to REGISTRY_GATE_WAIT_S, then answered normally. So the
Cloud Run startup probe passes at once, and a busy tracker can no longer turn boot into an
exit(1) crash loop. Why hold at all: with request-based billing Cloud Run gives an instance CPU
only while a request is in flight, so a held request is what keeps the background load moving
(a /health that answered in 1 ms would leave the download throttled until the next request).

Robustness:
  - every tracker / GCS call retries 429, 5xx, connection errors, timeouts and truncated reads
    with exponential backoff + jitter for up to REGISTRY_RETRY_BUDGET_S (~3 min);
  - if a whole load still fails, the thread reports "failed" on /health and tries again after
    REGISTRY_COOLDOWN_S — the instance stays up instead of crash-looping.

Where the bytes come from:
  - small artifacts (<= REGISTRY_DIRECT_MIN_BYTES, 30 MiB) through the tracker's artifact proxy
    (`/api/2.0/mlflow-artifacts/artifacts`) with the serve's own read-only basic-auth user;
  - larger `mlflow-artifacts:/` files straight from the tracker's backing bucket
    (MLFLOW_ARTIFACTS_DESTINATION, same object path) — Cloud Run truncates big proxy responses;
  - a version whose `source` is any other `gs://` URI (the "GCS bytes, MLflow index" convention,
    tag storage=gcs-direct) straight from GCS.
  GCS reads use the JSON API with the runtime service account's token (metadata server; locally
  `gcloud auth print-access-token`). Stdlib only: no TF, no google-cloud-storage.

Without MLFLOW_TRACKING_URI it is a no-op and ready at once, so tests and local runs keep their
env defaults. Copy-pasted identically into every 027-ml-workspace serve (no shared pkg).
"""
from __future__ import annotations

import asyncio
import base64
import http.client
import json
import logging
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

PROXY = "/api/2.0/mlflow-artifacts/artifacts"
GCS_API = "https://storage.googleapis.com/storage/v1"
METADATA_TOKEN = ("http://metadata.google.internal/computeMetadata/v1/instance/"
                  "service-accounts/default/token")
SERVE_DIR = "serve"   # run-level artifact dir holding everything the serve needs besides weights
HEALTH_PATHS = ("/health", "/api/health")
SOURCES: dict[str, dict] = {}   # registered model -> {alias, version, run_id, file}; for /api/model
STATE: dict = {"status": "idle", "error": None, "attempts": 0}   # what /health shows until ready
_READY = threading.Event()
_HOOKS: list[Callable[[], None]] = []    # on_ready functions, run by the loader before _READY
_HOOK_LOCK = threading.Lock()
_TOKEN: dict = {"value": "", "expires": 0.0}
_sleep = time.sleep          # tests swap these two for a fake clock
_clock = time.monotonic


def _tracker() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")  # e.g. https://garassino-mlflow-…run.app


def _alias() -> str:
    return os.environ.get("MODEL_ALIAS", "champion")  # e.g. challenger (to serve the challenger)


def _root() -> Path:
    return Path(os.environ.get("REGISTRY_DIR", "/tmp/registry"))  # e.g. /tmp/registry


def _destination() -> str:
    # the tracker's --artifacts-destination: mlflow-artifacts:/x is the object <destination>x
    return os.environ.get("MLFLOW_ARTIFACTS_DESTINATION", "gs://garassino-ml-mlflow-artifacts/")


def _budget() -> float:
    return float(os.environ.get("REGISTRY_RETRY_BUDGET_S", "180"))  # e.g. 180 (per call)


def _direct_min() -> int:
    return int(os.environ.get("REGISTRY_DIRECT_MIN_BYTES", str(30 << 20)))  # e.g. 31457280


def _cooldown() -> float:
    return float(os.environ.get("REGISTRY_COOLDOWN_S", "60"))  # e.g. 60 (between whole loads)


def _gate_wait() -> float:
    return float(os.environ.get("REGISTRY_GATE_WAIT_S", "90"))  # e.g. 90 (< the gateway's 120)


def _health_wait() -> float:
    return float(os.environ.get("REGISTRY_HEALTH_WAIT_S", "25"))  # e.g. 25 (then 503 "loading")


# ------------------------------------------------------------------------------ retries

class TruncatedDownload(RuntimeError):
    """A download ended short of its listed size (retried like a dropped connection)."""


def _transient(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError,  # socket.timeout too
                          http.client.HTTPException, TruncatedDownload))


def _retrying(what: str, fn: Callable):
    """fn() with exponential backoff + jitter on transient errors, within _budget() seconds."""
    deadline, delay, attempt = _clock() + _budget(), 2.0, 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:  # classified right below
            pause = random.uniform(delay / 2, delay)
            if not _transient(e) or _clock() + pause > deadline:
                raise
            logger.warning("%s: %s (attempt %d), retrying in %.1fs", what, e, attempt, pause)
            _sleep(pause)
            delay = min(delay * 2, 30.0)


# ------------------------------------------------------------------------------ tracker

def _open(path: str, query: dict | None = None):
    """One tracker request (no retry here: callers wrap the whole read in _retrying)."""
    url = f"{_tracker()}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(url)
    user, pw = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if user and pw:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode())
    return urllib.request.urlopen(req, timeout=90)  # the tracker scales to zero: first call is slow


def _get_json(path: str, query: dict | None = None) -> dict:
    def once() -> dict:
        with _open(path, query) as r:
            return json.loads(r.read() or b"{}")
    return _retrying(f"GET {path}", once)


# ------------------------------------------------------------------------------ GCS

def _token() -> str:
    """Runtime SA access token (metadata server), else the local gcloud login; cached."""
    if _TOKEN["value"] and _clock() < _TOKEN["expires"]:
        return _TOKEN["value"]
    try:
        req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
        _TOKEN.update(value=body["access_token"], expires=_clock() + int(body.get("expires_in", 300)) - 60)
    except (urllib.error.URLError, OSError):   # not on GCP: a developer machine
        out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                             text=True, timeout=60, check=True).stdout.strip()
        _TOKEN.update(value=out, expires=_clock() + 600)
    return _TOKEN["value"]


def _gcs_open(url: str):
    """One GCS JSON-API request with the runtime SA's token (no retry, like _open)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    return urllib.request.urlopen(req, timeout=90)


def split_gs(uri: str) -> tuple[str, str]:
    """gs://bucket/a/b -> ("bucket", "a/b")."""
    if not uri.startswith("gs://"):
        raise RuntimeError(f"not a gs:// uri: {uri!r}")
    bucket, _, obj = uri[5:].partition("/")
    return bucket, obj.strip("/")


def _gcs_url(bucket: str, obj: str, media: bool) -> str:
    return f"{GCS_API}/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}" + ("?alt=media" if media else "")


def gcs_list(uri: str) -> list[tuple[str, int | None]]:
    """Objects under a gs:// prefix -> [(path relative to it, size)]; [] if absent."""
    bucket, prefix = split_gs(uri)
    out, page = [], ""
    while True:
        q = {"prefix": prefix + "/", "fields": "items(name,size),nextPageToken"}
        if page:
            q["pageToken"] = page

        def once(q=q) -> dict:
            with _gcs_open(f"{GCS_API}/b/{bucket}/o?{urllib.parse.urlencode(q)}") as r:
                return json.loads(r.read() or b"{}")
        body = _retrying(f"list {uri}", once)
        out += [(o["name"][len(prefix) + 1:], int(o["size"])) for o in body.get("items", [])]
        page = body.get("nextPageToken", "")
        if not page:
            return out


def gcs_size(uri: str) -> int:
    def once() -> int:
        with _gcs_open(_gcs_url(*split_gs(uri), media=False)) as r:
            return int(json.loads(r.read())["size"])
    return _retrying(f"stat {uri}", once)


# ------------------------------------------------------------------------------ artifacts

def proxy_path(uri: str) -> str:
    """Artifact URI -> path under the proxy (`mlflow-artifacts:/x` or `<destination>x` -> `x`)."""
    for prefix in ("mlflow-artifacts:/", _destination()):
        if uri.startswith(prefix):
            return uri[len(prefix):].strip("/")
    raise RuntimeError(f"artifact uri {uri!r} is not served by the tracker's artifact proxy")


def list_files(path: str) -> list[tuple[str, int | None]]:
    """Recursive listing under a proxy path -> [(path relative to it, size)]; [] if absent."""
    out: list[tuple[str, int | None]] = []
    for f in _get_json(PROXY, {"path": path}).get("files", []):
        if f.get("is_dir"):
            out += [(f"{f['path']}/{p}", s) for p, s in list_files(f"{path}/{f['path']}")]
        else:
            out.append((f["path"], f.get("file_size")))
    return out


def _fetch(location: str, dest: Path, size: int | None) -> None:
    """One attempt: stream `location` (a gs:// URI or a proxy path) to dest via a .part file."""
    opened = (_gcs_open(_gcs_url(*split_gs(location), media=True)) if location.startswith("gs://")
              else _open(f"{PROXY}/{urllib.parse.quote(location)}"))
    tmp = dest.with_name(dest.name + ".part")
    with opened as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if size is not None and tmp.stat().st_size != size:
        got = tmp.stat().st_size
        tmp.unlink()
        raise TruncatedDownload(f"{location}: got a truncated download ({got} of {size} bytes)")
    tmp.replace(dest)


def download(location: str, dest: Path, size: int | None = None, fallback: str | None = None) -> None:
    """Fetch one artifact to `dest` (skipped when already there with the right size), retrying.

    `fallback`: the proxy path of a large mlflow-artifacts file read from the bucket — used when
    the bucket answers 403/404 (the runtime SA has no grant on that run's prefix yet).
    """
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _retrying(f"download {location}", lambda: _fetch(location, dest, size))
    except urllib.error.HTTPError as e:
        if fallback in (None, location) or e.code not in (403, 404):
            raise
        logger.warning("%s: HTTP %s from GCS, falling back to the tracker proxy", location, e.code)
        _retrying(f"download {fallback}", lambda: _fetch(fallback, dest, size))


def where(path: str, size: int | None) -> str:
    """Proxy path -> where to read it: the proxy, or the backing bucket when the file is large."""
    if size is not None and size > _direct_min() and _destination().startswith("gs://"):
        return _destination().rstrip("/") + "/" + path
    return path


def resolve(name: str, alias: str | None = None) -> dict:
    """`name@alias` -> the model-version dict (raises if the alias is not set)."""
    query = {"name": name, "alias": alias or _alias()}
    return _get_json("/api/2.0/mlflow/registered-models/alias", query)["model_version"]


def _run_artifacts(run_id: str) -> str:
    run = _get_json("/api/2.0/mlflow/runs/get", {"run_id": run_id})["run"]
    return proxy_path(run["info"]["artifact_uri"])


def _weights(name: str, mv: dict) -> tuple[str, str, int | None, str | None]:
    """The version's single .keras -> (where to download it, file name, size, proxy fallback)."""
    src = mv["source"]
    if src.startswith("gs://") and not src.startswith(_destination()):   # gcs-direct version
        if src.endswith(".keras"):
            return src, src.rsplit("/", 1)[-1], gcs_size(src), None
        found = [(f"{src.rstrip('/')}/{p}", p, s, None) for p, s in gcs_list(src) if p.endswith(".keras")]
    else:
        base = proxy_path(src)
        found = [(where(f"{base}/{p}", s), p, s, f"{base}/{p}")
                 for p, s in list_files(base) if p.endswith(".keras")]
    if len(found) != 1:
        raise RuntimeError(f"{name} v{mv['version']}: expected one .keras under {src}, "
                           f"got {[f[1] for f in found]}")
    return found[0]


def materialize(models: dict[str, str], env: dict[str, str]) -> dict[str, dict]:
    """models: {registered name: local relpath of its .keras}; env: {ENV VAR: relpath or ""}.

    Downloads every champion + its run's `serve/` bundle under REGISTRY_DIR and rewrites the
    env vars to the local paths. Returns SOURCES. No-op ({}) without MLFLOW_TRACKING_URI.
    Blocking; `start` runs it in a background thread.
    """
    if not _tracker():
        return {}
    root, alias = _root(), _alias()

    def locate(name: str) -> tuple[dict, tuple, str, list]:
        mv = resolve(name, alias)
        bundle = f"{_run_artifacts(mv['run_id'])}/{SERVE_DIR}"
        return mv, _weights(name, mv), bundle, list_files(bundle)

    # every tracker call pays a basic-auth hash check server-side: resolve all models at once,
    # then fetch every file at once (8 workers) — boot time stays flat in the number of models
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(locate, models))
    jobs, claimed = [], set()
    for (name, rel), (mv, (loc, fname, size, fallback), bundle, files) in zip(models.items(), found):
        jobs.append((loc, root / rel, size, fallback))
        for p, s in files:
            if p not in claimed and not (root / p).exists():   # first model listed wins a clash
                claimed.add(p)
                jobs.append((where(f"{bundle}/{p}", s), root / p, s, f"{bundle}/{p}"))
        direct = loc.startswith("gs://")
        SOURCES[name] = {"alias": alias, "version": mv["version"], "run_id": mv["run_id"],
                         "file": fname, "serve_files": len(files), "direct_gcs": direct}
        logger.info("%s@%s -> v%s (%s%s, %d serve files)", name, alias, mv["version"], fname,
                    ", direct from GCS" if direct else "", len(files))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    _point_env(env)
    return SOURCES


def _point_env(env: dict[str, str]) -> None:
    root = _root()
    for var, rel in env.items():   # "" disables a var (e.g. a gs:// sync prefix the app must skip)
        os.environ[var] = str(root / rel) if rel else ""


# ------------------------------------------------------------------------------ background boot

def ready() -> bool:
    return _READY.is_set()


def wait_ready(timeout: float | None = None) -> bool:
    """Block until the registry files are in place (for warm-up threads)."""
    return _READY.wait(timeout)


def on_ready(fn: Callable[[], None]) -> Callable[[], None]:
    """Run fn once the registry files are in place: now if already ready (no tracker), else on
    the loader thread just before ready() flips. A raising fn fails that load attempt (retried),
    so keep it idempotent. Usable as a decorator."""
    with _HOOK_LOCK:
        if not _READY.is_set():
            _HOOKS.append(fn)
            return fn
    fn()
    return fn


def _load_until_ready(load: Callable[[], None]) -> None:
    while True:
        STATE.update(status="loading", attempts=STATE["attempts"] + 1)
        try:
            load()
            with _HOOK_LOCK:
                for fn in _HOOKS:
                    fn()
                STATE.update(status="ready", error=None)
                _READY.set()
            return
        except Exception as e:  # stay up, report it on /health, try again
            logger.exception("registry load failed (attempt %d); retrying in %ss",
                             STATE["attempts"], _cooldown())
            STATE.update(status="failed", error=f"{type(e).__name__}: {e}"[:400])
            _sleep(_cooldown())


def run_in_background(load: Callable[[], None], background: bool = True) -> None:
    """Run load() (+ the on_ready hooks) until it succeeds, then flip ready() — in a daemon thread
    unless background=False. The generic form of `start`, for serves whose artifacts are not one
    .keras per registered model (deep-sculpt's TorchScript files)."""
    if background:
        threading.Thread(target=_load_until_ready, args=(load,), name="registry-load", daemon=True).start()
    else:
        _load_until_ready(load)


def start(models: dict[str, str], env: dict[str, str], background: bool = True) -> None:
    """Point env at REGISTRY_DIR now, then materialize (+ on_ready hooks) in a daemon thread.

    Without MLFLOW_TRACKING_URI: env untouched, ready at once (hooks run as registered).
    background=False runs the load inline (scripts, tests).
    """
    if not _tracker():
        STATE.update(status="ready")
        _READY.set()
        return
    _point_env(env)   # the paths are fixed up front; the files land under them later
    run_in_background(lambda: materialize(models, env), background)


class ReadinessGate:
    """ASGI gate: until ready, requests wait for readiness (health REGISTRY_HEALTH_WAIT_S, the
    rest REGISTRY_GATE_WAIT_S), then get 503 {"status": "loading"|"failed"} + Retry-After.

    CORS preflights pass straight through, and the 503 carries `access-control-allow-origin: *`
    itself, so it works whatever the middleware order.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or _READY.is_set() or scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        wait = _health_wait() if scope.get("path") in HEALTH_PATHS else _gate_wait()
        deadline = time.monotonic() + wait
        while not _READY.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        if _READY.is_set():
            return await self.app(scope, receive, send)
        body = json.dumps({"status": STATE["status"], "model_loaded": False,
                           "error": STATE["error"], "attempts": STATE["attempts"],
                           "detail": "loading the model from the MLflow registry"}).encode()
        await send({"type": "http.response.start", "status": 503, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
            (b"retry-after", b"5"), (b"access-control-allow-origin", b"*"),
            (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


def install(app) -> None:
    """Add the readiness gate to a FastAPI/Starlette app."""
    app.add_middleware(ReadinessGate)
