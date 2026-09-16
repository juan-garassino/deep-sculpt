# DeepSculpt

> **GCP migration note (2026-06-06):** Lives in `garassino-ml`. Images on `ghcr.io/juan-garassino/deepsculpt-runpod` (public). Artifacts in `gs://garassino-ml-artifacts/deepsculpt/`. No always-on resources — training is RunPod-driven. See root `CLAUDE.md` § "GCP architecture".

## What this project is
A 3D generative art system that learns to create sculptures from scratch.
It uses two types of AI models:
- A **GAN** (two networks fighting each other — one creates shapes, one judges them)
- A **Diffusion model** (gradually removes noise to reveal a shape, like developing a photo)

The sculptures are stored as 3D grids of numbers (numpy arrays):
- Monochrome shapes: a cube of numbers, each cell says "empty" (0) or "solid" (1)
- Color shapes: same cube but each cell also has a color encoded as 4 numbers (RGBA)

## What I'm trying to do
Train the AI to generate believable 3D shapes it has never seen before,
then navigate the "space" of all possible shapes it has learned — finding
new sculptures, blending between two shapes, or asking "show me something rounder".

## The pipeline (in order)
1. **Generate training data** — create thousands of example 3D shapes using math
2. **Validate the data** — make sure the shapes look right before training
3. **Train the GAN or diffusion model** on those shapes
4. **Evaluate** — check if the AI is learning or getting stuck
5. **Explore** — navigate the space of shapes the AI has learned

## Stack
- Python, NumPy, PyTorch
- Shapes stored as `.npy` files
- Training runs on GPU (CUDA)

## Skills available to help
- `ds-datagen` — help writing shape generation code
- `ds-dataval` — help writing data checking and visualization code
- `ds-gan` — help with GAN architecture and training code
- `ds-diffusion` — help with diffusion model code
- `ds-latent` — help with latent space navigation code
- `ds-improve` — autonomous agent that runs improvement loops on the codebase

## ⚡ AGENT BEHAVIOR — READ THIS FIRST

When the user says anything like:
- "this isn't working" / "it's broken" / "fix this"
- "the training is getting worse"
- "the shapes all look the same"
- "make it better" / "help me improve this"
- "something is wrong"

**Do NOT just give advice. Immediately launch the `ds-improve` agent.**
Run it autonomously: measure diversity and validity of generated shapes,
diagnose what's failing, apply targeted fixes to the code, and report
what changed.

When the user says anything like:
- "explain how X works"
- "I don't understand X"
- "what is X"

Use the skills interactively to explain — don't launch the agent.

When the user says anything like:
- "write the training loop"
- "help me build the generator"
- "implement X"

Use the relevant skill interactively to write the code together.

## How to ask (no jargon needed)
- "The shapes all look the same" → launches ds-improve (mode collapse)
- "Training is crashing" → launches ds-improve
- "Something looks wrong with the data" → launches ds-improve
- "Blend between two sculptures" → interactive, uses ds-latent
- "Explain what a GAN is" → interactive explanation
- "Write the shape generator" → interactive, uses ds-datagen

## Current status
- **PyTorch v2 is the only path.** TensorFlow/Keras legacy is archived under `boilerplate/`.
- **CLI works.** `python -m deepsculpt.main {train-gan,train-diffusion,generate-data,sample-gan,sample-diffusion,latent-walk,latent-traverse,latent-directions,preprocess,visualize,benchmark,evaluate,export}`.
- **Latent navigation is live** (`core/latent/`): slerp/lerp walks, per-dim traversal, GANSpace/PCA semantic directions, diffusion noise-space walks (`latent-walk --backend diffusion`), rendered as GIF/PNG/OBJ/STL via `core/visualization/volume_export.py`. Pass negative alpha lists as `--alphas="-2,0,2"`.
- **Optional-dep imports are guarded with `except Exception`** (Prefect 2/3 raises ValidationError, not ImportError — don't narrow these back).
- **Architectural data generator is the active mode** — columns + slabs + 3 orthogonal pipes (red/blue/yellow). See recent `git log` for the procedural-shape tuning history.
- **Test suite has pre-existing import bugs** (all `tests/unit/*.py` and `tests/integration/*` import the dead `deepSculpt` casing — package is `deepsculpt`). Don't trust `pytest` as a green light; smoke-test via the CLI instead. `tests/test_latent_ops.py` is the exception (pure tensor tests; `tests/conftest.py` is torch-only now, coverage is opt-in).
- **Cloud training is live**: `runpod/` directory ships a CUDA 12.8 + Claude Code container that runs on RunPod and syncs checkpoints/results to GCS bucket `garassino-ml-artifacts`. See `runpod/README.md`. `MODE=train` accepts `train_cmd`/`train_args` workflow inputs and auto-generates data when the GCS cache is empty.
- **This dev machine is old — no local training.** Verify with forward passes and nano runs only (void_dim 16, 1-2 minibatches max); real training goes to RunPod.

## Cloud training (RunPod + GCS + Claude-in-the-loop)
The `runpod/` directory contains the full deploy. Four modes:
- `MODE=train` — pure training, no Claude in the loop.
- `MODE=research` — one-shot Claude reading `runpod/prompts/research.md`, runs experiments to `TIME_BUDGET`.
- `MODE=improve` — one-shot Claude reading `runpod/prompts/improve.md`, drives the `ds-improve` skill once.
- `MODE=self-improve` — **continuous** Claude loop; each iteration uses `runpod/prompts/self_improve.md` which references `runpod/prompts/autoresearch_program.md` (mirrored from the [020-autoresearch](../020-autoresearch/) reference repo). **Toggleable on/off** via a GCS-synced object (`gs://garassino-ml-artifacts/deepsculpt/control/self_improve.enabled`); `make toggle-on` / `make toggle-off` from the `runpod/` Makefile — no pod restart needed.

Pod auth: **GHA-minted short-lived bearer token (WIF through `garassino-op`'s `gh-actions` pool). No SA JSON anywhere.** Refreshed every 50 min by `.github/workflows/refresh-token.yml`. Canonical deploy path: `gh workflow run deploy-runpod.yml -f mode=...` (or `make deploy MODE=...`). Infra is show-and-destroy via `infra/gcp/` Terraform.

GCS layout: `gs://garassino-ml-artifacts/deepsculpt/{data,checkpoints/<run_id>,results/<run_id>,prompts-archive,control,state}/` (bucket region: **`europe-west1`**). Crash-safe: periodic background `gsutil rsync` + final sync on `EXIT`. Image: `ghcr.io/juan-garassino/deepsculpt-runpod:latest` (built/pushed by `.github/workflows/build-push.yml` on every master push). See `runpod/README.md`, `docs/runpod.md`, and `infra/gcp/README.md`.

## File structure
```
deepsculpt/
├── main.py                       # CLI entry — 13 subcommands
├── config.yaml                   # central hyperparameters
├── core/
│   ├── data/{generation,loaders,sparse,transforms}/   # shape gen, dataloaders, encoding
│   ├── latent/{ops,loader,directions}.py              # latent navigation: walks, traversal, PCA directions
│   ├── models/
│   │   ├── gan/{generator,discriminator}.py           # 5 gens, 9 discs incl. SelfAttention3D, LightDiscriminator, SliceDiscriminator2D (+ MinibatchStdDev3D)
│   │   ├── diffusion/{unet,noise_scheduler,pipeline,pytorch_diffusion}.py  # pytorch_diffusion is unused reference code
│   │   ├── base_models.py, model_factory.py
│   ├── training/{gan_trainer,diffusion_trainer,base_trainer,optimizers,schedulers}.py
│   ├── utils/{logger,pytorch_utils,monitoring,performance_optimizer}.py
│   ├── visualization/{pytorch_visualization,volume_export}.py  # volume_export: pure GIF/PNG/mesh writers (no GCS)
│   └── workflow/{pytorch_workflow,pytorch_mlflow_tracking}.py
services/                         # Cloud Run trio: trainer job, inference FastAPI (has its own latent_walk endpoint), mlflow
notebooks/                        # 9 Jupyter notebooks (Colab + local)
scripts/                          # colab_train.py, colab_train_diffusion.py, autoresearch_report.py, preview_sample.py
tests/                            # pytest suite (most fixed via sed; 11 tier-2 errors remain — see Current status)
runpod/                           # RunPod + GCS deploy (Dockerfile, entrypoint.sh, Makefile, prompts/, scripts/)
infra/gcp/                        # show-and-destroy Terraform: runtime SA + WIF impersonation binding
.github/workflows/                # build-push, deploy-runpod, refresh-token (cron), notify-telegram (cron)
docs/                             # architecture, training, operations, inference, runpod, gcs_layout
boilerplate/                      # archived TF/Keras v1 code (do not propagate)
checkpoints/                      # local checkpoint dir; legacy samples committed under data/11/
```

## Ported upstream techniques (SliceGAN + 3DStyleGAN)

Ported from `003-knowledge-playgrounds/references/analysis/deep-sculpt-3d.md`. **All shape/import-verified only — NEEDS-GPU-VALIDATION** (see the RunPod checklist in PR `feat/port-slicegan-stability`). Do not treat as convergence-tested.

- **`SliceDiscriminator2D`** (`core/models/gan/discriminator.py`, registry key `"slice"`) — SliceGAN's 2D→3D slicing critic (stke9/SliceGAN `model.py:96-135`). A purely 2D WGAN critic scores axial slices of the 3D volume: `slice_volume()` turns `(B,C,D,H,W)` → `(l·B,C,l,l)` (via `movedim`, axis-correct for all 3 axes — the literal SliceGAN permute table only round-trips for isotropic cubes). `forward()` accepts a 5D volume (sliced internally along `axis`) or a pre-sliced 4D batch. Purpose: **train 3D generators from 2D reference imagery** with no 3D ground truth. Uses `BaseDiscriminator.input_channels` (6 color / 1 mono) — does NOT depend on the `feat/shodhan-dataset` `SEMANTIC_CLASSES` scheme.
- **`MinibatchStdDev3D`** (`core/models/gan/discriminator.py`) — 3D minibatch-stddev layer (sh4174/3DStyleGAN `networks3d_stylegan2.py:139`). Appends a batch-diversity channel to a critic's final block; anti-mode-collapse. Drop-in `nn.Module`, group-size clamps to divide the batch.
- **`r1_gradient_penalty(disc, real, gamma)`** (`core/training/gan_trainer.py`, module-level) — StyleGAN2 R1 penalty (3DStyleGAN `loss.py:52`), a cheaper/more-stable WGAN-GP alternative. Standalone testable twin of the existing `GANTrainer._compute_r1_penalty`. Config-toggled in the trainer via `TrainingConfig.gan_loss_type` (`"softplus"`→R1, `"wgan-gp"`→interpolated GP) + `r1_gamma`.
- **Volumetric eval bundle** (`core/training/training_metrics.py`, module-level fns): `msssim_3d` (3D MS-SSIM, volumetric-FID stand-in), `soft_dice` + `occupancy_class_weight` (vnet.pytorch imbalance-aware recon signal), `max_connected_component` (3dgan-release debris cleanup, rewritten on `scipy.ndimage.label`; accepts numpy or torch). Feeds the `ds-improve` eval loop.

Shape/import tests: `tests/unit/test_slicegan_stability_port.py` (21 tests). No local torch env — verified in an isolated CPU-torch venv; `tests/test_trainer.py` and 7 `test_pytorch_models.py` cases fail pre-existing on `main` (unrelated).
