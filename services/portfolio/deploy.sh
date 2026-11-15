#!/usr/bin/env bash
# Build (Cloud Build, amd64) and deploy deepsculpt-serve to Cloud Run on garassino-ml.
#   bash services/portfolio/deploy.sh            # weights must exist in services/portfolio/weights
#   WEIGHTS_FROM_GCS=1 bash services/portfolio/deploy.sh   # pull the exported weights first
set -euo pipefail
cd "$(dirname "$0")"
PROJECT=garassino-ml
REGION=europe-west1
SERVICE=deepsculpt-serve
GCS_WEIGHTS=gs://garassino-ml-artifacts/deepsculpt/serve/gan-shodhan-color-001
TAG=$(git rev-parse --short HEAD)
IMAGE=europe-west1-docker.pkg.dev/${PROJECT}/ml-images/${SERVICE}:${TAG}

if [ "${WEIGHTS_FROM_GCS:-0}" = 1 ]; then
  mkdir -p weights
  CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0 gcloud storage cp "${GCS_WEIGHTS}/*" weights/
fi
test -f weights/meta.json || { echo "no weights/ — run export.py or WEIGHTS_FROM_GCS=1" >&2; exit 1; }

gcloud builds submit --project "$PROJECT" --region "$REGION" --tag "$IMAGE" .
DIGEST=$(gcloud artifacts docker images describe "$IMAGE" --format='value(image_summary.digest)')
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --image "${IMAGE%:*}@${DIGEST}" --port 8080 --cpu 2 --memory 2Gi \
  --min-instances 0 --max-instances 2 --concurrency 4 --cpu-boost --timeout 60 \
  --allow-unauthenticated --labels app=deepsculpt,role=serve
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)'
