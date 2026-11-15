#!/usr/bin/env bash
# Build (Cloud Build, amd64) and deploy deepsculpt-serve to Cloud Run on garassino-ml.
#   bash services/portfolio/deploy.sh
# The image carries no weights: at boot app.py loads deepsculpt-gan@champion/@challenger from the
# shared ML tracker as the read-only MLflow user `deepsculpt` (password: garassino-op secret
# mlflow-user-deepsculpt, readable by the serve's runtime service account). REGISTRY_GATE_WAIT_S=50 keeps
# the boot gate's request hold under this service's 60 s Cloud Run timeout.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT=garassino-ml
REGION=europe-west1
SERVICE=deepsculpt-serve
TRACKER=https://garassino-mlflow-mjz4n7eeia-ew.a.run.app
TRACKER_PW_SECRET="projects/634336216563/secrets/mlflow-user-deepsculpt:latest"
TAG=$(git rev-parse --short HEAD)
IMAGE=europe-west1-docker.pkg.dev/${PROJECT}/ml-images/${SERVICE}:${TAG}

gcloud builds submit --project "$PROJECT" --region "$REGION" --tag "$IMAGE" .
DIGEST=$(gcloud artifacts docker images describe "$IMAGE" --format='value(image_summary.digest)')
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --image "${IMAGE%:*}@${DIGEST}" --port 8080 --cpu 2 --memory 2Gi \
  --min-instances 0 --max-instances 2 --concurrency 4 --cpu-boost --timeout 60 \
  --set-env-vars "MLFLOW_TRACKING_URI=${TRACKER},MLFLOW_TRACKING_USERNAME=deepsculpt,DEEPSCULPT_MODEL=deepsculpt-gan,REGISTRY_GATE_WAIT_S=50" \
  --set-secrets "MLFLOW_TRACKING_PASSWORD=${TRACKER_PW_SECRET}" \
  --allow-unauthenticated --labels app=deepsculpt,role=serve
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)'
