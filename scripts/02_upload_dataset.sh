#!/usr/bin/env bash
set -euo pipefail
: "${GCS_BUCKET:?Set GCS_BUCKET}"
ROOT="${1:-data/kaggle}"
REAL="$(find "$ROOT" -type d -iname original | head -1)"
FAKE="$(find "$ROOT" -type d -iname Deepfakes | head -1)"
gcloud storage cp --recursive "$REAL" "gs://${GCS_BUCKET}/raw/"
gcloud storage cp --recursive "$FAKE" "gs://${GCS_BUCKET}/raw/"
