#!/usr/bin/env bash
set -euo pipefail
: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
GCP_REGION="${GCP_REGION:-europe-west1}"
REPOSITORY="${REPOSITORY:-deepfake-training}"
IMAGE="${IMAGE:-deepfake-trainer}"
gcloud artifacts repositories describe "$REPOSITORY" --location="$GCP_REGION" >/dev/null 2>&1 || gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker --location="$GCP_REGION"
gcloud builds submit --config cloudbuild.yaml --substitutions="_REGION=${GCP_REGION},_REPOSITORY=${REPOSITORY},_IMAGE=${IMAGE}"
