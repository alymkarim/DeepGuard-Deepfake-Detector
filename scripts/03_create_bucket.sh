#!/usr/bin/env bash
set -euo pipefail
: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCS_BUCKET:?Set GCS_BUCKET}"
GCP_REGION="${GCP_REGION:-europe-west1}"
gcloud config set project "$GCP_PROJECT_ID"
gcloud services enable aiplatform.googleapis.com storage.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud storage buckets create "gs://${GCS_BUCKET}" --location="$GCP_REGION" --uniform-bucket-level-access
