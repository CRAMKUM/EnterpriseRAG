#!/bin/bash
set -e

echo "=== Enterprise RAG Deployment ==="

CONFIG_FILE="config/samples/graph_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

PROJECT_ID=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['project_id'])")
REGION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['location'])")
BUCKET_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['bucket_name'])")

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

echo "Enabling GCP APIs..."
gcloud services enable run.googleapis.com \
                       spanner.googleapis.com \
                       storage-api.googleapis.com \
                       artifactregistry.googleapis.com \
                       cloudbuild.googleapis.com \
                       aiplatform.googleapis.com \
                       --project="$PROJECT_ID"

if ! gsutil ls -b "gs://$BUCKET_NAME" >/dev/null 2>&1; then
    echo "Creating GCS Bucket gs://$BUCKET_NAME..."
    gsutil mb -l "$REGION" -p "$PROJECT_ID" "gs://$BUCKET_NAME"
fi

echo "Uploading config samples to GCS..."
gsutil -m cp config/samples/*.json "gs://$BUCKET_NAME/config/"

echo "Building and deploying to Cloud Run..."
gcloud run deploy enterprise-rag \
    --source . \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars CONFIG_BUCKET_NAME="$BUCKET_NAME",GOOGLE_CLOUD_PROJECT="$PROJECT_ID"

echo "=== Deployment Successful ==="
