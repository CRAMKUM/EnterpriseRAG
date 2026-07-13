# cloud_build_deploy.ps1 - Deploy using Cloud Build (for low-memory systems)
# This script builds the Docker image on Google Cloud Build instead of locally
# Perfect for 8GB RAM Windows machines - no local Docker resources needed!

$ErrorActionPreference = "Stop"

# ==============================================================================
# Configuration Banner
# ==============================================================================
Write-Host ""
Write-Host "  Enterprise RAG - Cloud Build Deployment (Low Memory)" -ForegroundColor Cyan
Write-Host "  Perfect for 8GB RAM systems - builds in the cloud!" -ForegroundColor Cyan
Write-Host ""

# ==============================================================================
# Pre-flight Checks (No Docker required!)
# ==============================================================================
Write-Host " Running pre-flight checks..." -ForegroundColor Yellow

# Check gcloud
try {
    $gcloudVersion = gcloud --version | Select-Object -First 1
    Write-Host "✅ gcloud CLI installed: $gcloudVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ gcloud CLI not found. Please install Google Cloud SDK" -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "✅ Python installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# ==============================================================================
# Load Configuration
# ==============================================================================
Write-Host "`n Loading configuration from config files..." -ForegroundColor Yellow

$configFile = "config\samples\graph_config.json"

if (-not (Test-Path $configFile)) {
    Write-Host "❌ Error: $configFile not found!" -ForegroundColor Red
    exit 1
}

try {
    $PROJECT_ID = python -c "import json; print(json.load(open('$configFile'))['project_id'])"
    $REGION = python -c "import json; print(json.load(open('$configFile'))['location'])"
    $BUCKET_NAME = python -c "import json; print(json.load(open('$configFile'))['bucket_name'])"
} catch {
    Write-Host "❌ Error parsing config file" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Project ID: $PROJECT_ID" -ForegroundColor Green
Write-Host "✅ Region: $REGION" -ForegroundColor Green
Write-Host "✅ Bucket: $BUCKET_NAME" -ForegroundColor Green

# Upload configs to GCS
Write-Host "`n Uploading config templates to GCS..." -ForegroundColor Yellow
gsutil -m cp config\samples\*.json "gs://$BUCKET_NAME/config/"
Write-Host "✅ Config files uploaded to gs://$BUCKET_NAME/config/" -ForegroundColor Green

# Build and Deploy
Write-Host "🚀 Building and deploying using Cloud Build..." -ForegroundColor Yellow
try {
    gcloud builds submit --config=cloudbuild.yaml --project=$PROJECT_ID
    Write-Host "✅ Cloud Build and deployment successful!" -ForegroundColor Green
} catch {
    Write-Host "❌ Cloud Build failed" -ForegroundColor Red
    exit 1
}
