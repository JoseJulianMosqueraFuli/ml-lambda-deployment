# Deployment Guide

## Overview

This project deploys a trained ML model (Iris classifier) to AWS Lambda with API Gateway for serverless inference.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT PIPELINE                          │
│                                                                   │
│  [GitHub Push] ──► [CI: Test + Lint] ──► [Build Lambda ZIP]      │
│                                                │                  │
│                                                ▼                  │
│  [Manual Trigger] ──► [Deploy Workflow] ──► [S3 Upload]           │
│                                                │                  │
│                                                ▼                  │
│                                    [Lambda Update + Publish]      │
│                                                │                  │
│                                                ▼                  │
│                                    [Smoke Test Verification]      │
└──────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### AWS Resources (via Terraform)
```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```

This creates:
- Lambda function with Python 3.12 runtime
- API Gateway (HTTP API) with CORS
- S3 bucket for deployment packages
- IAM roles with least-privilege
- CloudWatch log groups + alarms

### GitHub Secrets
Configure in repo Settings > Secrets:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### GitHub Environments
Create `staging` and `production` environments with appropriate protection rules.

## Deployment Methods

### 1. Automated (GitHub Actions)

**CI Pipeline** (automatic on push/PR):
- Runs tests with coverage (must be > 80%)
- Lint (ruff), type check (mypy), format check (black)
- Security scan (bandit)
- Builds Lambda package as artifact

**Deploy Pipeline** (manual trigger):
1. Go to Actions > "Deploy to AWS Lambda"
2. Select environment (staging/production)
3. Click "Run workflow"

### 2. Manual (CLI)

```bash
# 1. Train model
poetry run train

# 2. Build package
mkdir -p build/lambda
poetry export -f requirements.txt --without-hashes -o build/lambda/requirements.txt
pip install -r build/lambda/requirements.txt -t build/lambda/
cp -r src/ml_lambda build/lambda/
cp artifacts/*.joblib build/lambda/
cd build/lambda && zip -r ../../lambda-deployment.zip .

# 3. Upload to S3
aws s3 cp lambda-deployment.zip \
  s3://ml-lambda-deployment-artifacts/staging/lambda-deployment-latest.zip

# 4. Update Lambda
aws lambda update-function-code \
  --function-name ml-iris-predictor-staging \
  --s3-bucket ml-lambda-deployment-artifacts \
  --s3-key staging/lambda-deployment-latest.zip \
  --publish

# 5. Test
aws lambda invoke \
  --function-name ml-iris-predictor-staging \
  --payload '{"body": "{\"features\": [5.1, 3.5, 1.4, 0.2]}"}' \
  output.json

cat output.json
```

## Testing the Deployed API

### Via API Gateway
```bash
# Health check
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/health

# Predict
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Expected Response
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [0.95, 0.03, 0.02],
  "latency_ms": 12.5
}
```

## Monitoring

### CloudWatch Logs
```bash
aws logs tail /aws/lambda/ml-iris-predictor-staging --follow
```

### Alarms
- **Error Rate**: Triggers if > 5 errors in 2 minutes
- **Duration**: Triggers if average > 10 seconds for 3 minutes

### Estimated Costs (Monthly)

| Resource | Free Tier | After Free Tier |
|----------|-----------|-----------------|
| Lambda (1M requests) | Free | ~$0.20 |
| API Gateway (1M requests) | Free first 12 months | ~$1.00 |
| S3 Storage | < $0.01 | < $0.01 |
| CloudWatch Logs | First 5GB free | ~$0.50/GB |
| **Total** | **~$0** | **~$2/month** |

## Rollback

```bash
# List versions
aws lambda list-versions-by-function \
  --function-name ml-iris-predictor-staging

# Rollback to previous version
aws lambda update-alias \
  --function-name ml-iris-predictor-staging \
  --name live \
  --function-version <previous-version>
```
