# Pending Work & Roadmap

Items remaining to reach production readiness, organized by priority.

---

## High Priority

### 1. Deploy Module Implementation

The `deploy/` package has stub methods that need implementation:

**`deploy/packager.py` - PackageBuilder class:**
- `build()`: Create ZIP with source code, dependencies, and model artifact
- `_install_dependencies()`: Export Poetry deps and pip install to target directory
- `_compute_hash()`: Calculate SHA256 of the final package for integrity verification
- Size validation: Reject packages exceeding Lambda's 50MB limit

**`deploy/deployer.py` - AWSDeployer class:**
- `validate_credentials()`: Verify AWS credentials via STS GetCallerIdentity
- `deploy_lambda()`: Create or update Lambda function code from ZIP/S3
- `setup_api_gateway()`: Configure HTTP API with POST /predict route
- `deploy()`: Orchestrate full deployment (Lambda + API Gateway)
- `rollback()`: Revert to a previous Lambda version using aliases

### 2. Integration Tests

End-to-end tests covering the full inference pipeline:
- Load real trained model from artifacts
- Send sample requests through the Lambda handler
- Verify correct predictions with known inputs
- Test error handling with malformed requests
- Measure cold start and warm start latency

### 3. Production Deployment Guide

Step-by-step documentation for deploying to AWS:
- AWS account prerequisites (IAM user, permissions)
- GitHub secrets configuration (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Terraform init and apply for infrastructure
- First deployment via GitHub Actions workflow_dispatch
- Verifying the deployment (curl commands, CloudWatch logs)

---

## Medium Priority

### 4. Monitoring & Observability

- CloudWatch dashboard with key metrics (invocations, errors, duration, throttles)
- Structured log queries for debugging (request tracing by ID)
- SNS alerts for error threshold breaches
- X-Ray tracing for latency analysis

### 5. Model Versioning & A/B Testing

- Support multiple model versions deployed simultaneously
- Weighted routing between model versions via Lambda aliases
- Automatic rollback if new model version degrades metrics
- Model registry tracking version history and performance

### 6. Input/Output Schema Evolution

- Versioned API schemas (v1/predict, v2/predict)
- Backward-compatible response format changes
- Request/response validation with JSON Schema or Pydantic
- API documentation with OpenAPI/Swagger spec

---

## Low Priority

### 7. Performance Optimization

- Provisioned concurrency for eliminating cold starts
- Model quantization for faster inference
- Response caching for repeated inputs (ElastiCache or API Gateway cache)
- Lambda memory tuning via AWS Lambda Power Tuning

### 8. Security Hardening

- API key authentication via API Gateway usage plans
- Request rate limiting per client
- WAF rules for request filtering
- VPC deployment for private model access
- Secrets Manager for sensitive configuration

### 9. Multi-Model Support

- Generic model loading interface (not Iris-specific)
- Support for different ML frameworks (XGBoost, PyTorch, TensorFlow Lite)
- Model selection via request parameter or URL path
- Separate model artifacts per deployment environment

### 10. Advanced CI/CD

- Canary deployments with automatic traffic shifting
- Performance regression tests in CI
- Automated load testing before production promotion
- Infrastructure drift detection
- Cost monitoring and budget alerts

---

## Technical Debt

| Item | File | Description |
|------|------|-------------|
| Hardcoded class names | predictor.py | Should come from model metadata dynamically |
| Spanish/English mix | Multiple files | Standardize to one language in docstrings |
| No retry logic | handler.py | Model loading failures should retry with backoff |
| No health check | handler.py | GET /health should verify model is loaded |
| No request ID propagation | handler.py | Pass request_id through all log entries |
| No warm-up mechanism | lambda_function.py | Provisioned concurrency or scheduled pings |

---

## Completed (for reference)

- [x] Project scaffolding with Poetry
- [x] Data processing pipeline (DataProcessor)
- [x] Model training with cross-validation (ModelTrainer)
- [x] Model evaluation metrics (ModelEvaluator)
- [x] Model serialization with SHA256 integrity (ModelSerializer)
- [x] Input validation with range warnings (InputValidator)
- [x] Structured JSON logging (StructuredLogger)
- [x] Custom exception hierarchy
- [x] Lambda handler with cold start optimization (LambdaHandler)
- [x] Inference predictor with probabilities (Predictor)
- [x] CI pipeline: lint, type check, format, test, security scan
- [x] CD pipeline: build, upload to S3, deploy Lambda, smoke test
- [x] Terraform infrastructure: Lambda, API Gateway, S3, IAM, CloudWatch
- [x] Unit tests + property-based tests with Hypothesis
- [x] 80% minimum coverage enforcement
