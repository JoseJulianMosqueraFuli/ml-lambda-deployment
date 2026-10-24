# ML Lambda Deployment - Terraform Infrastructure
# Provisions: ECR repository, Lambda Function (container image), API Gateway, IAM roles

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state
  # backend "s3" {
  #   bucket = "ml-lambda-terraform-state"
  #   key    = "lambda/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ml-lambda-deployment"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  default     = "us-east-1"
  description = "AWS region for all resources"
}

variable "environment" {
  default     = "staging"
  description = "Deployment environment"
}

variable "lambda_function_name" {
  default     = "ml-iris-predictor"
  description = "Name of the Lambda function"
}

variable "image_tag" {
  default     = "latest"
  description = "Initial container image tag; later deployments update the image from GitHub Actions"
}

# ECR repository for the Lambda container image.
# The image (~240 MB of scikit-learn/scipy/numpy) does not fit comfortably in the
# 250 MB unzipped limit of ZIP packages; container images allow up to 10 GB.
resource "aws_ecr_repository" "predictor" {
  name                 = "${var.lambda_function_name}-${var.environment}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "predictor" {
  repository = aws_ecr_repository.predictor.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the 10 most recent images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.lambda_function_name}-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_role.name
}

# Lambda Function
resource "aws_lambda_function" "predictor" {
  function_name = "${var.lambda_function_name}-${var.environment}"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.predictor.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      ENVIRONMENT = var.environment
      LOG_LEVEL   = var.environment == "production" ? "INFO" : "DEBUG"
    }
  }

  lifecycle {
    ignore_changes = [image_uri, last_modified]
  }
}

# API Gateway (REST API)
resource "aws_apigatewayv2_api" "predictor_api" {
  name          = "${var.lambda_function_name}-api-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET", "OPTIONS"]
    allow_headers = ["Content-Type"]
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.predictor_api.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      path           = "$context.path"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      latency        = "$context.integrationLatency"
    })
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.predictor_api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.predictor.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.predictor_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.predictor_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predictor.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.predictor_api.execution_arn}/*/*"
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${aws_lambda_function.predictor.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.lambda_function_name}-${var.environment}"
  retention_in_days = 14
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.lambda_function_name}-${var.environment}-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Lambda function error rate too high"

  dimensions = {
    FunctionName = aws_lambda_function.predictor.function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  alarm_name          = "${var.lambda_function_name}-${var.environment}-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Average"
  threshold           = 10000
  alarm_description   = "Lambda function duration too high (>10s)"

  dimensions = {
    FunctionName = aws_lambda_function.predictor.function_name
  }
}

# Outputs
output "api_endpoint" {
  value       = aws_apigatewayv2_api.predictor_api.api_endpoint
  description = "API Gateway endpoint URL"
}

output "lambda_function_arn" {
  value       = aws_lambda_function.predictor.arn
  description = "Lambda function ARN"
}

output "lambda_function_name" {
  value       = aws_lambda_function.predictor.function_name
  description = "Lambda function name"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.predictor.repository_url
  description = "ECR repository for the Lambda container image"
}
