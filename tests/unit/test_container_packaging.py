"""Tests for the AWS Lambda container packaging configuration."""

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_dockerfile_builds_a_lambda_image_with_the_project_handler():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()

    assert "FROM public.ecr.aws/lambda/python:3.12" in dockerfile
    assert 'CMD ["ml_lambda.lambda_function.lambda_handler"]' in dockerfile
    assert "artifacts/model.joblib" in dockerfile
    assert "tests/" not in dockerfile


def test_dockerignore_excludes_development_files_but_keeps_the_model():
    dockerignore = (ROOT / ".dockerignore").read_text()

    assert "tests/" in dockerignore
    assert ".venv/" in dockerignore
    assert "*.pyc" in dockerignore
    assert "artifacts/*" in dockerignore
    assert "!artifacts/model.joblib" in dockerignore
