# syntax=docker/dockerfile:1

FROM python:3.11-slim AS dependency-builder

WORKDIR /build

# Export only the locked production dependency set before entering Lambda's image.
RUN pip install --no-cache-dir "poetry==2.3.2" "poetry-plugin-export>=1.9,<2.0"
COPY pyproject.toml poetry.lock ./
RUN poetry export \
    --without dev \
    --format requirements.txt \
    --output /tmp/requirements.txt \
    --without-hashes

# Python 3.12 Lambda image runs Amazon Linux 2023, which supports the locked
# manylinux wheels for scikit-learn/numpy/scipy; the 3.11 image (Amazon Linux 2)
# would force building scikit-learn from source.
FROM public.ecr.aws/lambda/python:3.12

COPY --from=dependency-builder /tmp/requirements.txt /tmp/requirements.txt
RUN pip install \
    --no-cache-dir \
    --no-compile \
    --requirement /tmp/requirements.txt \
    --target "${LAMBDA_TASK_ROOT}"

COPY src/ml_lambda "${LAMBDA_TASK_ROOT}/ml_lambda"
COPY artifacts/model.joblib "${LAMBDA_TASK_ROOT}/artifacts/model.joblib"

CMD ["ml_lambda.lambda_function.lambda_handler"]
