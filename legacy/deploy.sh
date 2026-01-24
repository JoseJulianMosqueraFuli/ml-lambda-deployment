#!/bin/bash
set -e

echo "Training model..."
python train_model.py

echo "Creating deployment package..."
mkdir -p package
pip install -r requirements.txt -t package/

cp model.pkl package/
cp lambda_function.py package/

cd package
zip -r ../lambda_deployment.zip . -q
cd ..

echo "Deployment package created: lambda_deployment.zip"
echo "Size: $(du -h lambda_deployment.zip | cut -f1)"
