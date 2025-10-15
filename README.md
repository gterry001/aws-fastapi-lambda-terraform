# FastAPI + AWS Lambda + Terraform Deployment

This project showcases how to deploy a FastAPI backend and a Worker Lambda fully serverless on AWS using Terraform as Infrastructure as Code.


## Architecture

- FastAPI Lambda: serves HTTP requests through API Gateway.
- Worker Lambda: consumes messages from SQS and processes data.
- S3 Bucket: stores uploaded data.
- SQS Queue: job trigger between Lambdas.
- ECR + CodeBuild: builds and hosts container images for Lambda.


## Tech Stack

- AWS Lambda (Container Images)
- AWS ECR, S3, SQS, API Gateway
- Terraform
- FastAPI (Python 3.11)
- AWS CodeBuild

## How to Deploy

1. Clone this repository
   git clone https://github.com/gterry001/aws-fastapi-lambda-terraform.git
   cd aws-fastapi-lambda-terraform/infrastructure

3. Initialize Terraform:
   terraform init

4. Apply:
   terraform apply -auto-approve

5. Output:
   terraform output

## Highlights

1. Clean modular Terraform setup.
2. Secure IAM roles with least privilege.
3. ECR repositories automatically configured for Lambda.
4. CodeBuild pipelines for Docker image build and push.
5. Example Python code for FastAPI and worker processing.

## Author

Guillermo Terry de Loredo
Cloud Engineer | AWS | Terraform | Python | FastAPI
