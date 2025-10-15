# FastAPI + AWS Lambda + Terraform Deployment

This project showcases how to deploy a **FastAPI backend** and a **Worker Lambda** fully serverless on AWS using **Terraform** as Infrastructure as Code.

---

## Architecture

- **FastAPI Lambda**: serves HTTP requests through API Gateway.
- **Worker Lambda**: consumes messages from SQS and processes data.
- **S3 Bucket**: stores uploaded data.
- **SQS Queue**: job trigger between Lambdas.
- **ECR + CodeBuild**: builds and hosts container images for Lambda.

![architecture](./docs/architecture-diagram.png)

---

##  Tech Stack

- **AWS Lambda (Container Images)**
- **AWS ECR, S3, SQS, API Gateway**
- **Terraform**
- **FastAPI (Python 3.11)**
- **AWS CodeBuild**

---

## How to Deploy

1.  Clone this repository:
   ```bash
   git clone https://github.com/<tu-usuario>/aws-fastapi-lambda-terraform.git
   cd aws-fastapi-lambda-terraform/infrastructure

2.  Initialize Terraform
  terraform init

3.  Apply => It is necessary to change the variables so it refers to your actual lambda images and website
terraform apply -auto-approve

4.  Output
  terraform output

## Folder structure

| Folder            | Description                          |
| ----------------- | ------------------------------------ |
| `infrastructure/` | Terraform configuration              |
| `lambda_fastapi/` | FastAPI handler Lambda (API Gateway) |
| `lambda_worker/`  | Worker Lambda (SQS consumer)         |
| `docs/`           | Diagrams and documentation           |


