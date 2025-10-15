# Worker Lambda

This Lambda runs the background worker responsible for processing messages from the SQS queue.  
It operates asynchronously, decoupled from the API, and handles longer-running or resource-intensive tasks triggered by the FastAPI Lambda.

The function is packaged as a **Docker container image**, built with **AWS CodeBuild** and stored in **Amazon ECR**.  
Terraform then deploys it as an AWS Lambda function with an **event source mapping** that automatically triggers it whenever a new message appears in the SQS queue.

---

### How it fits in the architecture

- **FastAPI Lambda** sends jobs or data to **SQS**.  
- **This worker Lambda** consumes those messages and performs the actual processing (e.g. file handling, data transformation, external API calls).  
- Processed outputs can be uploaded to **S3** or another downstream service.  

---

### Build and deployment

1. **CodeBuild** clones the repository and builds the Docker image using the `Dockerfile` in this directory.  
2. The image is **tagged and pushed to ECR** (`final_repo_lambda_b`).  
3. **Terraform** deploys the Lambda function using the ECR image and automatically links it to the SQS queue using an event source mapping.  
4. Environment variables (e.g. `BUCKET_NAME`) are configured by Terraform at deployment time.

---

### Requirements

- A configured **ECR repository** for the worker image.  
- A **CodeBuild project** with permissions to build Docker images and push to ECR.  
- Terraform with AWS credentials and access to your infrastructure.  
- The `boto3` Python package inside the image for S3 or SQS interactions.

---

This worker runs fully serverless and scales automatically based on the number of SQS messages. It can be easily extended to handle different types of background tasks.
