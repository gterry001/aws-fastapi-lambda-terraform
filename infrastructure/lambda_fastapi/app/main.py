import os, json
from uuid import uuid4
import boto3
from fastapi import FastAPI
from mangum import Mangum
from botocore.exceptions import ClientError
from fastapi.responses import Response, JSONResponse

BUCKET_NAME = os.environ["BUCKET_NAME"]
QUEUE_URL   = os.environ["QUEUE_URL"]

s3 = boto3.client("s3")
sqs = boto3.client("sqs")

app = FastAPI(title="Job Orchestrator (Lambda A)")

@app.get("/")
def root():
    return {"message": "API is running (Lambda A)"}

@app.get("/start-job")
def start_job():
    job_id = str(uuid4())

    # Estado inicial en S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"jobs/{job_id}.json",
        Body=json.dumps({"status": "running"}).encode("utf-8"),
        ContentType="application/json",
    )

    # Mensaje a SQS para que Lambda B lance el analisis
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({"type": "Analysis", "job_id": job_id})
    )

    return {"job_id": job_id, "status": "running"}

@app.get("/get-result/{job_id}")
def get_result(job_id: str):
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"jobs/{job_id}.json")
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"error": "Job not found"}
        raise
@app.get("/start-execution/{job_id}")
def start_order_execution(job_id: str):
    if not job_id:
        raise ValueError("No job_id")
    print("Iniciando ordenes con id:" + job_id)
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"executions/{job_id}.json",
            Body=json.dumps({"status": "running"}).encode("utf-8"),
            ContentType= "application/json",
        )
    except ClientError as e:
        print(e)
    # Mensaje a SQS para que Lambda B lance la ejecución
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({"type": "Orders", "job_id": job_id})
    )
    return {"job_id": job_id, "status": "running"}

@app.get("/get-execution/{job_id}")
def get_order_execution(job_id: str):
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"executions/{job_id}.json")
        body = obj["Body"].read().decode("utf-8")
        return Response(content=body, media_type="application/json")
    except ClientError as e:
         if e.response["Error"]["Code"] == "NoSuchKey":
            return {"error": "Error placing orders"}
         raise
        
handler = Mangum(app)
