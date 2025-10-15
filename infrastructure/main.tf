##########################################
# 🔹 S3 Bucket
##########################################
resource "aws_s3_bucket" "data_v2" {
  bucket        = "fastapi-bucket-project-tfv2"
  force_destroy = true
}

##########################################
# 🔹 SQS Queue
##########################################
resource "aws_sqs_queue" "jobs_v2" {
  name                       = "fastapi-sqs-tfv2"
  visibility_timeout_seconds = 900
}

##########################################
# 🔹 Lambda A (FastAPI)
##########################################
resource "aws_lambda_function" "fastapi_v2" {
  function_name = "fastapi-handler-tfv2"
  package_type  = "Image"
  image_uri     = var.fastapi_image
  role          = aws_iam_role.lambda_role_v2.arn
  timeout       = 900
  memory_size   = 2048

  environment {
    variables = {
      BUCKET_NAME = aws_s3_bucket.data_v2.bucket
      QUEUE_URL   = aws_sqs_queue.jobs_v2.id
    }
  }
}

##########################################
# 🔹 Lambda B (Worker)
##########################################
resource "aws_lambda_function" "worker_v2" {
  function_name = "worker-handler-tfv2"
  package_type  = "Image"
  image_uri     = var.worker_image
  role          = aws_iam_role.lambda_role_v2.arn
  timeout       = 900
  memory_size   = 3008

  ephemeral_storage {
    size = 4096
  }

  architectures = ["x86_64"]

  environment {
    variables = {
      BUCKET_NAME = aws_s3_bucket.data_v2.bucket
    }
  }
}

# Conectar SQS -> Worker Lambda
resource "aws_lambda_event_source_mapping" "sqs_to_lambda_v2" {
  event_source_arn = aws_sqs_queue.jobs_v2.arn
  function_name    = aws_lambda_function.worker_v2.arn
  batch_size       = 1
  enabled          = true
}

##########################################
# 🔹 API Gateway HTTP v2
##########################################
resource "aws_apigatewayv2_api" "http_api_v2" {
  name          = "fastapi-http-tfv2"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = [var.allowed_origin]
    allow_methods = ["*"]
    allow_headers = ["*"]
  }
}

resource "aws_apigatewayv2_integration" "fastapi_integration_v2" {
  api_id                 = aws_apigatewayv2_api.http_api_v2.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.fastapi_v2.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "proxy_route_v2" {
  api_id    = aws_apigatewayv2_api.http_api_v2.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.fastapi_integration_v2.id}"
}

resource "aws_apigatewayv2_stage" "default_v2" {
  api_id      = aws_apigatewayv2_api.http_api_v2.id
  name        = "$default"
  auto_deploy = true
}

# Permitir que API Gateway invoque la Lambda FastAPI
resource "aws_lambda_permission" "api_gw_invoke_v2" {
  statement_id  = "AllowAPIGatewayInvokeV2"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.fastapi_v2.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api_v2.execution_arn}/*/*"
}
