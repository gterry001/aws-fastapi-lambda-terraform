output "api_endpoint_v2" {
  description = "API Gateway endpoint (v2)"
  value       = aws_apigatewayv2_api.http_api_v2.api_endpoint
}

output "s3_bucket_v2" {
  description = "Nombre del bucket S3 creado (v2)"
  value       = aws_s3_bucket.data_v2.bucket
}

output "sqs_queue_url_v2" {
  description = "URL de la cola SQS (v2)"
  value       = aws_sqs_queue.jobs_v2.id
}
