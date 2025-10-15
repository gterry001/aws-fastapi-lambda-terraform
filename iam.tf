##########################################
# 🔹 IAM Role for Lambda (v2)
##########################################

data "aws_iam_policy_document" "lambda_assume_v2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_role_v2" {
  name               = "lambda-execution-role-tfv2"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_v2.json
}

##########################################
# 🔹 Basic Execution Role
##########################################
resource "aws_iam_role_policy_attachment" "lambda_basic_v2" {
  role       = aws_iam_role.lambda_role_v2.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

##########################################
# 🔹 Extended Policy: S3, SQS, and ECR Access
##########################################
resource "aws_iam_policy" "lambda_extra_v2" {
  name = "lambda-extra-access-tfv2"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # --- S3 Access ---
      {
        Effect   = "Allow"
        Action   = ["s3:*"]
        Resource = [
          aws_s3_bucket.data_v2.arn,
          "${aws_s3_bucket.data_v2.arn}/*"
        ]
      },

      # --- SQS Access ---
      {
        Effect   = "Allow"
        Action   = ["sqs:*"]
        Resource = [aws_sqs_queue.jobs_v2.arn]
      },

      # --- ECR Access (for Lambda container images) ---
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = "*"
      }
    ]
  })
}

##########################################
# 🔹 Attach Custom Policy
##########################################
resource "aws_iam_role_policy_attachment" "lambda_extra_attach_v2" {
  role       = aws_iam_role.lambda_role_v2.name
  policy_arn = aws_iam_policy.lambda_extra_v2.arn
}
