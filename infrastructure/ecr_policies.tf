# ECR policies

resource "aws_ecr_repository_policy" "lambda_repo_policy_a" {
  repository = "final_repo_lambda_a"

  policy = jsonencode({
    Version = "2008-10-17"
    Statement = [
      {
        Sid = "AllowLambdaPull",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ],
        Condition = {
          StringLike = {
            "aws:sourceArn" = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:*"
          }
        }
      }
    ]
  })
}

resource "aws_ecr_repository_policy" "lambda_repo_policy_b" {
  repository = "final_repo_lambda_b"

  policy = jsonencode({
    Version = "2008-10-17"
    Statement = [
      {
        Sid = "AllowLambdaPull",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ],
        Condition = {
          StringLike = {
            "aws:sourceArn" = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:*"
          }
        }
      }
    ]
  })
}

# Account ID
data "aws_caller_identity" "current" {}
