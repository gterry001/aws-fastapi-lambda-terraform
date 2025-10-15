
variable "region" {
  default = "eu-north-1"
}

variable "fastapi_image" {
  description = "Fastapi handler image"
  default = "559886226802.dkr.ecr.eu-north-1.amazonaws.com/fastapi-image:latest"
}

variable "worker_image" {
  description = "Analyzer image"
  default = "559886226802.dkr.ecr.eu-north-1.amazonaws.com/analyzer-image:latest"
}

variable "allowed_origin" {
  default = "https://sticmediatech.com"
}
