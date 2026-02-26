variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "stack_name" {
  description = "Prefix for resource naming"
  type        = string
  default     = "neural-dev"
}

variable "subnet_cidr" {
  description = "Subnet CIDR range"
  type        = string
  default     = "10.30.0.0/24"
}

variable "allow_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH"
  type        = list(string)
  default     = []
}

variable "machine_type" {
  description = "Runner machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "secret_ids" {
  description = "Secret IDs to provision in Secret Manager"
  type        = set(string)
  default = [
    "kalshi-api-key-id",
    "kalshi-private-key-pem",
  ]
}

variable "enable_alert_policy" {
  description = "Whether to create an alert policy"
  type        = bool
  default     = false
}

variable "notification_channels" {
  description = "Alert notification channel IDs"
  type        = list(string)
  default     = []
}

variable "startup_script" {
  description = "Startup script for docker bootstrap"
  type        = string
  default     = <<-EOT
    #!/usr/bin/env bash
    set -euo pipefail

    apt-get update
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker

    echo "Neural dev runner bootstrap complete" > /var/log/neural-bootstrap.log
  EOT
}
