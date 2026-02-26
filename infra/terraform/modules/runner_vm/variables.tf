variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "zone" {
  description = "GCP zone for the runner VM"
  type        = string
}

variable "instance_name" {
  description = "Runner VM instance name"
  type        = string
}

variable "machine_type" {
  description = "Runner VM machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "network_self_link" {
  description = "Self link of the VPC network"
  type        = string
}

variable "subnetwork_self_link" {
  description = "Self link of the subnet"
  type        = string
}

variable "create_service_account" {
  description = "Whether to create a service account for the runner"
  type        = bool
  default     = true
}

variable "service_account_id" {
  description = "Service account ID when create_service_account=true"
  type        = string
  default     = "neural-runner"
}

variable "service_account_email" {
  description = "Existing service account email when create_service_account=false"
  type        = string
  default     = null
}

variable "service_account_scopes" {
  description = "OAuth scopes granted to the runner VM service account"
  type        = list(string)
  default = [
    "https://www.googleapis.com/auth/logging.write",
    "https://www.googleapis.com/auth/monitoring.write",
    "https://www.googleapis.com/auth/devstorage.read_only",
    "https://www.googleapis.com/auth/secretmanager",
  ]
}

variable "startup_script" {
  description = "Startup script for VM bootstrap"
  type        = string
  default     = <<-EOT
    #!/usr/bin/env bash
    set -euo pipefail

    apt-get update
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker
  EOT
}

variable "metadata" {
  description = "Metadata map for instance configuration"
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "Network tags assigned to the instance"
  type        = list(string)
  default     = ["neural-runner"]
}

variable "boot_image" {
  description = "Boot image for the runner VM"
  type        = string
  default     = "projects/debian-cloud/global/images/family/debian-12"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "boot_disk_type" {
  description = "Boot disk type"
  type        = string
  default     = "pd-balanced"
}
