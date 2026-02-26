variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the subnet"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "subnet_name" {
  description = "Subnet name"
  type        = string
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
}

variable "enable_private_google_access" {
  description = "Whether Private Google Access is enabled on the subnet"
  type        = bool
  default     = true
}

variable "allow_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH into tagged instances"
  type        = list(string)
  default     = []
}
