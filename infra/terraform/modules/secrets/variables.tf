variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "secret_ids" {
  description = "Set of secret IDs to provision"
  type        = set(string)
}

variable "runner_service_account_email" {
  description = "Runner service account email granted secret access"
  type        = string
}
