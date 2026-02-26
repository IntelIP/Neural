variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "metric_name" {
  description = "Log-based metric name"
  type        = string
  default     = "neural_runner_error_count"
}

variable "instance_name" {
  description = "Runner VM instance name used in log filters"
  type        = string
}

variable "enable_alert_policy" {
  description = "Whether to create an alert policy for runtime errors"
  type        = bool
  default     = false
}

variable "notification_channels" {
  description = "Notification channel IDs used by alert policy"
  type        = list(string)
  default     = []
}
