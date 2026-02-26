output "log_metric_name" {
  description = "Name of the log-based metric"
  value       = google_logging_metric.runner_errors.name
}

output "log_metric_type" {
  description = "Fully qualified metric type"
  value       = "logging.googleapis.com/user/${google_logging_metric.runner_errors.name}"
}

output "alert_policy_id" {
  description = "Alert policy ID when enabled"
  value       = var.enable_alert_policy ? google_monitoring_alert_policy.runner_error_alert[0].id : null
}
