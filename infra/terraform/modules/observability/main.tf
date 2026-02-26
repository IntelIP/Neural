resource "google_logging_metric" "runner_errors" {
  project = var.project_id
  name    = var.metric_name

  filter = <<-EOT
    resource.type="gce_instance"
    resource.labels.instance_id:*
    labels."compute.googleapis.com/resource_name"="${var.instance_name}"
    severity>=ERROR
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    labels {
      key         = "instance_name"
      value_type  = "STRING"
      description = "Runner instance name"
    }
  }

  label_extractors = {
    instance_name = "EXTRACT(labels.\"compute.googleapis.com/resource_name\")"
  }
}

resource "google_monitoring_alert_policy" "runner_error_alert" {
  count   = var.enable_alert_policy ? 1 : 0
  project = var.project_id

  display_name = "Neural Runner Error Alert"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Runner emits error logs"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.runner_errors.name}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "60s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_DELTA"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels
}
