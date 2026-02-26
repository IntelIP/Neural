output "instance_name" {
  description = "Runner instance name"
  value       = google_compute_instance.runner.name
}

output "instance_self_link" {
  description = "Runner instance self link"
  value       = google_compute_instance.runner.self_link
}

output "instance_external_ip" {
  description = "External IP address for the runner VM"
  value       = length(google_compute_instance.runner.network_interface[0].access_config) > 0 ? google_compute_instance.runner.network_interface[0].access_config[0].nat_ip : null
}

output "service_account_email" {
  description = "Runner service account email"
  value       = local.resolved_service_account_email
}
