output "runner_instance_name" {
  value       = module.runner.instance_name
  description = "Compute instance name"
}

output "runner_external_ip" {
  value       = module.runner.instance_external_ip
  description = "External IP of runner instance"
}

output "runner_service_account_email" {
  value       = module.runner.service_account_email
  description = "Runner service account email"
}

output "secret_ids" {
  value       = module.secrets.secret_ids
  description = "Provisioned Secret Manager IDs"
}
