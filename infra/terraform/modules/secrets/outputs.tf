output "secret_ids" {
  description = "Secret IDs created by this module"
  value       = [for secret in google_secret_manager_secret.this : secret.secret_id]
}

output "secret_resource_ids" {
  description = "Secret resource IDs"
  value       = [for secret in google_secret_manager_secret.this : secret.id]
}
