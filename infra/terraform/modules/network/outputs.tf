output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.this.name
}

output "network_self_link" {
  description = "VPC network self link"
  value       = google_compute_network.this.self_link
}

output "subnetwork_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.this.name
}

output "subnetwork_self_link" {
  description = "Subnet self link"
  value       = google_compute_subnetwork.this.self_link
}
