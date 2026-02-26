resource "google_compute_network" "this" {
  project                 = var.project_id
  name                    = var.network_name
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"
}

resource "google_compute_subnetwork" "this" {
  project                  = var.project_id
  region                   = var.region
  name                     = var.subnet_name
  ip_cidr_range            = var.subnet_cidr
  network                  = google_compute_network.this.id
  private_ip_google_access = var.enable_private_google_access
}

resource "google_compute_firewall" "allow_internal" {
  project = var.project_id
  name    = "${var.network_name}-allow-internal"
  network = google_compute_network.this.name

  dynamic "allow" {
    for_each = length(var.internal_tcp_ports) > 0 ? [1] : []
    content {
      protocol = "tcp"
      ports    = var.internal_tcp_ports
    }
  }

  dynamic "allow" {
    for_each = length(var.internal_udp_ports) > 0 ? [1] : []
    content {
      protocol = "udp"
      ports    = var.internal_udp_ports
    }
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.subnet_cidr]
}

resource "google_compute_firewall" "allow_ssh" {
  count = length(var.allow_ssh_cidrs) > 0 ? 1 : 0

  project = var.project_id
  name    = "${var.network_name}-allow-ssh"
  network = google_compute_network.this.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allow_ssh_cidrs
}
