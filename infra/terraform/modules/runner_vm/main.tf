locals {
  resolved_service_account_email = var.create_service_account ? google_service_account.runner[0].email : var.service_account_email
}

resource "google_service_account" "runner" {
  count = var.create_service_account ? 1 : 0

  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "Neural runner service account"
}

resource "google_compute_instance" "runner" {
  project      = var.project_id
  zone         = var.zone
  name         = var.instance_name
  machine_type = var.machine_type
  tags         = var.tags

  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  network_interface {
    network    = var.network_self_link
    subnetwork = var.subnetwork_self_link

    dynamic "access_config" {
      for_each = var.assign_public_ip ? [1] : []
      content {}
    }
  }

  metadata = var.metadata

  metadata_startup_script = var.startup_script

  service_account {
    email  = local.resolved_service_account_email
    scopes = var.service_account_scopes
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  lifecycle {
    precondition {
      condition     = var.create_service_account || var.service_account_email != null
      error_message = "service_account_email must be provided when create_service_account=false."
    }
  }
}
