terraform {
  required_version = ">= 1.5.0, < 2.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

module "network" {
  source = "../../modules/network"

  project_id   = var.project_id
  region       = var.region
  network_name = "${var.stack_name}-vpc"
  subnet_name  = "${var.stack_name}-subnet"
  subnet_cidr  = var.subnet_cidr

  allow_ssh_cidrs = var.allow_ssh_cidrs
}

module "runner" {
  source = "../../modules/runner_vm"

  project_id           = var.project_id
  zone                 = var.zone
  instance_name        = "${var.stack_name}-runner"
  machine_type         = var.machine_type
  network_self_link    = module.network.network_self_link
  subnetwork_self_link = module.network.subnetwork_self_link

  startup_script = var.startup_script
  tags           = ["neural-runner", "env-dev"]
}

module "secrets" {
  source = "../../modules/secrets"

  project_id                   = var.project_id
  secret_ids                   = var.secret_ids
  runner_service_account_email = module.runner.service_account_email
}

module "observability" {
  source = "../../modules/observability"

  project_id            = var.project_id
  instance_name         = module.runner.instance_name
  enable_alert_policy   = var.enable_alert_policy
  notification_channels = var.notification_channels
}
