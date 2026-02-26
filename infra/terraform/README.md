# Neural Terraform Baseline (GCP)

This directory contains a reference Terraform baseline for running Neural bots on GCP.
It is designed as a starting point for teams that want reproducible infrastructure around
Docker-based execution.

## Modules

- `modules/network`: VPC, subnet, and baseline firewall rules.
- `modules/runner_vm`: Compute Engine runner VM with Docker-friendly bootstrap hooks.
- `modules/secrets`: Secret Manager secret containers and runner service account access grants.
- `modules/observability`: Log-based metric and optional alert policy for runtime errors.

## Module contracts

### `network`
Inputs:
- `project_id` (string)
- `region` (string)
- `network_name` (string)
- `subnet_name` (string)
- `subnet_cidr` (string)
- `enable_private_google_access` (bool, default `true`)
- `allow_ssh_cidrs` (list(string), default `[]`)
- `internal_tcp_ports` (list(string), default `[]`)
- `internal_udp_ports` (list(string), default `[]`)

Outputs:
- `network_name`
- `network_self_link`
- `subnetwork_name`
- `subnetwork_self_link`

### `runner_vm`
Inputs:
- `project_id` (string)
- `zone` (string)
- `instance_name` (string)
- `machine_type` (string, default `e2-standard-2`)
- `network_self_link` (string)
- `subnetwork_self_link` (string)
- `create_service_account` (bool, default `true`)
- `service_account_id` (string, default `neural-runner`)
- `service_account_email` (string, required when `create_service_account=false`)
- `service_account_scopes` (list(string), default logging/monitoring/container-pull scopes; add Secret Manager scope if needed)
- `assign_public_ip` (bool, default `true`)
- `startup_script` (string, optional)
- `metadata` (map(string), default `{}`)
- `tags` (list(string), default `["neural-runner"]`)
- `boot_image` (string, default Debian 12 family image)
- `boot_disk_size_gb` (number, default `50`)
- `boot_disk_type` (string, default `pd-balanced`)

Outputs:
- `instance_name`
- `instance_self_link`
- `instance_external_ip`
- `service_account_email`

### `secrets`
Inputs:
- `project_id` (string)
- `secret_ids` (set(string))
- `runner_service_account_email` (string)

Outputs:
- `secret_ids`
- `secret_resource_ids`

### `observability`
Inputs:
- `project_id` (string)
- `metric_name` (string, default `neural_runner_error_count`)
- `instance_name` (string)
- `enable_alert_policy` (bool, default `false`)
- `notification_channels` (list(string), default `[]`)

Outputs:
- `log_metric_name`
- `log_metric_type`
- `alert_policy_id`

## Usage

Wire these modules from environment stacks (added in PR-3) and run:

```bash
terraform init
terraform fmt -check -recursive
terraform validate
```

This baseline intentionally avoids provider-specific app deployment logic so teams can swap
the runtime bootstrap (Docker, private providers, or orchestrators) without rewriting core IaC.
