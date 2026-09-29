"""Vulture whitelist for callback arguments required by external protocols."""

def _required_callback_parameters(signum, exc_type, exc_val, exc_tb, tb):
    """Keep names required by signal and context-manager protocols."""
    return signum, exc_type, exc_val, exc_tb, tb
