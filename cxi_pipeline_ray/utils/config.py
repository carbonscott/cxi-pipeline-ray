"""
Configuration loading and validation utilities.

This module provides functions for loading YAML configuration files
and merging with command-line overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is not valid YAML
        ValueError: If required fields are missing
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required top-level sections
    required_sections = ['ray', 'queue', 'output']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    # Validate Ray config
    if 'namespace' not in config['ray']:
        raise ValueError("Missing 'ray.namespace' in config")

    # Validate queue config
    queue_required = ['name', 'num_shards']
    missing_queue = [f for f in queue_required if f not in config['queue']]
    if missing_queue:
        raise ValueError(f"Missing required queue config: {missing_queue}")

    # Validate output config
    if 'output_dir' not in config['output']:
        raise ValueError("Missing 'output.output_dir' in config")

    # Set defaults for optional fields
    if 'maxsize_per_shard' not in config['queue']:
        config['queue']['maxsize_per_shard'] = 1600

    if 'poll_timeout' not in config['queue']:
        config['queue']['poll_timeout'] = 0.01

    if 'processing' not in config:
        config['processing'] = {}

    if 'num_cpu_workers' not in config['processing']:
        config['processing']['num_cpu_workers'] = 16

    if 'max_pending_tasks' not in config['processing']:
        config['processing']['max_pending_tasks'] = 100

    if 'peak_finding' not in config:
        config['peak_finding'] = {}

    if 'min_num_peak' not in config['peak_finding']:
        config['peak_finding']['min_num_peak'] = 10

    if 'max_num_peak' not in config['peak_finding']:
        config['peak_finding']['max_num_peak'] = 2048

    if 'buffer_size' not in config['output']:
        config['output']['buffer_size'] = 100

    if 'file_prefix' not in config['output']:
        config['output']['file_prefix'] = "peaknet_cxi"

    if 'create_output_dir' not in config['output']:
        config['output']['create_output_dir'] = True

    return config


def merge_config_with_overrides(config: Dict[str, Any], cli_args) -> Dict[str, Any]:
    """
    Merge CLI argument overrides into configuration.

    Args:
        config: Base configuration dictionary
        cli_args: Parsed command-line arguments (argparse.Namespace)

    Returns:
        Updated configuration dictionary
    """
    # Override processing settings if provided
    if hasattr(cli_args, 'num_cpu_workers') and cli_args.num_cpu_workers is not None:
        config['processing']['num_cpu_workers'] = cli_args.num_cpu_workers

    if hasattr(cli_args, 'max_pending_tasks') and cli_args.max_pending_tasks is not None:
        config['processing']['max_pending_tasks'] = cli_args.max_pending_tasks

    # Override output settings if provided
    if hasattr(cli_args, 'output_dir') and cli_args.output_dir is not None:
        config['output']['output_dir'] = cli_args.output_dir

    if hasattr(cli_args, 'file_prefix') and cli_args.file_prefix is not None:
        config['output']['file_prefix'] = cli_args.file_prefix

    # Override geometry file if provided
    if hasattr(cli_args, 'geom_file') and cli_args.geom_file is not None:
        if 'geometry' not in config:
            config['geometry'] = {}
        config['geometry']['geom_file'] = cli_args.geom_file

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'ray.namespace')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {'ray': {'namespace': 'test'}}
        >>> get_config_value(config, 'ray.namespace')
        'test'
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
