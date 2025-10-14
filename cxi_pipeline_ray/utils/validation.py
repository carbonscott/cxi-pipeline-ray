"""
Configuration consistency validation utilities.

This module provides functions for validating that pipeline and writer
configurations have consistent Ray and queue settings.
"""

from typing import Dict, Any


def validate_consistency(pipeline_config: Dict[str, Any], writer_config: Dict[str, Any]):
    """
    Validate that pipeline and writer configs have consistent Ray/queue settings.

    This function checks that critical coordination settings match between
    the pipeline and writer configurations to ensure they can communicate.

    Args:
        pipeline_config: Pipeline configuration dictionary
        writer_config: Writer configuration dictionary

    Raises:
        ValueError: If critical settings don't match

    Example:
        >>> pipeline_cfg = yaml.safe_load(open('pipeline.yaml'))
        >>> writer_cfg = yaml.safe_load(open('writer.yaml'))
        >>> validate_consistency(pipeline_cfg, writer_cfg)
    """
    errors = []

    # Check Ray namespace
    pipeline_namespace = pipeline_config.get('ray', {}).get('namespace')
    writer_namespace = writer_config.get('ray', {}).get('namespace')

    if pipeline_namespace != writer_namespace:
        errors.append(
            f"Ray namespace mismatch: pipeline='{pipeline_namespace}' vs writer='{writer_namespace}'"
        )

    # Check Q2 queue name
    pipeline_q2_name = pipeline_config.get('runtime', {}).get('queue_names', {}).get('output_queue')
    writer_queue_name = writer_config.get('queue', {}).get('name')

    if pipeline_q2_name != writer_queue_name:
        errors.append(
            f"Queue name mismatch: pipeline output_queue='{pipeline_q2_name}' vs writer queue.name='{writer_queue_name}'"
        )

    # Check Q2 shards
    pipeline_shards = pipeline_config.get('runtime', {}).get('queue_num_shards')
    writer_shards = writer_config.get('queue', {}).get('num_shards')

    if pipeline_shards != writer_shards:
        errors.append(
            f"Queue shards mismatch: pipeline queue_num_shards={pipeline_shards} vs writer queue.num_shards={writer_shards}"
        )

    # Check if Q2 output is enabled in pipeline
    output_enabled = pipeline_config.get('runtime', {}).get('enable_output_queue')
    if not output_enabled:
        errors.append(
            "Pipeline does not have 'enable_output_queue: true' - Q2 queue will not be created"
        )

    if errors:
        error_msg = "Configuration consistency validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


def print_validation_result(pipeline_config: Dict[str, Any], writer_config: Dict[str, Any]):
    """
    Print validation result with colorized output.

    Args:
        pipeline_config: Pipeline configuration dictionary
        writer_config: Writer configuration dictionary
    """
    try:
        validate_consistency(pipeline_config, writer_config)

        # If we get here, validation passed
        print("✓ Configuration validation passed")
        print(f"  ✓ Ray namespace matches: {writer_config['ray']['namespace']}")
        print(f"  ✓ Q2 queue name matches: {writer_config['queue']['name']}")
        print(f"  ✓ Q2 num shards matches: {writer_config['queue']['num_shards']}")
        print(f"  ✓ Pipeline has output queue enabled")

    except ValueError as e:
        print("✗ Configuration validation failed")
        print(str(e))
        raise
