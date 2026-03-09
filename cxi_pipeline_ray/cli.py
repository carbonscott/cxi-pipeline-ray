#!/usr/bin/env python3
"""
Command-line interface for CXI pipeline writer.

This module provides the main entry point for the cxi-writer command.
"""

import argparse
import sys
import logging
import yaml
import ray

from .utils.config import load_config, merge_config_with_overrides
from .utils.validation import print_validation_result
from .core.coordinator import run_sync_pipeline
from .core.file_writer import CXIFileWriterActor
from .version import __version__


def setup_logging(level: str):
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_configs_mode(args):
    """
    Run configuration validation mode.

    Args:
        args: Parsed command-line arguments
    """
    print("=== Configuration Validation Mode ===")
    print(f"Pipeline config: {args.pipeline_config}")
    print(f"Writer config: {args.writer_config}")
    print()

    # Load both configs
    try:
        with open(args.pipeline_config, 'r') as f:
            pipeline_config = yaml.safe_load(f)

        with open(args.writer_config, 'r') as f:
            writer_config = yaml.safe_load(f)

        # Validate consistency
        print_validation_result(pipeline_config, writer_config)

    except Exception as e:
        print(f"\nValidation failed: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for cxi-writer command.

    Modes:
    1. Normal mode: Run the CXI writer pipeline
    2. Validation mode: Validate config consistency
    """
    parser = argparse.ArgumentParser(
        description="CXI Pipeline Writer - Synchronous post-processing for PeakNet inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal operation
  cxi-writer --config cxi_writer.yaml

  # With CLI overrides
  cxi-writer --config cxi_writer.yaml --batches-per-file 20 --output-dir /custom/path

  # Validate config consistency
  cxi-writer --validate-config --pipeline-config pipeline.yaml --writer-config writer.yaml
        """
    )

    parser.add_argument('--version', action='version', version=f'cxi-pipeline-ray {__version__}')

    # Config file or validation mode
    parser.add_argument(
        '--config',
        type=str,
        help='Path to writer configuration YAML file'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration consistency between pipeline and writer'
    )

    parser.add_argument(
        '--pipeline-config',
        type=str,
        help='Path to pipeline configuration (for validation mode)'
    )

    parser.add_argument(
        '--writer-config',
        type=str,
        help='Path to writer configuration (for validation mode)'
    )

    # CLI overrides
    parser.add_argument(
        '--batches-per-file',
        type=int,
        help='Write CXI file every N batches (default: 10)'
    )

    parser.add_argument(
        '--save-segmentation-maps',
        action='store_true',
        help='Save segmentation maps to /entry_1/result_1/segmentation_map (debug mode)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )

    parser.add_argument(
        '--file-prefix',
        type=str,
        help='Override CXI file prefix'
    )

    parser.add_argument(
        '--geom-file',
        type=str,
        help='Geometry file for CrystFEL coordinate conversion. When provided, enables CrystFEL mode with additional LCLS datasets for downstream compatibility.'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Handle validation mode
    if args.validate_config:
        if not args.pipeline_config or not args.writer_config:
            parser.error("--validate-config requires --pipeline-config and --writer-config")
        validate_configs_mode(args)
        sys.exit(0)

    # Normal mode: require config file
    if not args.config:
        parser.error("--config is required (or use --validate-config mode)")

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Merge CLI overrides
    config = merge_config_with_overrides(config, args)

    logger.info("=== CXI Pipeline Writer ===")
    logger.info(f"Version: {__version__}")
    logger.info(f"Ray namespace: {config['ray']['namespace']}")
    logger.info(f"Queue: {config['queue']['name']} ({config['queue']['num_shards']} shards)")
    logger.info(f"Output dir: {config['output']['output_dir']}")
    logger.info(f"Batches per file: {config['output']['batches_per_file']}")

    # Connect to Ray
    logger.info("Connecting to Ray cluster...")
    try:
        ray.init(
            namespace=config['ray']['namespace'],
            ignore_reinit_error=True
        )
        logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")
    except Exception as e:
        logger.error(f"Failed to connect to Ray: {e}")
        sys.exit(1)

    # Connect to Q2 queue
    logger.info(f"Connecting to Q2 queue: {config['queue']['name']}")
    try:
        # Import here to avoid dependency issues if peaknet-pipeline-ray not installed
        from peaknet_pipeline_ray.utils.queue import ShardedQueueManager

        q2_manager = ShardedQueueManager(
            base_name=config['queue']['name'],
            num_shards=config['queue']['num_shards'],
            maxsize_per_shard=config['queue'].get('maxsize_per_shard', 1600)
        )
        logger.info("Successfully connected to Q2 queue")
    except ImportError as e:
        logger.error("Failed to import ShardedQueueManager from peaknet-pipeline-ray")
        logger.error("Make sure peaknet-pipeline-ray is installed: pip install peaknet-pipeline-ray")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to connect to Q2 queue: {e}")
        sys.exit(1)

    # Create file writer actor
    logger.info("Creating CXI file writer actor...")
    geom_file = config.get('geometry', {}).get('geom_file')
    file_writer = CXIFileWriterActor.remote(
        output_dir=config['output']['output_dir'],
        geom_file=geom_file,
        buffer_size=config['output']['buffer_size'],
        min_num_peak=config['peak_finding']['min_num_peak'],
        max_num_peak=config['peak_finding']['max_num_peak'],
        file_prefix=config['output']['file_prefix'],
        crystfel_mode=geom_file is not None,
        save_segmentation_maps=config['output'].get('save_segmentation_maps', False),
    )

    # Run pipeline
    logger.info("Starting synchronous post-processing pipeline...")
    try:
        run_sync_pipeline(
            q2_manager=q2_manager,
            file_writer=file_writer,
            batches_per_file=config['output']['batches_per_file'],
            save_segmentation_maps=config['output'].get('save_segmentation_maps', False),
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
