# cxi-pipeline-ray

Ray-based CPU post-processing pipeline for PeakNet inference results.

## Overview

`cxi-pipeline-ray` is a scalable, distributed post-processing pipeline that:
- Consumes PeakNet inference results from Ray queues (Q2)
- Performs CPU-based peak finding using scipy.ndimage
- Writes results to CXI files for crystallography analysis

This package implements a hybrid Ray architecture with:
- **Stateless Ray tasks** for parallel peak finding
- **Stateful Ray actor** for buffered CXI file writing
- **Ray best practices** for optimal throughput (backpressure control, batched operations, pipelining)

## Installation

### Development Installation

```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
pip install -e .
```

### Dependencies

Core dependencies (automatically installed):
- `ray>=2.0.0` - Distributed computing framework
- `numpy>=1.20.0` - Array operations
- `scipy>=1.7.0` - Peak finding algorithms
- `h5py>=3.0.0` - HDF5/CXI file writing
- `torch>=2.0.0` - Tensor operations
- `pyyaml>=6.0` - Configuration loading

External dependencies (must be installed separately):
- `peaknet-pipeline-ray` - For ShardedQueueManager
- `crystfel-stream-parser` - For CheetahConverter (optional)

## Quick Start

### 1. Create Configuration

Copy the example config and customize:

```bash
cp examples/configs/cxi_writer_default.yaml my_config.yaml
# Edit my_config.yaml to set:
#   - ray.namespace (must match pipeline)
#   - queue.name and queue.num_shards (must match pipeline)
#   - output.output_dir
#   - geometry.geom_file (optional)
```

### 2. Run the Writer

```bash
# Basic usage
cxi-writer --config my_config.yaml

# With CLI overrides
cxi-writer --config my_config.yaml \
  --num-cpu-workers 32 \
  --output-dir /custom/output

# Validate config consistency with pipeline
cxi-writer --validate-config \
  --pipeline-config path/to/pipeline_config.yaml \
  --writer-config my_config.yaml
```

### 3. Typical Workflow

```bash
# Terminal 1: Start Q2 writer (waits for data)
cxi-writer --config my_config.yaml

# Terminal 2: Run PeakNet inference pipeline
peaknet-pipeline --config peaknet-socket-profile-673m-with-output.yaml
```

## Configuration

See `examples/configs/cxi_writer_default.yaml` for a fully documented configuration file.

### Critical Settings (Must Match Pipeline)

| Setting | Pipeline Location | Writer Location | Why Critical |
|---------|------------------|-----------------|--------------|
| Ray namespace | `ray.namespace` | `ray.namespace` | Must be in same namespace |
| Q2 queue name | `runtime.queue_names.output_queue` | `queue.name` | Must consume from correct queue |
| Q2 num shards | `runtime.queue_num_shards` | `queue.num_shards` | Queue topology must match |

### Key Parameters

- `num_cpu_workers`: Parallel Ray tasks (default: 16)
  - Start with `num_cpu_cores // 4`
  - Increase if CPU utilization is low
  - Decrease if task overhead is high

- `max_pending_tasks`: Backpressure limit (default: 100)
  - Prevents OOM by limiting in-flight work
  - Lower = less memory, higher = better throughput

- `buffer_size`: Events per CXI file (default: 100)
  - Larger = fewer files, more memory

- `min_num_peak`: Minimum peaks to save event (default: 10)
  - Quality filter for events

## Architecture

The pipeline consists of three main components:

1. **Peak Finding (Ray Task)** - `cxi_pipeline_ray/core/peak_finding.py`
   - Stateless CPU-based peak finding
   - Converts logits → segmentation maps → peak positions
   - Uses scipy.ndimage for connected component labeling

2. **File Writer (Ray Actor)** - `cxi_pipeline_ray/core/file_writer.py`
   - Stateful CXI file writer
   - Maintains CheetahConverter for coordinate conversion
   - Buffers events and writes when buffer is full

3. **Coordinator** - `cxi_pipeline_ray/core/coordinator.py`
   - Main pipeline orchestration
   - Implements backpressure control
   - Batched ray.get() calls for optimal throughput
   - Pipelining for overlapping I/O and compute

For detailed architecture documentation, see:
- `PLAN-Q2-CXI-WRITER-v2.md` - Technical design details
- `PLAN-Q2-CXI-WRITER-ARCH.md` - Implementation architecture
- `RAY-BEST-PRACTICES-REVIEW.md` - Ray optimizations

## Performance Tuning

### Symptoms and Solutions

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Low CPU utilization (<50%) | Underutilized CPUs | Increase `num_cpu_workers` |
| High task scheduling overhead | Too many tiny tasks | Decrease `num_cpu_workers` |
| Q2 queue growing | Writer too slow | Increase `num_cpu_workers` |
| Memory usage growing | Too many pending tasks | Decrease `max_pending_tasks` |
| Too many small CXI files | Buffer flushing too often | Increase `buffer_size` |

### Monitoring

```bash
# Ray dashboard
ray dashboard

# Monitor writer logs
cxi-writer --config my_config.yaml --log-level DEBUG

# Monitor CXI output
watch -n 5 'ls -lh /output/dir/*.cxi'
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration test (requires running pipeline)
# See tests/test_integration.py for details
```

### Code Style

```bash
# Format code
black cxi_pipeline_ray/

# Lint
ruff check cxi_pipeline_ray/
```

## Troubleshooting

### Common Issues

**Ray namespace mismatch**
```
Error: Cannot find queue 'peaknet_q2' in namespace 'peaknet-pipeline'
```
Solution: Check `ray.namespace` matches in both pipeline and writer configs

**Queue shards mismatch**
```
Error: Queue 'peaknet_q2' has 4 shards, expected 8
```
Solution: Check `queue.num_shards` matches pipeline's `queue_num_shards`

**Geometry file not found**
```
FileNotFoundError: /path/to/detector.geom
```
Solution: Verify `geometry.geom_file` path or set to `null` to skip conversion

**OOM (Out of Memory)**
```
Ray ObjectStoreFullError: ...
```
Solution: Decrease `max_pending_tasks` or increase Ray object store size

## Citation

If you use this software in your research, please cite:

```
@software{cxi_pipeline_ray,
  title = {cxi-pipeline-ray: Ray-based CPU post-processing pipeline for PeakNet inference},
  author = {Yoon, Chun Hong},
  year = {2025},
  url = {https://github.com/...}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please contact:
- Chun Hong Yoon <cxyoon@slac.stanford.edu>
