"""
CXI File Writer Ray Actor for stateful file writing.

This module provides a Ray actor that maintains state across batches,
buffers events, and writes CXI files with proper formatting.
"""

import ray
import h5py
import numpy as np
import logging
from pathlib import Path
from datetime import datetime


@ray.remote
class CXIFileWriterActor:
    """
    Stateful Ray actor for CXI file writing.

    Responsibilities:
    - Initialize CheetahConverter once (expensive operation)
    - Buffer events across batches
    - Filter events by minimum peak count
    - Convert coordinates to Cheetah format
    - Write CXI files when buffer is full
    """

    def __init__(
        self,
        output_dir: str,
        geom_file: str,
        buffer_size: int = 100,
        min_num_peak: int = 10,
        max_num_peak: int = 2048,
        file_prefix: str = "peaknet_cxi",
        crystfel_mode: bool = False,
        save_segmentation_maps: bool = False,
        save_logit_maps: bool = False
    ):
        """
        Initialize CXI file writer with CheetahConverter.

        Args:
            output_dir: Directory for output CXI files
            geom_file: CrystFEL geometry file for coordinate conversion
            buffer_size: Number of events to buffer before writing
            min_num_peak: Minimum peaks required to save event
            max_num_peak: Maximum peaks per event (CXI array size)
            file_prefix: Prefix for CXI filenames
            crystfel_mode: Enable strict CrystFEL mode (requires geom_file,
                          adds LCLS datasets for downstream compatibility)
            save_segmentation_maps: Save segmentation maps to CXI (debug mode)
            save_logit_maps: Save logit maps to CXI (debug mode)
        """
        logging.basicConfig(level=logging.INFO)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.min_num_peak = min_num_peak
        self.max_num_peak = max_num_peak
        self.file_prefix = file_prefix
        self.crystfel_mode = crystfel_mode
        self.save_segmentation_maps = save_segmentation_maps
        self.save_logit_maps = save_logit_maps

        # Initialize CheetahConverter (expensive - done once per actor)
        self.cheetah_converter = None
        self.use_cheetah = geom_file is not None and geom_file != ""

        if self.use_cheetah:
            try:
                from crystfel_stream_parser.joblib_engine import StreamParser
                from crystfel_stream_parser.cheetah_converter import CheetahConverter

                logging.info(f"Initializing CheetahConverter from {geom_file}")
                geom_block = StreamParser(geom_file).parse(
                    num_cpus=1,
                    returns_stream_dict=True
                )[0].get('GEOM_BLOCK')
                self.cheetah_converter = CheetahConverter(geom_block)
                logging.info("CheetahConverter initialized successfully")

                if self.crystfel_mode:
                    logging.info("CrystFEL mode enabled - will include LCLS datasets")
            except Exception as e:
                error_msg = f"Failed to initialize CheetahConverter: {e}"
                if self.crystfel_mode:
                    # In CrystFEL mode, CheetahConverter is required
                    logging.error(error_msg)
                    raise RuntimeError(f"CrystFEL mode requires valid geometry file. {error_msg}")
                else:
                    # In default mode, gracefully fall back
                    logging.warning(error_msg)
                    logging.warning("Will proceed without coordinate conversion")
                    self.use_cheetah = False

        # State management
        self.buffer = []
        self.chunk_id = 0
        self.total_events_written = 0
        self.total_events_filtered = 0

        logging.info(f"CXIFileWriterActor initialized: output_dir={output_dir}")

    def submit_processed_batch(self, images, peaks_list, metadata_list, seg_maps_list=None, logit_maps_list=None):
        """
        Non-blocking submission of processed batch.

        Args:
            images: Image data for events (may be ObjectRef or numpy array)
            peaks_list: List of peak positions per event (may be ObjectRef)
            metadata_list: Metadata for each event
            seg_maps_list: Optional list of segmentation maps per event (debug mode)
            logit_maps_list: Optional list of logit maps per event (debug mode), shape (num_classes, C, H, W)
        """
        # Dereference if ObjectRef
        if isinstance(peaks_list, ray.ObjectRef):
            peaks_list = ray.get(peaks_list)
        if isinstance(images, ray.ObjectRef):
            images = ray.get(images)
        if isinstance(seg_maps_list, ray.ObjectRef):
            seg_maps_list = ray.get(seg_maps_list)
        if isinstance(logit_maps_list, ray.ObjectRef):
            logit_maps_list = ray.get(logit_maps_list)

        # Process each event in batch
        # Handle seg_maps_list and logit_maps_list being None
        if seg_maps_list is None:
            seg_maps_list = [None] * len(images)
        if logit_maps_list is None:
            logit_maps_list = [None] * len(images)

        for img, peaks, metadata, seg_map, logit_map in zip(images, peaks_list, metadata_list, seg_maps_list, logit_maps_list):
            # Filter by peak count
            if len(peaks) < self.min_num_peak:
                self.total_events_filtered += 1
                continue

            # Convert to Cheetah coordinates if converter available
            if self.use_cheetah and self.cheetah_converter is not None:
                try:
                    # Assemble multi-panel image: (C, H, W) → (H_assembled, W)
                    # Use reduces_geom=True to match coordinate conversion (16 module-level panels)
                    cheetah_image = self.cheetah_converter.convert_to_cheetah_img(img, reduces_geom=True)
                    # Transform peak coordinates to assembled detector space
                    cheetah_peaks = self.cheetah_converter.convert_to_cheetah_coords(peaks)

                    # Apply same transformation to segmentation map (if present)
                    if seg_map is not None:
                        cheetah_seg_map = self.cheetah_converter.convert_to_cheetah_img(seg_map, reduces_geom=True)
                    else:
                        cheetah_seg_map = None

                    # Apply same transformation to logit maps (if present)
                    # logit_map shape: (num_classes, C, H, W) - convert each class separately
                    if logit_map is not None:
                        cheetah_logit_maps = []
                        for class_idx in range(logit_map.shape[0]):
                            class_logit = logit_map[class_idx]  # (C, H, W)
                            cheetah_class_logit = self.cheetah_converter.convert_to_cheetah_img(class_logit, reduces_geom=True)
                            cheetah_logit_maps.append(cheetah_class_logit)
                        cheetah_logit_map = np.stack(cheetah_logit_maps, axis=0)  # (num_classes, H_assembled, W)
                    else:
                        cheetah_logit_map = None
                except Exception as e:
                    logging.warning(f"CheetahConverter failed: {e}, using manual assembly")
                    # Fallback: manually stack panels vertically
                    if len(img.shape) == 3:  # (C, H, W)
                        cheetah_image = img.reshape(-1, img.shape[-1])  # Stack vertically
                    else:
                        cheetah_image = img
                    cheetah_peaks = peaks

                    # Apply same fallback to segmentation map
                    if seg_map is not None and len(seg_map.shape) == 3:
                        cheetah_seg_map = seg_map.reshape(-1, seg_map.shape[-1])
                    else:
                        cheetah_seg_map = seg_map

                    # Apply same fallback to logit maps
                    if logit_map is not None and len(logit_map.shape) == 4:  # (num_classes, C, H, W)
                        cheetah_logit_maps = []
                        for class_idx in range(logit_map.shape[0]):
                            class_logit = logit_map[class_idx]  # (C, H, W)
                            cheetah_logit_maps.append(class_logit.reshape(-1, class_logit.shape[-1]))  # (C*H, W)
                        cheetah_logit_map = np.stack(cheetah_logit_maps, axis=0)  # (num_classes, C*H, W)
                    else:
                        cheetah_logit_map = logit_map
            else:
                # No CheetahConverter: manually assemble
                if len(img.shape) == 3:  # (C, H, W)
                    cheetah_image = img.reshape(-1, img.shape[-1])  # (C*H, W)
                else:
                    cheetah_image = img
                cheetah_peaks = peaks

                # Apply same manual assembly to segmentation map
                if seg_map is not None and len(seg_map.shape) == 3:
                    cheetah_seg_map = seg_map.reshape(-1, seg_map.shape[-1])  # (C*H, W)
                else:
                    cheetah_seg_map = seg_map

                # Apply same manual assembly to logit maps
                if logit_map is not None and len(logit_map.shape) == 4:  # (num_classes, C, H, W)
                    cheetah_logit_maps = []
                    for class_idx in range(logit_map.shape[0]):
                        class_logit = logit_map[class_idx]  # (C, H, W)
                        cheetah_logit_maps.append(class_logit.reshape(-1, class_logit.shape[-1]))  # (C*H, W)
                    cheetah_logit_map = np.stack(cheetah_logit_maps, axis=0)  # (num_classes, C*H, W)
                else:
                    cheetah_logit_map = logit_map

            # Add to buffer
            self.buffer.append({
                'image': cheetah_image,
                'peaks': cheetah_peaks,
                'metadata': metadata,
                'seg_map': cheetah_seg_map,  # Now assembled to match image dimensions
                'logit_map': cheetah_logit_map  # (num_classes, H_assembled, W) or None
            })

        # Flush if buffer full
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer_to_cxi()

    def _flush_buffer_to_cxi(self):
        """Write buffered events to CXI file."""
        if not self.buffer:
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.file_prefix}_{timestamp}_chunk{self.chunk_id:04d}.cxi"
        filepath = self.output_dir / filename

        try:
            with h5py.File(filepath, 'w') as f:
                num_events = len(self.buffer)

                # Get shape from first event (should be 2D assembled: H_assembled × W)
                image_shape = self.buffer[0]['image'].shape
                logging.debug(f"Writing CXI with image shape per event: {image_shape}")

                # Create datasets
                f.create_dataset(
                    '/entry_1/data_1/data',
                    (num_events, *image_shape),
                    dtype='float32'
                )
                f.create_dataset(
                    '/entry_1/result_1/peakSegPosRaw',
                    (num_events, self.max_num_peak),
                    dtype='float32',
                    fillvalue=-1
                )
                f.create_dataset(
                    '/entry_1/result_1/peakXPosRaw',
                    (num_events, self.max_num_peak),
                    dtype='float32',
                    fillvalue=-1
                )
                f.create_dataset(
                    '/entry_1/result_1/peakYPosRaw',
                    (num_events, self.max_num_peak),
                    dtype='float32',
                    fillvalue=-1
                )
                f.create_dataset(
                    '/entry_1/result_1/nPeaks',
                    (num_events,),
                    dtype='int'
                )

                # Debug mode: Add segmentation_map dataset
                if self.save_segmentation_maps:
                    # Get seg_map shape from first event (should be H_assembled, W - same as detector image)
                    first_seg_map = next((evt['seg_map'] for evt in self.buffer if evt.get('seg_map') is not None), None)
                    if first_seg_map is not None:
                        seg_map_shape = first_seg_map.shape  # (H_assembled, W) - matches detector image
                        f.create_dataset(
                            '/entry_1/result_1/segmentation_map',
                            (num_events, *seg_map_shape),
                            dtype='uint8',
                            compression='gzip',
                            compression_opts=4
                        )
                        logging.debug(f"Created segmentation_map dataset: ({num_events}, {seg_map_shape}) - matches detector image dimensions")

                # Debug mode: Add logit_map datasets (one per class)
                if self.save_logit_maps:
                    # Get logit_map shape from first event (should be num_classes, H_assembled, W)
                    first_logit_map = next((evt['logit_map'] for evt in self.buffer if evt.get('logit_map') is not None), None)
                    if first_logit_map is not None:
                        num_classes = first_logit_map.shape[0]  # Should be 2 for PeakNet
                        logit_map_shape = first_logit_map.shape[1:]  # (H_assembled, W)

                        # Create separate datasets for each class
                        for class_idx in range(num_classes):
                            f.create_dataset(
                                f'/entry_1/result_1/logit_map_class{class_idx}',
                                (num_events, *logit_map_shape),
                                dtype='float32',
                                compression='gzip',
                                compression_opts=4
                            )
                        logging.debug(f"Created {num_classes} logit_map datasets: ({num_events}, {logit_map_shape}) - matches detector image dimensions")

                # CrystFEL mode: Add peakTotalIntensity dataset
                # Note: This is a placeholder - was never real data in old pipeline
                if self.crystfel_mode:
                    f.create_dataset(
                        '/entry_1/result_1/peakTotalIntensity',
                        (num_events, self.max_num_peak),
                        dtype='float32',
                        fillvalue=0.0
                    )

                # Extract photon energies and timestamps from metadata
                photon_energies = []
                timestamps = []
                for evt in self.buffer:
                    meta = evt['metadata'] if isinstance(evt['metadata'], dict) else {}
                    photon_energies.append(meta.get('photon_energy', 0.0))

                    # Handle timestamp - could be scalar or array
                    ts = meta.get('timestamp', 0.0)
                    if isinstance(ts, np.ndarray):
                        # Take first element if array
                        timestamps.append(float(ts.flatten()[0]) if ts.size > 0 else 0.0)
                    else:
                        timestamps.append(float(ts))

                # Create LCLS metadata datasets
                f.create_dataset(
                    '/LCLS/photon_energy_eV',
                    data=np.array(photon_energies, dtype='float32')
                )

                # Add timestamp dataset if any non-zero timestamps exist
                # Convert to array first to safely check
                timestamps_array = np.array(timestamps, dtype='float32')
                if np.any(timestamps_array != 0.0):
                    f.create_dataset(
                        '/LCLS/detector_1/timestamp',
                        data=timestamps_array
                    )

                # CrystFEL mode or fallback: Add EncoderValue dataset
                if self.crystfel_mode:
                    # CrystFEL mode: Extract encoder values from metadata
                    encoder_values = [
                        evt['metadata'].get('encoder_value', 0.0) if isinstance(evt['metadata'], dict) else 0.0
                        for evt in self.buffer
                    ]
                    f.create_dataset(
                        '/LCLS/detector_1/EncoderValue',
                        data=np.array(encoder_values, dtype='float32')
                    )
                elif not np.any(timestamps_array != 0.0):
                    # For backward compatibility with old data (no timestamps), create EncoderValue placeholder
                    f.create_dataset(
                        '/LCLS/detector_1/EncoderValue',
                        (num_events,),
                        dtype='float32',
                        fillvalue=0.0
                    )

                # Write events
                for event_idx, evt in enumerate(self.buffer):
                    # Write image
                    f['/entry_1/data_1/data'][event_idx] = evt['image']

                    # Write segmentation map (debug mode)
                    if self.save_segmentation_maps and evt.get('seg_map') is not None:
                        f['/entry_1/result_1/segmentation_map'][event_idx] = evt['seg_map']

                    # Write logit maps (debug mode)
                    if self.save_logit_maps and evt.get('logit_map') is not None:
                        logit_map = evt['logit_map']  # (num_classes, H_assembled, W)
                        for class_idx in range(logit_map.shape[0]):
                            f[f'/entry_1/result_1/logit_map_class{class_idx}'][event_idx] = logit_map[class_idx]

                    # Write peak count
                    num_peaks = min(len(evt['peaks']), self.max_num_peak)
                    f['/entry_1/result_1/nPeaks'][event_idx] = num_peaks

                    # Write peak positions
                    for peak_idx, peak in enumerate(evt['peaks']):
                        if peak_idx >= self.max_num_peak:
                            break

                        # Handle different peak formats [p, y, x] or [seg, y, x]
                        if len(peak) >= 3:
                            seg, row, col = peak[0], peak[1], peak[2]
                        elif len(peak) == 2:
                            seg, row, col = 0, peak[0], peak[1]
                        else:
                            continue

                        f['/entry_1/result_1/peakSegPosRaw'][event_idx, peak_idx] = seg
                        f['/entry_1/result_1/peakYPosRaw'][event_idx, peak_idx] = row
                        f['/entry_1/result_1/peakXPosRaw'][event_idx, peak_idx] = col

                # Add file-level metadata
                f.attrs['creation_time'] = datetime.now().isoformat()
                f.attrs['num_events'] = num_events
                f.attrs['min_num_peak'] = self.min_num_peak

            # Update statistics
            self.total_events_written += num_events
            self.chunk_id += 1

            file_size_mb = filepath.stat().st_size / (1024**2)
            logging.info(
                f"Wrote CXI chunk {self.chunk_id}: {num_events} events, "
                f"{file_size_mb:.2f} MB → {filepath}"
            )

        except Exception as e:
            logging.error(f"Failed to write CXI file {filepath}: {e}")
            import traceback
            logging.error(traceback.format_exc())

        finally:
            # Clear buffer
            self.buffer.clear()

    def flush_final(self):
        """Flush any remaining events in buffer and return statistics."""
        self._flush_buffer_to_cxi()
        return {
            'total_events_written': self.total_events_written,
            'total_events_filtered': self.total_events_filtered,
            'chunks_written': self.chunk_id
        }

    def get_statistics(self):
        """Return current statistics."""
        return {
            'total_events_written': self.total_events_written,
            'total_events_filtered': self.total_events_filtered,
            'buffer_size': len(self.buffer),
            'chunks_written': self.chunk_id
        }
