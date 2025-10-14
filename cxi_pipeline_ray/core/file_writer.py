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
        file_prefix: str = "peaknet_cxi"
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
        """
        logging.basicConfig(level=logging.INFO)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.min_num_peak = min_num_peak
        self.max_num_peak = max_num_peak
        self.file_prefix = file_prefix

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
            except Exception as e:
                logging.warning(f"Failed to initialize CheetahConverter: {e}")
                logging.warning("Will proceed without coordinate conversion")
                self.use_cheetah = False

        # State management
        self.buffer = []
        self.chunk_id = 0
        self.total_events_written = 0
        self.total_events_filtered = 0

        logging.info(f"CXIFileWriterActor initialized: output_dir={output_dir}")

    def submit_processed_batch(self, images, peaks_list, metadata_list):
        """
        Non-blocking submission of processed batch.

        Args:
            images: Image data for events (may be ObjectRef or numpy array)
            peaks_list: List of peak positions per event (may be ObjectRef)
            metadata_list: Metadata for each event
        """
        # Dereference if ObjectRef
        if isinstance(peaks_list, ray.ObjectRef):
            peaks_list = ray.get(peaks_list)
        if isinstance(images, ray.ObjectRef):
            images = ray.get(images)

        # Process each event in batch
        for img, peaks, metadata in zip(images, peaks_list, metadata_list):
            # Filter by peak count
            if len(peaks) < self.min_num_peak:
                self.total_events_filtered += 1
                continue

            # Convert to Cheetah coordinates if converter available
            if self.use_cheetah and self.cheetah_converter is not None:
                try:
                    cheetah_peaks = self.cheetah_converter.convert_to_cheetah_coords(peaks)
                    cheetah_image = self.cheetah_converter.convert_to_cheetah_img(img)
                except Exception as e:
                    logging.warning(f"CheetahConverter failed: {e}, using original coordinates")
                    cheetah_peaks = peaks
                    cheetah_image = img
            else:
                cheetah_peaks = peaks
                cheetah_image = img

            # Add to buffer
            self.buffer.append({
                'image': cheetah_image,
                'peaks': cheetah_peaks,
                'metadata': metadata
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

                # Get shape from first event
                image_shape = self.buffer[0]['image'].shape

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

                # Extract photon energies from metadata
                photon_energies = [
                    evt['metadata'].get('photon_energy', 0.0) if isinstance(evt['metadata'], dict) else 0.0
                    for evt in self.buffer
                ]
                f.create_dataset(
                    '/LCLS/photon_energy_eV',
                    data=np.array(photon_energies, dtype='float32')
                )

                # Write events
                for event_idx, evt in enumerate(self.buffer):
                    # Write image
                    f['/entry_1/data_1/data'][event_idx] = evt['image']

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
