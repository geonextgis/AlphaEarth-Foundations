"""
Data loaders
- Only Sentinel-2 inputs (B2, B3, B4, B8, B11) at 10 m
- NPZ dataset option (from STAC chips)
- Simple collate that pads time to max within batch
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AEFDataset(Dataset):
    """
    Synthetic multi-temporal satellite dataset (Sentinel-2 only).

    This dataset generates random spatio-temporal samples for AlphaEarth Foundations that operate on Earth observation data.

    Each sample consists of:
        - A time series of image patches (Sentinel-2)
        - Corresponding timestamps
        - A valid period (target/summary window)
        - Optional text description
    """

    def __init__(
        self,
        num_samples: int = 1000,
        patch_size: int = 128,
        num_frames: int = 16,
    ):
        """
        Initialize the dataset.

        Args:
            num_samples (int): Number of samples in the dataset (dataset length).
            patch_size (int): Height and width of image patches (H = W).
            num_frames (int): Number of temporal frames per sample.
        """
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.input_source = ["sentinel2"]  # Only Sentinel-2 for example

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate a single synthetic training sample.

        Each sample includes:
            - Multi-temporal Sentinel-2 image sequence
            - Corresponding timestamps
            - A valid period (can extend beyond input support period)
            - Optional text description

        Args:
            idx (int): Sample index (not used for deterministic retrieval).

        Returns:
            Dict[str, Any]: A dictionary with the following structure:
                {
                    "source_data": {
                        "sentinel2": Tensor [T, H, W, C]
                    },
                    "timestamps": {
                        "sentinel2": Tensor [T]
                    },
                    "valid_period": (start_ms, end_ms),
                }
        """

        # Support period (input data range) - up to 1 year
        support_start_ms = 1577836800000.0  # 2020-01-01 in ms
        support_end_ms = support_start_ms + (365 * 24 * 3600 * 1000)  # 1 year later

        # Valid period (sumamry period) - can be different from support
        valid_start_ms = support_start_ms + np.random.uniform(
            0, 6 * 30 * 24 * 3600 * 1000
        )  # Random start within 6 months
        valid_end_ms = valid_start_ms + (365 * 24 * 3600 * 1000)  # 1 year summary

        # Generate timestamp (S2 only)
        timestamps = {
            "sentinel2": self._generate_timestamps(
                support_start_ms, support_end_ms, self.num_frames
            )
        }

        # Generate input source data (S2 only)
        input_data = {}
        for source in self.input_source:
            num_frames = len(timestamps[source])
            input_data[source] = self._generate_source_data(
                source, num_frames, is_input=True
            )
        source_data = input_data

        item = {
            "source_data": source_data,
            "timestamps": timestamps,
            "valid_period": (valid_start_ms, valid_end_ms),
        }

        return item

    def _generate_timestamps(
        self, start_ms: float, end_ms: float, num_frames: int
    ) -> torch.Tensor:
        """Generate random timestamps within period."""
        timestamps = np.random.uniform(start_ms, end_ms, num_frames)
        timestamps = np.sort(timestamps)
        return torch.tensor(timestamps, dtype=torch.float32)

    def _generate_source_data(
        self,
        source: str,
        num_frames: int,
        is_input: bool = True,
        num_channels_from_datasource: int = 5,
    ) -> torch.Tensor:
        """Generate synthetic data for a source."""
        return torch.rand(
            num_frames, self.patch_size, self.patch_size, num_channels_from_datasource
        )


def create_aef_dataloader(
    num_samples: int = 1000,
    batch_size: int = 4,
    num_workers: int = 2,
    num_frames: int = 16,
    patch_size: int = 128,
    return_text: bool = False,
):
    """Create AlphaEarth Foundations dataloader with proper collation."""

    dataset = AEFDataset(
        num_samples=num_samples,
        patch_size=patch_size,
        num_frames=num_frames,
    )

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function handling variable-length sequences."""

        # Collate S2
        collated_sources = {}
        source = "sentinel2"
        source_tensors = []
        for sample in batch:
            source_tensors.append(sample["source_data"][source])
        max_time = max(t.shape[0] for t in source_tensors)
        padded_tensors = []
        for tensor in source_tensors:
            if tensor.shape[0] < max_time:
                padding = torch.zeros(max_time - tensor.shape[0], *tensor.shape[1:])
                tensor = torch.cat([tensor, padding], dim=0)
            padded_tensors.append(tensor)
        collated_sources[source] = torch.stack(padded_tensors)

        # Collate timestamps for S2
        collated_timestamps = {}
        timestamps_list = []
        for sample in batch:
            timestamps_list.append(sample["timestamps"]["sentinel2"])
        max_time_ts = max(len(t) for t in timestamps_list)
        padded_timestamps = []
        for ts in timestamps_list:
            if len(ts) < max_time_ts:
                last_ts = ts[-1] if len(ts) > 0 else torch.tensor(0.0)
                padding = torch.full((max_time_ts - len(ts),), float(last_ts))
                ts = torch.cat([ts, padding])
            padded_timestamps.append(ts)
        collated_timestamps["sentinel2"] = torch.stack(padded_timestamps)

        batch_dict = {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "valid_periods": [sample["valid_period"] for sample in batch],
        }

        return batch_dict

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_samples,
        collate_fn=collate_fn,
        pin_memory=True,
    )


class AEFNPZDataset(Dataset):
    """
    Dataset that reads multi-modal Earth observation data from pre-extracted .npz files.

    Data Sources and Specifications:
        Input sources (available at inference):
            - sentinel2: (T, H, W, 5) - Bands: B2, B3, B4, B8, B11 (blue, green, red, NIR, SWIR)
            - sentinel1: (T, H, W, 5) - Polarization: VV, VH, HH, HV, angle
            - landsat8: (T, H, W, 7) - Bands: B2-B6, B8, B10 (optical + thermal)
            - landsat9: (T, H, W, 7) - Same as Landsat8

        Target/auxiliary sources (may be used for training targets or auxiliary features):
            - gedi: (T, H, W, 101) - Relative height percentiles [0-100]
            - era5: (T, H, W, 12) - Weather features: precip, temp, dewpoint, pressure (sum/min/max)
            - glo30: (1, H, W, 1) - Static DEM (Digital Elevation Model)
            - palsar2: (T, H, W, 3) - PALSAR-2 data: HH, HV, linearized intensity
            - grace: (T, H, W, 1) - Gravity Recovery & Climate Experiment: water equivalent thickness
            - nlcd: (1, H, W, 1) - Static landcover classification

    Attributes:
        root (Path): Path to directory containing .npz files
        files (List[Path]): Sorted list of .npz file paths
        sources (List[str]): List of data sources to load from each file
        INPUT_SOURCES (set): Class constant defining input-only sources
        ALL_SOURCES (set): Class constant defining all available sources
    """

    # Define input sources (used at inference) vs target-only sources
    INPUT_SOURCES = {"sentinel2", "sentinel1", "landsat8", "landsat9"}
    ALL_SOURCES = {
        "sentinel2",
        "sentinel1",
        "landsat8",
        "landsat9",
        "gedi",
        "era5",
        "glo30",
        "palsar2",
        "grace",
        "nlcd",
    }

    def __init__(self, root: str, sources: list = None):
        """
        Initialize the dataset.

        Args:
            root (str): Path to directory containing pre-extracted .npz files.
                       Each .npz file should contain arrays for various data sources.
            sources (list, optional): List of data sources to load. If None, all available
                                     sources in ALL_SOURCES will be loaded. If specified,
                                     only sources present in both the file and this list
                                     will be included.

        Raises:
            FileNotFoundError: If no .npz files are found in the root directory.
        """
        self.root = Path(root)
        self.files = sorted([p for p in self.root.glob("*.npz")])
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root}")
        # Which sources to load (default: all available)
        self.sources = sources if sources else list(self.ALL_SOURCES)

    def __len__(self) -> int:
        """Return the total number of .npz files (samples) in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve (0 to len(self)-1).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "source_data" (Dict[str, Tensor]): Loaded data arrays for each source.
                    Keys are source names, values are PyTorch tensors with shape (T, H, W, C).
                - "timestamps" (Dict[str, Tensor]): Temporal information for each source.
                    Keys are source names, values are 1D tensors with timestamps in milliseconds.
                    For static sources (glo30, nlcd), a default timestamp is used.
                - "valid_period" (Tuple[float, float]): Time window (start_ms, end_ms) centered
                    on the median timestamp across all sources, with ±180 day margins.

        Notes:
            - Timestamps are expected to be stored as "ts_<source_name>" in the .npz file.
            - Static sources (glo30, nlcd) without timestamps use a default timestamp
              of 1577836800000.0 (2020-01-01 in milliseconds since epoch).
            - The valid_period represents a ±180 day window around the temporal center
              of the data, useful for defining target prediction windows or data validity.
        """
        path = self.files[idx]
        data = np.load(path)

        src = {}
        ts = {}

        for source in self.sources:
            if source in data:
                arr = data[source].astype(np.float32)
                src[source] = torch.from_numpy(arr)
                ts_key = f"ts_{source}"
                if ts_key in data:
                    ts[source] = torch.from_numpy(data[ts_key].astype(np.float32))
                elif source in {"glo30", "nlcd"}:
                    # Static sources - use default timestamp
                    ts[source] = torch.tensor([1577836800000.0], dtype=torch.float32)

        # Compute valid period from timestamps
        if ts:
            med = float(np.median(np.concatenate([t.numpy() for t in ts.values()])))
        else:
            med = 1577836800000.0
        vp = (med - 15552000000.0, med + 15552000000.0)  # +/- 180 days

        return {"source_data": src, "timestamps": ts, "valid_period": vp}


def create_aef_dataloader_from_npz(
    root: str, batch_size: int = 2, num_workers: int = 4
) -> DataLoader:
    dataset = AEFNPZDataset(root)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated_sources = {}
        collated_timestamps = {}
        s = "sentinel2"
        tensors = [sample["source_data"][s] for sample in batch]
        ts_list = [sample["timestamps"][s] for sample in batch]
        max_t = max(t.shape[0] for t in tensors)
        padded = []
        padded_ts = []
        for x, t in zip(tensors, ts_list):
            if x.shape[0] < max_t:
                pad_x = torch.zeros(max_t - x.shape[0], *x.shape[1:])
                x = torch.cat([x, pad_x], dim=0)
                last = t[-1] if t.numel() else torch.tensor(0.0)
                pad_t = last.repeat(max_t - t.shape[0])
                t = torch.cat([t, pad_t], dim=0)
            padded.append(x)
            padded_ts.append(t)
        collated_sources[s] = torch.stack(padded)
        collated_timestamps[s] = torch.stack(padded_ts)

        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "valid_periods": [sample["valid_period"] for sample in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
