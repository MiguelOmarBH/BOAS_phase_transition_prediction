"""Read calibrated BOAS channels from ordinary 16-bit EDF recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _text(raw: bytes) -> str:
    return raw.decode("latin-1").strip()


def _number(raw: bytes, cast: type[int] | type[float]) -> int | float:
    value = _text(raw)
    if not value:
        raise ValueError("Missing numeric field in EDF header")
    return cast(value)


@dataclass(frozen=True)
class EDFSignal:
    label: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: int

    @property
    def scale(self) -> float:
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)

    @property
    def offset(self) -> float:
        return self.physical_min - self.digital_min * self.scale


@dataclass(frozen=True)
class EDFHeader:
    path: Path
    header_bytes: int
    n_records: int
    record_duration: float
    signals: tuple[EDFSignal, ...]

    @property
    def samples_per_record(self) -> int:
        return sum(signal.samples_per_record for signal in self.signals)

    def sampling_frequency(self, label: str) -> float:
        signal = next((s for s in self.signals if s.label == label), None)
        if signal is None:
            raise KeyError(f"Signal {label!r} not found")
        return signal.samples_per_record / self.record_duration


def read_header(path: str | Path) -> EDFHeader:
    """Read and validate the fixed and per-signal sections of an EDF header."""
    edf_path = Path(path)
    with edf_path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Incomplete EDF header: {edf_path}")
        header_bytes = int(_number(fixed[184:192], int))
        n_records = int(_number(fixed[236:244], int))
        record_duration = float(_number(fixed[244:252], float))
        n_signals = int(_number(fixed[252:256], int))
        signal_header = handle.read(header_bytes - 256)

    if len(signal_header) != header_bytes - 256:
        raise ValueError(f"Incomplete EDF signal header: {edf_path}")
    if n_records <= 0 or record_duration <= 0 or n_signals <= 0:
        raise ValueError(f"Unsupported EDF dimensions in {edf_path}")

    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields: list[list[bytes]] = []
    cursor = 0
    for width in widths:
        block = signal_header[cursor : cursor + width * n_signals]
        fields.append([block[i * width : (i + 1) * width] for i in range(n_signals)])
        cursor += width * n_signals

    signals = tuple(
        EDFSignal(
            label=_text(fields[0][i]),
            physical_dimension=_text(fields[2][i]),
            physical_min=float(_number(fields[3][i], float)),
            physical_max=float(_number(fields[4][i], float)),
            digital_min=int(_number(fields[5][i], int)),
            digital_max=int(_number(fields[6][i], int)),
            samples_per_record=int(_number(fields[8][i], int)),
        )
        for i in range(n_signals)
    )
    expected_size = header_bytes + 2 * n_records * sum(
        signal.samples_per_record for signal in signals
    )
    actual_size = edf_path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(
            f"Truncated EDF file {edf_path}: expected at least {expected_size} bytes, "
            f"found {actual_size}"
        )
    return EDFHeader(
        path=edf_path,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration=record_duration,
        signals=signals,
    )


def read_signals(
    path: str | Path, labels: tuple[str, ...] = ("HB_1", "HB_2")
) -> tuple[np.ndarray, float]:
    """Return calibrated signals as ``(channels, samples)`` and sampling rate."""
    header = read_header(path)
    index_by_label = {signal.label: i for i, signal in enumerate(header.signals)}
    missing = [label for label in labels if label not in index_by_label]
    if missing:
        raise KeyError(f"Missing EDF channels {missing} in {path}")

    selected = [header.signals[index_by_label[label]] for label in labels]
    frequencies = {signal.samples_per_record / header.record_duration for signal in selected}
    if len(frequencies) != 1:
        raise ValueError(f"Selected channels do not share a sampling rate in {path}")
    sampling_frequency = frequencies.pop()

    data = np.memmap(
        header.path,
        dtype="<i2",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, header.samples_per_record),
    )
    starts = np.cumsum([0] + [s.samples_per_record for s in header.signals[:-1]])
    output = []
    for label in labels:
        signal_index = index_by_label[label]
        signal = header.signals[signal_index]
        start = int(starts[signal_index])
        stop = start + signal.samples_per_record
        digital = np.asarray(data[:, start:stop]).reshape(-1).astype(np.float64)
        output.append(digital * signal.scale + signal.offset)
    return np.vstack(output), float(sampling_frequency)


def _read_channel(header: EDFHeader, label: str) -> tuple[np.ndarray, float]:
    labels = [item.label for item in header.signals]
    if label not in labels:
        raise KeyError(f"Missing EDF channel {label!r} in {header.path}")
    selected_index = labels.index(label)
    selected = header.signals[selected_index]
    data = np.memmap(
        header.path,
        dtype="<i2",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, header.samples_per_record),
    )
    starts = np.cumsum([0] + [item.samples_per_record for item in header.signals[:-1]])
    start = int(starts[selected_index])
    stop = start + selected.samples_per_record
    digital = np.asarray(data[:, start:stop]).reshape(-1).astype(np.float64)
    physical = digital * selected.scale + selected.offset
    return physical, selected.samples_per_record / header.record_duration


def _channel_groups(header: EDFHeader) -> dict[str, list[str]]:
    labels = [item.label for item in header.signals]
    lower = {label: label.lower().replace("-", "_").replace(" ", "_") for label in labels}
    dimensions = {item.label: item.physical_dimension.lower() for item in header.signals}
    eeg = [label for label in labels if lower[label] in {"hb_1", "hb_2"}]
    ppg = [
        label
        for label in labels
        if any(token in lower[label] for token in ("ppg", "pleth", "pulse"))
    ]
    accel = [
        label
        for label in labels
        if (
            any(token in lower[label] for token in ("acc", "acceler"))
            or ("imu" in lower[label] and dimensions[label] in {"g", "m/s2", "m/s^2"})
        )
        and "accuracy" not in lower[label]
    ]
    gyro = [
        label
        for label in labels
        if any(token in lower[label] for token in ("gyr", "gyro"))
        or ("imu" in lower[label] and "deg/s" in dimensions[label])
    ]
    return {"eeg": eeg, "ppg": ppg, "accel": accel, "gyro": gyro}
