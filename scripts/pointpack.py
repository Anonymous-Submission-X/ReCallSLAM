"""Compact, browser-readable RGB point packs from immutable benchmark products."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def read_ply(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header incomplete: {path}")
            lines.append(line)
            if line.strip() == b"end_header":
                break
        offset = stream.tell()
    header = b"".join(lines).decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Unsupported PLY format: {path}")
    count = int(next(line.split()[-1] for line in header.splitlines() if line.startswith("element vertex ")))
    for prop in ("property float x", "property float y", "property float z",
                 "property uchar red", "property uchar green", "property uchar blue"):
        if prop not in header:
            raise ValueError(f"Missing {prop}: {path}")
    if path.stat().st_size != offset + count * POINT_DTYPE.itemsize:
        raise ValueError(f"Unexpected PLY layout or size: {path}")
    return np.memmap(path, dtype=POINT_DTYPE, mode="r", offset=offset, shape=(count,))


def read_tum(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 8 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid trajectory: {path}")
    return data


def rotation_from_quat(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def estimate_up(trajectory: np.ndarray) -> np.ndarray:
    ups = np.array([-rotation_from_quat(row[4:8])[:, 1]
                    for row in trajectory[::max(1, len(trajectory)//256)]])
    up = np.median(ups, axis=0)
    return up / np.linalg.norm(up)


def display_basis(trajectory: np.ndarray, guide: np.ndarray | None = None) -> np.ndarray:
    """Every method estimates vertical from its own cameras; heading can be shared."""
    up = estimate_up(trajectory)
    positions = trajectory[::max(1, len(trajectory)//1000), 1:4]
    if guide is None:
        _, eigenvectors = np.linalg.eigh(np.cov(positions.T))
        guide = eigenvectors[:, 2]
    forward = guide - np.dot(guide, up) * up
    forward /= np.linalg.norm(forward)
    across = np.cross(up, forward)
    across /= np.linalg.norm(across)
    return np.column_stack((forward, across, up))


def first_camera_alignment(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = rotation_from_quat(reference[0, 4:8]) @ rotation_from_quat(source[0, 4:8]).T
    translation = reference[0, 1:4] - rotation @ source[0, 1:4]
    return rotation, translation


def pack_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> dict:
    """Gzip RCP1: 32-byte header, then N×(uint16 xyz, uint8 RGB)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]
    if len(xyz) < 8:
        raise ValueError(f"Too few points for {path}")
    minimum = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - minimum, 1e-6)
    quantized = np.rint((xyz - minimum) / extent * 65535).clip(0, 65535).astype("<u2")
    records = np.empty(len(xyz), dtype=np.dtype([
        ("xyz", "<u2", (3,)), ("rgb", "u1", (3,))
    ]))
    records["xyz"] = quantized
    records["rgb"] = rgb
    header = struct.pack("<4sI6f", b"RCP1", len(xyz), *minimum, *extent)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("wb") as stream:
        # gzip.compress gives a deterministic mtime and keeps publication atomic.
        stream.write(gzip.compress(header + records.tobytes(), compresslevel=6, mtime=0))
    temp.replace(path)
    return dict(points=len(xyz), bytes=path.stat().st_size,
                min=minimum.tolist(), max=(minimum+extent).tolist())


def sample_ply(path: Path, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = read_ply(path)
    indices = np.linspace(0, len(cloud)-1, min(len(cloud), maximum), dtype=np.int64)
    selected = cloud[indices]
    xyz = np.column_stack((selected["x"], selected["y"], selected["z"]))
    rgb = np.column_stack((selected["red"], selected["green"], selected["blue"]))
    return xyz, rgb


def compact_trajectory(trajectory: np.ndarray, basis: np.ndarray, center: np.ndarray,
                       rotation: np.ndarray | None = None, translation: np.ndarray | None = None,
                       maximum: int = 1200) -> list[list[float]]:
    stride = max(1, int(np.ceil(len(trajectory) / maximum)))
    rows = trajectory[::stride]
    if rows[-1, 0] != trajectory[-1, 0]:
        rows = np.vstack((rows, trajectory[-1]))
    xyz = rows[:, 1:4]
    if rotation is not None:
        xyz = xyz @ rotation.T + translation
    xyz = (xyz - center) @ basis
    return np.round(xyz, 4).tolist()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
