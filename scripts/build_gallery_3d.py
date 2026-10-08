"""Build compact, actual point-cloud products for the static comparison gallery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pointpack import (
    compact_trajectory, display_basis, pack_points, read_tum, sample_ply, write_json
)

SITE = Path(__file__).resolve().parents[1]
SLAM = SITE.parent / "FwdSlam"
RESULTS = SITE.parent / "benchmark_results/consolidated_20260924"
OXFORD_LINGBOT_ABOT = SITE.parent / "benchmark_results/oxford_spires_lingbot_abot"
OUT = SITE / "assets/3d"
METHODS = ("ours", "lingbot", "abot", "scarf", "vggt_rgbd", "vggt_pi3x", "twodgs", "profusion")
FASTCAMO = (
    "apartment_1", "apartment_2", "gym", "lab", "lounge_1", "lounge_2",
    "meeting_room", "office", "stairwell", "studio", "workshop_1", "workshop_2",
)
OXFORD = (
    "2024-03-12-keble-college-04", "2024-03-13-observatory-quarter-01",
    "2024-03-14-blenheim-palace-01", "2024-03-18-christ-church-02",
)
OXFORD_METHODS = ("ours", "lingbot", "abot", "scarf", "vggt_rgbd", "vggt_pi3x")


def source_dir(group: str, scene: str, method: str) -> Path:
    if method == "ours":
        if group == "fastcamo":
            return SLAM / "outputs/ours_benchmarks_20260925/fastcamo_real" / scene.replace("_", "-")
        return SLAM / "outputs/ours_benchmarks_20260925/oxford_spires" / scene
    if group == "oxford_spires" and method in {"lingbot", "abot"}:
        return OXFORD_LINGBOT_ABOT / scene / method
    return RESULTS / group / scene / method


def emit_cloud(group: str, scene: str, method: str, maximum: int) -> dict:
    source = source_dir(group, scene, method)
    if not (source / "pointcloud.ply").is_file() or not (source / "trajectory.tum").is_file():
        return dict(status="pending_transfer", method=method,
                    missing=[f"{group}/{scene}/{method}/{file}" for file in ("pointcloud.ply", "trajectory.tum")
                             if not (source / file).is_file()])
    trajectory = read_tum(source / "trajectory.tum")
    basis = display_basis(trajectory)
    center = np.median(trajectory[:, 1:4], axis=0)
    xyz, rgb = sample_ply(source / "pointcloud.ply", maximum)
    xyz = (xyz - center) @ basis
    relative = f"{group}/{scene}/{method}"
    point_path = OUT / f"{relative}.rcp.gz"
    info = pack_points(point_path, xyz, rgb)
    track = [] if group == "fastcamo" else compact_trajectory(trajectory, basis, center, maximum=650)
    return dict(status="ready", method=method, cloud=str(point_path.relative_to(SITE)),
                trajectory=track, frames=len(trajectory),
                source_time_range=[float(trajectory[0,0]),float(trajectory[-1,0])],
                sampled_points=info["points"],
                compressed_bytes=info["bytes"], display_up=basis[:, 2].tolist(),
                source_id=f"{group}/{scene}/{method}",
                cloud_role="sensor_RGB_D_backprojection_with_inferred_poses")


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    reflection = np.eye(3)
    reflection[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ reflection @ vt
    translation = b - a @ rotation
    return rotation, translation


def match_to_gt(trajectory: np.ndarray, gt_times: np.ndarray, gt_xyz: np.ndarray):
    nearest = np.searchsorted(gt_times, trajectory[:, 0])
    nearest = np.clip(nearest, 0, len(gt_times)-1)
    previous = np.maximum(nearest-1, 0)
    nearest = np.where(abs(gt_times[previous]-trajectory[:, 0]) < abs(gt_times[nearest]-trajectory[:, 0]),
                       previous, nearest)
    valid = abs(gt_times[nearest] - trajectory[:, 0]) <= .02
    if valid.sum() < 3:
        raise ValueError("Too few trajectory/GT timestamp matches")
    rotation, translation = rigid_fit(trajectory[valid, 1:4], gt_xyz[nearest[valid]])
    return trajectory[:, 1:4] @ rotation + translation, int(valid.sum())


def oxford_ground_truth(scene: str, ours: np.ndarray):
    sys.path[:0] = [str(SLAM / "tools"), str(SLAM / "src")]
    from evaluate_outdoor_trajectory import oxford_camera_gt
    dataset = Path("/nvme1/datasets/Oxford_Spires")
    mask, poses = oxford_camera_gt(
        ours[:, 0],
        dataset / "sequences" / scene / "processed/trajectory/gt-tum.txt",
        dataset / "calibration/calibration-sequences-2024-06-29/cam-lidar-imu.yaml",
    )
    times = ours[mask, 0]
    xyz = poses[:, :3, 3]
    gt_tum = np.column_stack((times, xyz, Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
    basis = display_basis(gt_tum)
    center = np.median(xyz, axis=0)
    return times, xyz, basis, center


def build_fastcamo() -> None:
    scenes = {}
    for scene in FASTCAMO:
        entries = {}
        for method in METHODS:
            entries[method] = emit_cloud("fastcamo", scene, method, 260_000)
        scenes[scene] = entries
        print(f"FastCaMo {scene}: {sum(e['status']=='ready' for e in entries.values())}/{len(entries)} clouds", flush=True)
    write_json(OUT / "fastcamo.json", dict(group="fastcamo", no_trajectory_ground_truth=True,
                                           show_trajectories=False, scenes=scenes))


def build_oxford() -> None:
    output = OUT / "oxford.json"
    previous = json.loads(output.read_text()) if output.is_file() else {"scenes": {}}
    scenes = {}
    for scene in OXFORD:
        ours = read_tum(source_dir("oxford_spires", scene, "ours") / "trajectory.tum")
        gt_times, gt_xyz, gt_basis, gt_center = oxford_ground_truth(scene, ours)
        old_scene = previous["scenes"].get(scene, {})
        plot = dict(old_scene.get("plot", {}))
        if "gt" not in plot:
            stride = max(1, len(gt_xyz)//1200)
            plot["gt"] = np.round((gt_xyz[::stride]-gt_center) @ gt_basis[:, :2], 3).tolist()
        entries = {}
        for method in OXFORD_METHODS:
            old_entry = old_scene.get("methods", {}).get(method, {})
            if (old_entry.get("status") == "ready" and method in plot and
                    (SITE / old_entry["cloud"]).is_file()):
                entries[method] = old_entry
                continue
            entries[method] = emit_cloud("oxford_spires", scene, method, 260_000)
            if entries[method]["status"] == "ready":
                track = read_tum(source_dir("oxford_spires", scene, method) / "trajectory.tum")
                aligned, matches = match_to_gt(track, gt_times, gt_xyz)
                step = max(1, len(aligned)//1200)
                plot[method] = np.round((aligned[::step]-gt_center) @ gt_basis[:, :2], 3).tolist()
                entries[method]["gt_matched_poses_for_display"] = matches
        scenes[scene] = dict(methods=entries, plot=plot,
                             full_sequence_frames=len(ours), gt_positions=len(gt_xyz))
        print(f"Oxford {scene}: {sum(e['status']=='ready' for e in entries.values())}/{len(entries)} clouds", flush=True)
    write_json(OUT / "oxford.json", dict(group="oxford_spires", scenes=scenes,
                                         trajectory_plot="rigid_aligned_to_released_GT_for_display_no_new_metric"))


if __name__ == "__main__":
    build_fastcamo()
    build_oxford()
