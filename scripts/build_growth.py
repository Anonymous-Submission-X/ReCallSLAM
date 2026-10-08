"""Filtered RGB-D growth videos and matching interactive clouds for long scenes.

No inference or SLAM is run. The same retained sensor pixels are backprojected
at each method's delivered camera poses. Every display independently estimates up.
"""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pointpack import (
    compact_trajectory, display_basis, first_camera_alignment, pack_points,
    read_tum, write_json,
)

SITE = Path(__file__).resolve().parents[1]
SLAM = SITE.parent / "FwdSlam"
RESULTS = SITE.parent / "benchmark_results/consolidated_20260924/aaai_videos"
DATA = Path("/data/StructuredLight/aaai_videos")
OUT = SITE / "assets/3d/long"
VIDEO = SITE / "assets/videos"
FFMPEG = Path(__import__("imageio_ffmpeg").get_ffmpeg_exe())
WIDTH, HEIGHT, FPS, VIDEO_FRAMES = 960, 540, 12, 120
DEPTH_MAX_M = 8.0
EDGE_RTOL = .008
EDGE_ABS_M = .005
SCENES = {
    "build_2_f3-4": dict(total=4300, samples=260, per_frame=5000,
                         methods=("ours", "lingbot", "abot", "scarf", "vggt_rgbd",
                                  "vggt_pi3x", "twodgs", "profusion")),
    "highschool": dict(total=27002, samples=460, per_frame=4800,
                       methods=("ours", "lingbot", "abot", "scarf", "vggt_rgbd", "vggt_pi3x")),
    "bupt": dict(total=30079, samples=480, per_frame=4800,
                 methods=("ours", "lingbot", "abot", "scarf", "vggt_rgbd", "vggt_pi3x")),
}


def source_dir(scene: str, method: str) -> Path:
    if method == "ours":
        return SLAM / "outputs/ours_benchmarks_20260925/aaai_videos" / scene
    return RESULTS / scene / method


def camera_intrinsics(scene: str) -> np.ndarray:
    sys.path.insert(0, str(SLAM / "src"))
    from fwdslam.runnable.rgbd import load_intrinsics
    matrix, calibration = load_intrinsics(DATA / scene / "params.npy", dataset_profile="aaai_rgb")
    assert calibration["width_px"] == 848 and calibration["height_px"] == 480
    return matrix


def nearest_pose(trajectory: np.ndarray, frame: int):
    times = trajectory[:, 0]
    index = np.searchsorted(times, frame)
    candidates = [min(index, len(times)-1), max(0, index-1)]
    index = min(candidates, key=lambda i: abs(times[i]-frame))
    if abs(times[index]-frame) > .51:
        return None
    return trajectory[index]


def quat_rotation(quaternion: np.ndarray) -> np.ndarray:
    from pointpack import rotation_from_quat
    return rotation_from_quat(quaternion)


def filtered_local_pixels(scene: str, frame: int, intrinsics: np.ndarray,
                          pose: np.ndarray, up_world: np.ndarray, limit: int):
    from fwdslam.runnable.rgbd import dataset_frame_paths
    # Use the same ordinal RGB and pred_depth binding as the delivered run.
    rgb_path, depth_path = dataset_frame_paths(DATA / scene, frame, dataset_profile="aaai_rgb")
    with Image.open(depth_path) as image:
        depth = np.asarray(image, dtype=np.float32) / 1000.0
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > .1) & (depth <= DEPTH_MAX_M)
    # Remove holes and thin depth boundaries before sampling. The 3x3 range
    # rejects smoothed silhouettes and cross-surface interpolation.
    kernel = np.ones((3, 3), np.uint8)
    full_support = cv2.erode(valid.astype(np.uint8), kernel, iterations=2).astype(bool)
    minimum = cv2.erode(np.where(valid, depth, 100.), kernel)
    maximum = cv2.dilate(np.where(valid, depth, 0.), kernel)
    stable = full_support & ((maximum-minimum) <= EDGE_ABS_M + EDGE_RTOL * depth)
    grid_y, grid_x = np.mgrid[2:depth.shape[0]-2:2, 2:depth.shape[1]-2:2]
    y, x = grid_y[stable[grid_y, grid_x]], grid_x[stable[grid_y, grid_x]]
    if len(x) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), dict(valid=0, edges=0, ceiling=0)
    # A stable, spread-out sample keeps one-pixel splats legible without inflating
    # the point radius. Sensor pixels and the original depth array remain intact.
    take = np.linspace(0, len(x)-1, min(len(x), limit), dtype=np.int64)
    x, y = x[take], y[take]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = depth[y, x]

    def camera_point(px, py, dz):
        return np.column_stack(((px-cx)*dz/fx, (py-cy)*dz/fy, dz))

    local = camera_point(x, y, z)
    left = camera_point(x-1, y, depth[y, x-1])
    right = camera_point(x+1, y, depth[y, x+1])
    upper = camera_point(x, y-1, depth[y-1, x])
    lower = camera_point(x, y+1, depth[y+1, x])
    normal = np.cross(right-left, lower-upper)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    rotation = quat_rotation(pose[4:8])
    world = local @ rotation.T + pose[1:4]
    height = (world-pose[1:4]) @ up_world
    horizontal = np.abs((normal @ rotation.T) @ up_world) > .60
    # The height cap removes overhead points even when oversmoothed depth
    # corrupts the estimated normal; the lower threshold catches flat patches.
    ceiling = (height > .40) | ((height > .20) & horizontal)
    return local[~ceiling].astype(np.float32), rgb[y[~ceiling], x[~ceiling]], dict(
        valid=int(valid.sum()), edges=int(stable.sum()), ceiling=int(ceiling.sum()))


def load_frames(scene: str, config: dict, ours: np.ndarray, matrix: np.ndarray):
    selected = np.unique(np.linspace(0, config["total"]-1, config["samples"], dtype=int))
    from pointpack import estimate_up
    up = estimate_up(ours)
    frames = []
    counters = dict(selected=len(selected), input_valid=0, after_edges=0, removed_ceiling=0, retained=0)
    for ordinal, frame in enumerate(selected):
        pose = nearest_pose(ours, int(frame))
        if pose is None:
            continue
        xyz, rgb, count = filtered_local_pixels(scene, int(frame), matrix, pose, up, config["per_frame"])
        frames.append((int(frame), xyz, rgb))
        counters["input_valid"] += count["valid"]
        counters["after_edges"] += count["edges"]
        counters["removed_ceiling"] += count["ceiling"]
        counters["retained"] += len(xyz)
        if ordinal % 40 == 0 or ordinal == len(selected)-1:
            print(f"{scene}: filtered {ordinal+1}/{len(selected)} RGB-D frames, {counters['retained']} points", flush=True)
    return frames, counters


def build_method_cloud(scene: str, method: str, frames, ours: np.ndarray,
                       ours_guide: np.ndarray, export_cloud: bool = True):
    trajectory = read_tum(source_dir(scene, method) / "trajectory.tum")
    if method == "ours":
        guide = ours_guide
    else:
        rotation, _ = first_camera_alignment(trajectory, ours)
        guide = rotation.T @ ours_guide
    basis = display_basis(trajectory, guide=guide)
    center = np.median(trajectory[:, 1:4], axis=0)
    points, colors, owners = [], [], []
    for frame, local, rgb in frames:
        pose = nearest_pose(trajectory, frame)
        if pose is None or not len(local):
            continue
        rotation = quat_rotation(pose[4:8])
        world = local @ rotation.T + pose[1:4]
        points.append(((world-center) @ basis).astype(np.float32))
        colors.append(rgb)
        owners.append(np.full(len(local), frame, dtype=np.int32))
    if not points:
        raise RuntimeError(f"No visualized points: {scene}/{method}")
    xyz = np.concatenate(points)
    rgb = np.concatenate(colors)
    frame_ids = np.concatenate(owners)
    cloud_path = OUT / scene / f"{method}.rcp.gz"
    if export_cloud:
        web_indices = np.linspace(0, len(xyz)-1, min(len(xyz), 600_000), dtype=np.int64)
        packed = pack_points(cloud_path, xyz[web_indices], rgb[web_indices])
    else:
        packed = {"points": 0}
    track = compact_trajectory(trajectory, basis, center, maximum=1600)
    return dict(status="ready", method=method, cloud=str(cloud_path.relative_to(SITE)),
                trajectory=track, frames=len(trajectory), point_count=packed["points"],
                filtered_video_points=len(xyz),
                source_frame_range=[int(round(trajectory[0,0])), int(round(trajectory[-1,0]))],
                display_up=basis[:, 2].tolist(), source_id=f"aaai_videos/{scene}/{method}",
                filter=f"bound pred_depth 0.1-{DEPTH_MAX_M:g}m; twice-eroded 3x3 full-support; depth-range <= 0.005m+0.008z; overhead height/normal mask",
                xyz=xyz, rgb=rgb, frame_ids=frame_ids, trajectory_raw=trajectory,
                basis=basis, center=center)


def projection_axes():
    # The display basis already puts each reconstruction's estimated up on Z.
    # Looking straight down makes growth and trajectory geometry readable.
    right = np.array([1., 0., 0.], dtype=np.float64)
    upward = np.array([0., 1., 0.], dtype=np.float64)
    towards = np.array([0., 0., 1.], dtype=np.float64)
    return right, upward, towards


def video_projection(xyz: np.ndarray, trajectory: np.ndarray):
    right, upward, towards = projection_axes()
    screen = np.column_stack((xyz @ right, xyz @ upward))
    tr = np.column_stack((trajectory @ right, trajectory @ upward))
    sample = np.vstack((screen[::max(1,len(screen)//100_000)], tr))
    low, high = np.quantile(sample, [.005, .995], axis=0)
    mid = (low+high)/2
    half = np.maximum((high-low)*.61, [2., 2.])
    aspect = WIDTH/HEIGHT
    if half[0]/half[1] > aspect:
        half[1] = half[0]/aspect
    else:
        half[0] = half[1]*aspect
    low, high = mid-half, mid+half
    x = np.rint((screen[:,0]-low[0]) / (high[0]-low[0]) * (WIDTH-1)).astype(np.int32)
    y = np.rint((high[1]-screen[:,1]) / (high[1]-low[1]) * (HEIGHT-1)).astype(np.int32)
    tx = np.rint((tr[:,0]-low[0]) / (high[0]-low[0]) * (WIDTH-1)).astype(np.int32)
    ty = np.rint((high[1]-tr[:,1]) / (high[1]-low[1]) * (HEIGHT-1)).astype(np.int32)
    return x, y, xyz @ towards, tx, ty, low, high


def append_points(canvas: np.ndarray, depth_buffer: np.ndarray,
                  x: np.ndarray, y: np.ndarray, depth: np.ndarray, rgb: np.ndarray):
    valid = (x>=0)&(x<WIDTH)&(y>=0)&(y<HEIGHT)&np.isfinite(depth)
    pixel = y[valid]*WIDTH+x[valid]
    if not len(pixel):
        return
    distance, colors = depth[valid].astype(np.float32), rgb[valid]
    order = np.lexsort((distance, pixel))
    unique, positions = np.unique(pixel[order], return_index=True)
    final = order[np.r_[positions[1:]-1, len(order)-1]]
    closer = distance[final] > depth_buffer[unique]
    chosen = unique[closer]
    depth_buffer[chosen] = distance[final[closer]]
    canvas.reshape(-1,3)[chosen] = colors[final[closer]]


def trajectory_color(t: float) -> tuple[int, int, int]:
    # Viridis from 0.40 to 1.00 keeps its recognizable progression bright.
    stops = ((42, 120, 142), (30, 156, 137), (68, 191, 112),
             (155, 217, 60), (253, 231, 37))
    position = min(1., max(0., t)) * (len(stops)-1)
    index = min(len(stops)-2, int(position))
    alpha = position - index
    return tuple(int(round((1-alpha)*stops[index][channel] + alpha*stops[index+1][channel]))
                 for channel in range(3))


def draw_trajectory(draw: ImageDraw.ImageDraw, tx: np.ndarray, ty: np.ndarray,
                    visible: np.ndarray) -> None:
    if len(visible) < 2:
        return
    stride = max(1, len(visible)//900)
    indices = visible[::stride]
    if indices[-1] != visible[-1]:
        indices = np.r_[indices, visible[-1]]
    for a, b in zip(indices[:-1], indices[1:]):
        if (0 <= tx[a] < WIDTH and 0 <= ty[a] < HEIGHT and
                0 <= tx[b] < WIDTH and 0 <= ty[b] < HEIGHT):
            draw.line(((int(tx[a]), int(ty[a])), (int(tx[b]), int(ty[b]))),
                      fill=trajectory_color(float(b)/max(1, len(tx)-1)), width=3)


def render_video(scene: str, method: str, result: dict, total: int):
    xyz, rgb, owners = result["xyz"], result["rgb"], result["frame_ids"]
    trajectory = result["trajectory_raw"]
    basis, center = result["basis"], result["center"]
    track = (trajectory[:, 1:4]-center) @ basis
    x, y, depth, tx, ty, low, high = video_projection(xyz, track)
    # Points are ordered by source frame. A standard near-surface z-buffer
    # resolves same-pixel collisions; every retained 3D point is one pixel.
    order = np.argsort(owners, kind="stable")
    x, y, depth, rgb, owners = x[order], y[order], depth[order], rgb[order], owners[order]
    canvas = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    depth_buffer = np.full(WIDTH*HEIGHT, -np.inf, np.float32)
    output = VIDEO / scene / f"{method}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".mp4.pending")
    command = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
               "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-f", "mp4", str(temp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    previous = 0
    first_frame = int(round(trajectory[0,0]))
    max_frame = int(round(trajectory[-1,0]))
    typical_stride = int(round(np.median(np.diff(trajectory[:,0]))))
    partial_interval = first_frame > 1 or max_frame < total-1-typical_stride
    preview = canvas.copy()
    preview_depth = depth_buffer.copy()
    append_points(preview, preview_depth, x, y, depth, rgb)
    intro = Image.fromarray(preview)
    draw_trajectory(ImageDraw.Draw(intro), tx, ty, np.arange(len(trajectory)))
    intro_bytes = np.asarray(intro, dtype=np.uint8).tobytes()
    for _ in range(FPS * 2):
        process.stdin.write(intro_bytes)
    for index, frame in enumerate(np.linspace(0, total-1, VIDEO_FRAMES, dtype=int)):
        end = int(np.searchsorted(owners, frame, side="right"))
        append_points(canvas, depth_buffer, x[previous:end], y[previous:end],
                      depth[previous:end], rgb[previous:end])
        previous = end
        image = Image.fromarray(canvas.copy())
        draw = ImageDraw.Draw(image)
        visible = np.flatnonzero(trajectory[:,0] <= frame)
        draw_trajectory(draw, tx, ty, visible)
        try:
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg failed for {scene}/{method}")
        if index % 30 == 0:
            print(f"{scene}/{method}: video frame {index+1}/{VIDEO_FRAMES}", flush=True)
    final_bytes = np.asarray(image, dtype=np.uint8).tobytes()
    for _ in range(FPS * 2):
        process.stdin.write(final_bytes)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg exited nonzero for {scene}/{method}")
    temp.replace(output)
    image.save(output.with_suffix(".webp"), "WEBP", quality=88, method=6)
    return dict(path=str(output.relative_to(SITE)), bytes=output.stat().st_size,
                fps=FPS, frames=VIDEO_FRAMES+FPS*4, pixel_radius=1,
                returned_frame_range=[first_frame,max_frame], partial_interval=partial_interval,
                projection="estimated-up orthographic top view; standard nearest-surface z-buffer",
                viewport_m=[low.tolist(),high.tolist()])


def build_scene(scene: str, config: dict, videos_only: bool = False):
    matrix = camera_intrinsics(scene)
    ours = read_tum(source_dir(scene,"ours") / "trajectory.tum")
    frames, counters = load_frames(scene, config, ours, matrix)
    ours_basis = display_basis(ours)
    guide = ours_basis[:,0]
    previous = json.loads((OUT / f"{scene}.json").read_text()) if videos_only else None
    methods = previous["methods"] if videos_only else {}
    for method in config["methods"]:
        result = build_method_cloud(scene, method, frames, ours, guide,
                                    export_cloud=not videos_only)
        video = render_video(scene, method, result, config["total"])
        if not videos_only:
            methods[method] = {key:value for key,value in result.items()
                               if key not in {"xyz","rgb","frame_ids","trajectory_raw","basis","center"}}
        methods[method]["video"] = video
        print(f"{scene}/{method}: video refreshed", flush=True)
    if videos_only:
        return previous
    return dict(scene=scene, total_frames=config["total"], source_frames=counters,
                depth_filter=dict(min_m=.1,max_m=DEPTH_MAX_M,edge_rtol=EDGE_RTOL,
                                  edge_absolute_m=EDGE_ABS_M,valid_neighborhood="twice-eroded 3x3",
                                  ceiling="height >0.40m above camera or flat normal >0.20m"),
                methods=methods)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--videos-only", action="store_true")
    args = parser.parse_args()
    for scene in ([args.scene] if args.scene else SCENES):
        config = SCENES[scene]
        payload = build_scene(scene, config, videos_only=args.videos_only)
        write_json(OUT / f"{scene}.json", payload)


if __name__ == "__main__":
    main()
