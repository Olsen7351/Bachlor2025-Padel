"""
Standalone Mode Runners.

Contains individual runners for each pipeline mode:
- Player tracking (with progress indicator)
- Rally detection  
- Shot classification
- Heatmap generation

Each runner can be used independently of the full pipeline.
"""

import sys
import cv2
import numpy as np
import pandas as pd
import pickle
from typing import Optional
from tqdm import tqdm

from ..config import PipelineConfig
from ..trackers import RallyTracker
from ..processors import ShotClassificationProcessor
from ..utils import (
    ensure_dir,
    get_output_paths,
    parse_calib_points,
    build_homography,
    load_calib_csv,
    to_meters,
    load_court_config,
    generate_heatmap
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def draw_player_annotations(
    frame: np.ndarray, 
    xyxy: np.ndarray, 
    ids: np.ndarray, 
    pts_px: np.ndarray, 
    pts_m: Optional[np.ndarray] = None
) -> None:
    """Draw player bounding boxes and annotations on frame."""
    for i, box in enumerate(xyxy.astype(int)):
        x1, y1, x2, y2 = box
        tid = int(ids[i])
        px, py = int(pts_px[i, 0]), int(pts_px[i, 1])

        label = f"ID {tid}"
        if pts_m is not None and pts_m.shape[0] > i and np.all(np.isfinite(pts_m[i])):
            xm, ym = pts_m[i]
            label += f" | {xm:.2f}m, {ym:.2f}m"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)


def run_standalone_player_tracker(config: PipelineConfig, args) -> None:
    """
    Run standalone player tracking with YOLOv8 + ByteTrack.
    
    Now includes progress indicator showing frame count.
    
    Args:
        config: Pipeline configuration
        args: Argparse namespace (for backward compatibility)
    """
    if YOLO is None:
        raise ImportError("ultralytics package required for player tracking")
    
    video_path = config.video.input_path
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not fps or fps <= 1:
        fps = 25.0
    
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get output paths
    output_paths = get_output_paths(video_path)
    out_video = config.video.output_path or output_paths['video'].replace('_output.avi', '_players.mp4')
    out_csv = getattr(args, 'out_csv', None) or output_paths['player_csv']
    
    ensure_dir(out_video)
    ensure_dir(out_csv)

    # Load homography
    H = _load_homography(config, args)
    
    model = YOLO(config.models.player_detector)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    rows = []
    frame_idx = -1

    track_kwargs = dict(
        source=video_path, 
        stream=True, 
        conf=config.player_tracking.confidence,
        classes=[0], 
        tracker=config.player_tracking.tracker_config, 
        device=config.player_tracking.device,
        verbose=False  # Suppress per-frame YOLO output
    )
    if config.player_tracking.image_size:
        track_kwargs["imgsz"] = config.player_tracking.image_size

    print(f"Running player tracking on {video_path}...")
    print(f"Total frames: {total_frames}, Resolution: {width}x{height}, FPS: {fps:.1f}")

    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Player tracking", unit="frame")

    for result in model.track(**track_kwargs):
        frame_idx += 1
        pbar.update(1)
        
        frame = result.orig_img

        if result.boxes is None or len(result.boxes) == 0:
            writer.write(frame)
            continue

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],))
        ids = boxes.id
        if ids is None:
            ids = np.arange(xyxy.shape[0])
        else:
            ids = ids.cpu().numpy().astype(int)

        feet_px = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]))
        feet_m = to_meters(H, feet_px) if H is not None else None

        if feet_m is not None:
            xm = feet_m[:, 0]
            ym = feet_m[:, 1]
            valid = np.isfinite(xm) & np.isfinite(ym)
            in_court = (xm >= 0.0) & (xm <= config.court.court_width) & \
                       (ym >= 0.0) & (ym <= config.court.court_height)
            keep = valid & in_court

            xyxy = xyxy[keep]
            confs = confs[keep]
            ids = ids[keep]
            feet_px = feet_px[keep]
            feet_m = feet_m[keep]

            if xyxy.shape[0] == 0:
                writer.write(frame)
                continue

        for i in range(xyxy.shape[0]):
            if feet_m is not None and np.all(np.isfinite(feet_m[i])):
                xm_val, ym_val = float(feet_m[i, 0]), float(feet_m[i, 1])
            else:
                xm_val, ym_val = np.nan, np.nan

            rows.append((
                frame_idx, int(ids[i]), 
                float(feet_px[i, 0]), float(feet_px[i, 1]),
                xm_val, ym_val, float(confs[i])
            ))

        draw_player_annotations(frame, xyxy, ids, feet_px, feet_m)
        writer.write(frame)

    pbar.close()
    writer.release()
    cap.release()

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x_px", "y_px", "x_m", "y_m", "confidence"])
    df.to_csv(out_csv, index=False)

    print("\nDone!")
    print(f"- Annotated video: {out_video}")
    print(f"- Tracks CSV:      {out_csv}")
    
    # Auto-generate heatmap
    if config.player_tracking.auto_heatmap:
        _generate_auto_heatmap(config, args, out_csv, output_paths, first_frame)


def _load_homography(config: PipelineConfig, args) -> Optional[np.ndarray]:
    """Load homography matrix from various sources."""
    H = None
    
    # Priority 1: Load from court JSON
    if config.court.court_number and config.court.court_json:
        try:
            court_config = load_court_config(config.court.court_number, config.court.court_json)
            H = court_config.get('HOMOGRAPHY')
            if H is not None:
                print(f"Loaded homography from court {config.court.court_number} in {config.court.court_json}")
        except Exception as e:
            print(f"Warning: Could not load court config: {e}")
    
    # Priority 2: CSV calibration file
    if H is None and config.court.calib_csv:
        try:
            H = load_calib_csv(config.court.calib_csv)
            print(f"Loaded homography from {config.court.calib_csv}")
        except Exception as e:
            print(f"Warning: Failed to load {config.court.calib_csv}: {e}")
    
    # Priority 3: Command line calibration string
    if H is None and config.court.calib_string:
        try:
            img_pts = parse_calib_points(config.court.calib_string)
            H = build_homography(img_pts, court_w=config.court.court_width, court_h=config.court.court_height)
            print("Loaded homography from 4-corner string.")
        except Exception as e:
            print(f"Warning: Failed to parse calibration string: {e}")
    
    return H


def _generate_auto_heatmap(
    config: PipelineConfig, 
    args, 
    csv_path: str, 
    output_paths: dict, 
    first_frame: np.ndarray
) -> None:
    """Generate heatmap automatically after tracking."""
    print("\nGenerating heatmap automatically...")
    
    court_img_path = output_paths['court_frame']
    ensure_dir(court_img_path)
    cv2.imwrite(court_img_path, first_frame)
    
    heatmap_output = output_paths['heatmap']
    
    players_filter = None
    if config.heatmap.players_filter:
        players_filter = [int(s) for s in config.heatmap.players_filter.split(",") if s.strip().isdigit()]
    
    try:
        generate_heatmap(
            csv_path=csv_path,
            court_img_path=court_img_path,
            out_png=heatmap_output,
            bins_x=config.heatmap.bins_x,
            bins_y=config.heatmap.bins_y,
            gauss=config.heatmap.gaussian_sigma,
            inplay_only=config.heatmap.inplay_only,
            min_conf=config.heatmap.min_confidence,
            players=players_filter,
            heat_alpha=config.heatmap.alpha,
            show_axes=config.heatmap.show_axes,
        )
        print(f"- Heatmap:         {heatmap_output}")
    except Exception as e:
        print(f"Warning: Could not generate heatmap: {e}")


def run_rally_detection(config: PipelineConfig, args) -> tuple:
    """
    Run standalone rally detection from ball detections pickle.
    
    Args:
        config: Pipeline configuration
        args: Argparse namespace
        
    Returns:
        Tuple of (rallies, rally_tracker)
    """
    print("=" * 60)
    print("STANDALONE RALLY DETECTION")
    print("=" * 60)
    
    ball_pkl = getattr(args, 'ball_detections_pkl', None)
    if not ball_pkl:
        print("Error: --ball_detections_pkl required for rally mode")
        sys.exit(1)
    
    print(f"Loading ball detections from {ball_pkl}...")
    with open(ball_pkl, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, tuple):
            ball_detections, _ = data
        else:
            ball_detections = data
    
    print(f"Loaded {len(ball_detections)} frames of ball detections")
    
    frame_height = getattr(args, 'frame_height', None) or 720
    
    rally_tracker = RallyTracker(
        fps=config.video.fps,
        frame_height=frame_height,
        min_velocity=config.rally.min_velocity,
        serve_velocity=config.rally.serve_velocity,
        base_gap_during_rally=config.rally.base_gap_during_rally,
        base_rally_end_gap=config.rally.base_rally_end_gap,
        max_gap_extension=config.rally.max_gap_extension,
        min_rally_frames=config.rally.min_rally_frames,
        min_rally_distance=config.rally.min_rally_distance,
    )
    
    rallies = rally_tracker.process_all_frames(ball_detections)
    
    summary = rally_tracker.get_summary()
    print("\n" + "-" * 40)
    print("RALLY SUMMARY")
    print("-" * 40)
    print(f"  Total rallies: {summary['rally_count']}")
    
    if summary['rally_count'] > 0:
        print(f"  Total rally time: {summary['total_rally_time_sec']:.1f}s")
        print(f"  Average duration: {summary['avg_duration_sec']:.1f}s")
        
        for rally in rallies:
            print(f"  Rally #{rally.rally_id}: "
                  f"Frames {rally.start_frame}-{rally.end_frame} "
                  f"({rally.duration_seconds(config.video.fps):.1f}s)")
    
    rally_csv = getattr(args, 'rally_csv', None) or 'output_videos/rallies.csv'
    ensure_dir(rally_csv)
    rally_tracker.export_rallies_csv(rally_csv)
    print(f"\nRally CSV saved to: {rally_csv}")
    
    return rallies, rally_tracker


def run_shot_inference(config: PipelineConfig, args) -> None:
    """
    Run standalone shot classification inference.
    
    Args:
        config: Pipeline configuration
        args: Argparse namespace
    """
    device = config.device
    print(f"Using device: {device}")
    
    video_path = config.video.input_path
    output_paths = get_output_paths(video_path)
    output_path = config.video.output_path or output_paths['video'].replace('_output.avi', '_shots.mp4')
    
    # Read video
    print(f"\nReading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    pbar = tqdm(total=total_frames, desc="Loading video", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    
    print(f"Loaded {len(frames)} frames at {fps:.1f} FPS")
    
    # Process
    processor = ShotClassificationProcessor(
        shot_model_path=config.models.shot_classifier,
        yolo_pose_path=config.models.yolo_pose,
        tracknet_path=config.models.tracknet,
        device=device,
        confidence_threshold=config.shot.confidence_threshold,
        window_size=config.shot.window_size,
        stride=config.shot.stride,
        classifier_resolution=config.shot.classifier_resolution,
    )
    
    shot_data = processor.process_video(frames, fps=fps, yolo_resolution=config.shot.yolo_resolution)
    
    # Draw overlay
    print("Drawing shot overlays...")
    output_frames = processor.draw_shot_overlay(frames, shot_data)
    
    # Save video
    ensure_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    
    for frame in tqdm(output_frames, desc="Writing video", unit="frame"):
        out.write(frame)
    out.release()
    
    # Export shots CSV
    shots_csv = output_paths['shots_csv']
    processor.export_shots_csv(shots_csv)
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Shots CSV: {shots_csv}")
    print(f"\nPlayer Stats: {processor.player_stats}")


def run_heatmap_generation(config: PipelineConfig, args) -> None:
    """
    Run standalone heatmap generation.
    
    Args:
        config: Pipeline configuration
        args: Argparse namespace
    """
    csv_path = getattr(args, 'csv', None)
    court_img = getattr(args, 'court_img', None)
    
    if not csv_path or not court_img:
        print("Error: --csv and --court_img required for heatmap mode")
        sys.exit(1)

    players = None
    if config.heatmap.players_filter:
        players = [int(s) for s in config.heatmap.players_filter.split(",") if s.strip().isdigit()]
    
    output = getattr(args, 'heatmap_output', None) or get_output_paths(csv_path)['heatmap']

    generate_heatmap(
        csv_path=csv_path,
        court_img_path=court_img,
        out_png=output,
        bins_x=config.heatmap.bins_x,
        bins_y=config.heatmap.bins_y,
        gauss=config.heatmap.gaussian_sigma,
        inplay_only=config.heatmap.inplay_only,
        min_conf=config.heatmap.min_confidence,
        players=players,
        heat_alpha=config.heatmap.alpha,
        show_axes=config.heatmap.show_axes,
    )