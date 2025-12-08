#!/usr/bin/env python3
"""
Merged Padel Match Analysis Pipeline
=====================================
Combines all team members' work:

1. Main Pipeline (your jens_inference.py):
   - PlayerTracker, BallTrackerTrackNet (with filters), RallyTracker
   - Polygon exclusion zones, trajectory filtering, player proximity
   
2. Your BallTrackerTrackNet class (ball_tracker_tracknet.py):
   - Batch processing with sophisticated filtering
   - PolygonExclusionFilter, CourtInclusionFilter, SpatialPlayerFilter, TrajectoryFilter
   
3. Jens's Shot Classification (simple_inference.py):
   - EnhancedVideoClassifier (R(2+1)D-18 + Bi-LSTM)
   - Streaming BallTracker + SmartBallTracker
   - Shot type detection (forehand, backhand, serve, etc.)
   
4. Teammate's Player Tracker (player_tracker.py):
   - YOLOv8 + ByteTrack with homography calibration
   - Pixel-to-meters conversion
   
5. Teammate's Heatmap (heatmap_on_image.py):
   - Position heatmap visualization

Usage Examples:
---------------
# Full pipeline (your main script - rally detection + ball tracking with filters)
python merged_padel_inference.py --mode full

# Shot classification mode (Jens's model)
python merged_padel_inference.py --mode shots --input video.mp4

# Standalone player tracking with ByteTrack
python merged_padel_inference.py --mode player_track --video input.mp4

# Generate heatmap from existing CSV
python merged_padel_inference.py --mode heatmap --csv tracks.csv --court_img frame.png

# Calibrate exclusion zones
python merged_padel_inference.py --mode calibrate --video input.mp4
"""

import argparse
import os
import sys
import csv
import json
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import video as video_models
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict, deque
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Ensure directory exists for a file path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_calib_points(s: str):
    """Parse calibration string: "x1,y1;x2,y2;x3,y3;x4,y4" (TL,TR,BR,BL)"""
    pts = []
    for pair in s.split(";"):
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Calibration must have exactly 4 points as x,y pairs.")
    return pts


def build_homography(corner_px, court_w=20.0, court_h=10.0):
    """Four-corner homography: TL,TR,BR,BL (pixels) -> meters."""
    target = np.array(
        [[0.0, 0.0], [court_w, 0.0], [court_w, court_h], [0.0, court_h]], 
        dtype=np.float32
    )
    H, _ = cv2.findHomography(corner_px, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H


def load_calib_csv(path):
    """Load homography from CSV with columns: x_px,y_px,x_m,y_m"""
    px_pts, m_pts = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            px_pts.append([float(row["x_px"]), float(row["y_px"])])
            m_pts.append([float(row["x_m"]), float(row["y_m"])])
    px_pts = np.array(px_pts, dtype=np.float32)
    m_pts = np.array(m_pts, dtype=np.float32)
    if len(px_pts) < 4:
        raise ValueError("Need at least 4 rows in calib CSV.")
    H, _ = cv2.findHomography(px_pts, m_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H


def to_meters(H, pts_xy):
    """Map Nx2 pixel points to meters with homography H."""
    if H is None or pts_xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_xy = pts_xy.astype(np.float32)
    pts_xy = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = (H @ pts_xy.T).T
    mapped /= mapped[:, 2:3]
    return mapped[:, :2]


def get_bbox_iou(box1, box2):
    """Calculate intersection over union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_motion_score(keypoints, prev_keypoints):
    """Calculate how much a player is moving (higher = more motion)."""
    if prev_keypoints is None:
        return 0.0
    displacement = np.linalg.norm(keypoints[:, :2] - prev_keypoints[:, :2], axis=1)
    weights = keypoints[:, 2] * prev_keypoints[:, 2]
    return np.sum(displacement * weights) / (np.sum(weights) + 1e-6)


# =============================================================================
# TRACKNET MODEL ARCHITECTURE (Shared by all implementations)
# =============================================================================

class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """TrackNet architecture for ball detection heatmap prediction."""
    
    def __init__(self, in_dim=27, out_dim=8):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
        x2 = self.down_block_2(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
        x3 = self.down_block_3(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
        x = self.bottleneck(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x


# =============================================================================
# YOUR BALL TRACKING FILTERS (from ball_tracker_tracknet.py)
# =============================================================================

class PolygonExclusionFilter:
    """Rejects ball detections in polygon-shaped exclusion zones."""

    def __init__(self):
        self.exclusion_zones: List[np.ndarray] = []
        self.zone_names: List[str] = []

    def add_polygon_zone(self, points: List[Tuple[float, float]], name: str = "zone"):
        polygon = np.array(points, dtype=np.float32)
        self.exclusion_zones.append(polygon)
        self.zone_names.append(name)
        print(f"Added exclusion zone '{name}': {len(points)} vertices")

    def add_left_glass_zone(self, top_left, top_right, bottom_right, bottom_left):
        """Add tilted exclusion zone for left glass panel."""
        self.add_polygon_zone([top_left, top_right, bottom_right, bottom_left], name="left_glass")

    def add_right_glass_zone(self, top_left, top_right, bottom_right, bottom_left):
        """Add tilted exclusion zone for right glass panel."""
        self.add_polygon_zone([top_left, top_right, bottom_right, bottom_left], name="right_glass")

    def is_excluded(self, x: float, y: float) -> Tuple[bool, str]:
        for zone, name in zip(self.exclusion_zones, self.zone_names):
            if cv2.pointPolygonTest(zone, (x, y), False) >= 0:
                return True, name
        return False, ""

    def filter_detection(self, bbox: List[float]) -> Tuple[bool, str]:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        is_excluded, zone_name = self.is_excluded(cx, cy)
        if is_excluded:
            return False, f"in_{zone_name}"
        return True, ""

    def draw_zones(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 180),
                   alpha: float = 0.35) -> np.ndarray:
        overlay = frame.copy()
        for zone in self.exclusion_zones:
            pts = zone.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


class CourtInclusionFilter:
    """Only accepts detections INSIDE the main court polygon."""

    def __init__(self, court_polygon: Optional[np.ndarray] = None):
        self.court_polygon = court_polygon

    def set_court_polygon(self, points: List[Tuple[float, float]]):
        self.court_polygon = np.array(points, dtype=np.float32)
        print(f"Court inclusion zone set: {len(points)} vertices")

    def filter_detection(self, bbox: List[float]) -> bool:
        if self.court_polygon is None:
            return True
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return cv2.pointPolygonTest(self.court_polygon, (cx, cy), False) >= 0

    def draw_court(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        if self.court_polygon is None:
            return frame
        pts = self.court_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)
        return frame


class SpatialPlayerFilter:
    """Filters ball detections based on player positions (no ID dependency)."""

    def __init__(self, court_polygon: Optional[np.ndarray] = None,
                 max_distance_any_player: float = 500,
                 max_distance_court_player: float = 400,
                 prefer_court_players: bool = True):
        self.court_polygon = court_polygon
        self.max_distance_any_player = max_distance_any_player
        self.max_distance_court_player = max_distance_court_player
        self.prefer_court_players = prefer_court_players

    def set_court_polygon(self, points: List[Tuple[float, float]]):
        self.court_polygon = np.array(points, dtype=np.float32)

    def _is_player_on_court(self, bbox: List[float]) -> bool:
        if self.court_polygon is None:
            return True
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        return cv2.pointPolygonTest(self.court_polygon, (foot_x, foot_y), False) >= 0

    def filter_detection(self, ball_bbox: List[float], player_detections: Dict) -> Tuple[bool, str]:
        if not player_detections:
            return True, "no_players"

        ball_cx = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_cy = (ball_bbox[1] + ball_bbox[3]) / 2

        min_dist_court_player = float('inf')
        min_dist_any_player = float('inf')

        for player_id, player_bbox in player_detections.items():
            player_cx = (player_bbox[0] + player_bbox[2]) / 2
            player_cy = (player_bbox[1] + player_bbox[3]) / 2
            dist = np.sqrt((ball_cx - player_cx)**2 + (ball_cy - player_cy)**2)
            min_dist_any_player = min(min_dist_any_player, dist)

            if self._is_player_on_court(player_bbox):
                min_dist_court_player = min(min_dist_court_player, dist)

        if self.prefer_court_players and min_dist_court_player < float('inf'):
            if min_dist_court_player <= self.max_distance_court_player:
                return True, "near_court_player"
            else:
                return False, f"far_from_court_players ({min_dist_court_player:.0f}px)"

        if min_dist_any_player <= self.max_distance_any_player:
            return True, "near_any_player"

        return False, f"far_from_all_players ({min_dist_any_player:.0f}px)"


class TrajectoryFilter:
    """Filters based on physical plausibility of ball movement."""

    def __init__(self, max_speed: float = 150.0, max_acceleration: float = 80.0,
                 history_size: int = 10, recovery_frames: int = 30):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.history_size = history_size
        self.recovery_frames = recovery_frames
        self.position_history: deque = deque(maxlen=history_size)
        self.velocity_history: deque = deque(maxlen=history_size)
        self.last_valid_pos: Optional[Tuple[float, float]] = None
        self.last_valid_frame: int = -1
        self.frames_without_detection: int = 0

    def reset(self):
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_valid_pos = None
        self.last_valid_frame = -1
        self.frames_without_detection = 0

    def filter_detection(self, frame_idx: int, ball_bbox: List[float]) -> Tuple[bool, str]:
        cx = (ball_bbox[0] + ball_bbox[2]) / 2
        cy = (ball_bbox[1] + ball_bbox[3]) / 2
        current_pos = (cx, cy)

        if self.frames_without_detection > self.recovery_frames:
            self.reset()
            self._accept(frame_idx, current_pos)
            return True, "reset"

        if self.last_valid_pos is None:
            self._accept(frame_idx, current_pos)
            return True, "first"

        frames_gap = max(1, frame_idx - self.last_valid_frame)
        dx = cx - self.last_valid_pos[0]
        dy = cy - self.last_valid_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / frames_gap

        effective_max_speed = self.max_speed * min(frames_gap, 5)
        if speed > effective_max_speed:
            self.frames_without_detection += 1
            return False, f"speed ({speed:.0f})"

        if len(self.velocity_history) >= 2:
            current_vel = (dx / frames_gap, dy / frames_gap)
            avg_vel = (
                np.mean([v[0] for v in self.velocity_history]),
                np.mean([v[1] for v in self.velocity_history])
            )
            accel = np.sqrt((current_vel[0] - avg_vel[0])**2 + (current_vel[1] - avg_vel[1])**2)

            if frames_gap <= 2 and accel > self.max_acceleration:
                self.frames_without_detection += 1
                return False, f"accel ({accel:.0f})"

        self._accept(frame_idx, current_pos)
        return True, "ok"

    def _accept(self, frame_idx: int, pos: Tuple[float, float]):
        if self.last_valid_pos is not None:
            frames_gap = max(1, frame_idx - self.last_valid_frame)
            vx = (pos[0] - self.last_valid_pos[0]) / frames_gap
            vy = (pos[1] - self.last_valid_pos[1]) / frames_gap
            self.velocity_history.append((vx, vy))

        self.position_history.append(pos)
        self.last_valid_pos = pos
        self.last_valid_frame = frame_idx
        self.frames_without_detection = 0

    def record_miss(self):
        self.frames_without_detection += 1


# =============================================================================
# YOUR BALL TRACKER (from ball_tracker_tracknet.py) - Batch Processing with Filters
# =============================================================================

class BallTrackerTrackNet:
    """
    Ball tracker with polygon exclusion zones, player filtering, trajectory filtering.
    Uses batch processing approach.
    """

    MODEL_WIDTH = 512
    MODEL_HEIGHT = 288
    SEQ_LEN = 8

    def __init__(self, tracknet_path: str = 'models/TrackNet_best.pt',
                 detection_threshold: float = 0.5, min_ball_radius: int = 2,
                 max_ball_radius: int = 50, min_heatmap_confidence: float = 0.5):
        self.tracknet_path = tracknet_path
        self.detection_threshold = detection_threshold
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius
        self.min_heatmap_confidence = min_heatmap_confidence

        # Filters
        self.exclusion_filter = PolygonExclusionFilter()
        self.inclusion_filter = CourtInclusionFilter()
        self.player_filter = SpatialPlayerFilter()
        self.trajectory_filter = TrajectoryFilter()

        # Filter enable flags
        self.use_exclusion_filter = True
        self.use_inclusion_filter = False
        self.use_player_filter = True
        self.use_trajectory_filter = True

        # Model
        self.device = self._get_device()
        print(f"BallTrackerTrackNet using device: {self.device}")
        self.tracknet = self._load_tracknet()

        # State
        self.background_tensor = None
        self.original_width = None
        self.original_height = None
        self.detection_confidences: List[Optional[float]] = []

        # Stats
        self.stats = {
            'total_raw': 0, 'rejected_confidence': 0, 'rejected_exclusion': 0,
            'rejected_inclusion': 0, 'rejected_player': 0, 'rejected_trajectory': 0, 'accepted': 0,
        }

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_tracknet(self) -> TrackNet:
        print(f"Loading TrackNet from {self.tracknet_path}...")
        checkpoint = torch.load(self.tracknet_path, map_location=self.device, weights_only=False)

        model = TrackNet(in_dim=27, out_dim=8)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        print("TrackNet loaded successfully.")
        return model

    def add_exclusion_zone(self, points: List[Tuple[float, float]], name: str = "zone"):
        self.exclusion_filter.add_polygon_zone(points, name)
        self.use_exclusion_filter = True

    def set_court_polygon(self, points: List[Tuple[float, float]]):
        self.inclusion_filter.set_court_polygon(points)
        self.player_filter.set_court_polygon(points)

    def setup_glass_panel_exclusions(self, left_zone: List[Tuple[float, float]], 
                                      right_zone: List[Tuple[float, float]]):
        """Convenience method to set up left and right glass panel exclusion zones."""
        self.exclusion_filter.add_polygon_zone(left_zone, "left_glass")
        self.exclusion_filter.add_polygon_zone(right_zone, "right_glass")
        self.use_exclusion_filter = True

    def _compute_background(self, frames: List[np.ndarray], num_frames: int = 30) -> torch.Tensor:
        print(f"Computing background from {min(num_frames, len(frames))} frames...")
        sample_frames = frames[:num_frames]
        stack = np.stack(sample_frames, axis=0)
        bg = np.median(stack, axis=0).astype(np.uint8)
        bg_resized = cv2.resize(bg, (self.MODEL_WIDTH, self.MODEL_HEIGHT))
        bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
        bg_tensor = torch.from_numpy(bg_rgb).permute(2, 0, 1).float() / 255.0
        return bg_tensor.to(self.device)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(frame, (self.MODEL_WIDTH, self.MODEL_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    def _heatmap_to_detection(self, heatmap: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        peak_confidence = float(heatmap.max())

        if peak_confidence < self.min_heatmap_confidence:
            self.stats['rejected_confidence'] += 1
            return None

        _, thresh = cv2.threshold(heatmap, self.detection_threshold, 1, cv2.THRESH_BINARY)
        thresh = (thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)

        if radius < self.min_ball_radius or radius > self.max_ball_radius:
            return None

        self.stats['total_raw'] += 1
        return (cx, cy, radius, peak_confidence)

    def detect_frames(self, frames: List[np.ndarray], player_detections: Optional[List[Dict]] = None,
                      read_from_stub: bool = False, stub_path: Optional[str] = None) -> List[Dict]:
        """Detect ball in all frames with filtering."""
        self.stats = {k: 0 for k in self.stats}
        self.trajectory_filter.reset()
        self.detection_confidences = []

        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, tuple):
                        detections, self.detection_confidences = data
                    else:
                        detections = data
                        self.detection_confidences = [None] * len(detections)
                print(f"Loaded {len(detections)} detections from {stub_path}")
                return detections
            except FileNotFoundError:
                print(f"Stub {stub_path} not found, detecting...")

        self.original_height, self.original_width = frames[0].shape[:2]
        scale_x = self.original_width / self.MODEL_WIDTH
        scale_y = self.original_height / self.MODEL_HEIGHT

        self.background_tensor = self._compute_background(frames)

        all_detections = []
        total_frames = len(frames)

        print(f"\nDetecting ball in {total_frames} frames...")
        print(f"  Exclusion filter: {'ON' if self.use_exclusion_filter else 'OFF'} "
              f"({len(self.exclusion_filter.exclusion_zones)} zones)")
        print(f"  Inclusion filter: {'ON' if self.use_inclusion_filter else 'OFF'}")
        print(f"  Player filter: {'ON' if self.use_player_filter and player_detections else 'OFF'}")
        print(f"  Trajectory filter: {'ON' if self.use_trajectory_filter else 'OFF'}")

        frame_idx = 0
        while frame_idx < total_frames:
            batch_tensors = []
            batch_end = min(frame_idx + self.SEQ_LEN, total_frames)
            actual_batch_size = batch_end - frame_idx

            for i in range(frame_idx, batch_end):
                batch_tensors.append(self._preprocess_frame(frames[i]))

            while len(batch_tensors) < self.SEQ_LEN:
                batch_tensors.append(batch_tensors[-1])

            frames_cat = torch.cat(batch_tensors, dim=0)
            input_tensor = torch.cat([frames_cat, self.background_tensor], dim=0)
            input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                preds = self.tracknet(input_tensor)
                preds = preds.squeeze(0)

            for i in range(actual_batch_size):
                current_frame_idx = frame_idx + i
                heatmap = preds[i].cpu().numpy()
                detection = self._heatmap_to_detection(heatmap)

                if detection is None:
                    all_detections.append({})
                    self.detection_confidences.append(None)
                    self.trajectory_filter.record_miss()
                    continue

                cx, cy, radius, confidence = detection
                cx_orig = cx * scale_x
                cy_orig = cy * scale_y
                radius_orig = radius * max(scale_x, scale_y)

                bbox = [cx_orig - radius_orig, cy_orig - radius_orig,
                        cx_orig + radius_orig, cy_orig + radius_orig]

                # Apply filters
                if self.use_exclusion_filter:
                    accepted, reason = self.exclusion_filter.filter_detection(bbox)
                    if not accepted:
                        all_detections.append({})
                        self.detection_confidences.append(None)
                        self.stats['rejected_exclusion'] += 1
                        self.trajectory_filter.record_miss()
                        continue

                if self.use_inclusion_filter:
                    if not self.inclusion_filter.filter_detection(bbox):
                        all_detections.append({})
                        self.detection_confidences.append(None)
                        self.stats['rejected_inclusion'] += 1
                        self.trajectory_filter.record_miss()
                        continue

                if self.use_player_filter and player_detections:
                    if current_frame_idx < len(player_detections):
                        accepted, reason = self.player_filter.filter_detection(
                            bbox, player_detections[current_frame_idx])
                        if not accepted:
                            all_detections.append({})
                            self.detection_confidences.append(None)
                            self.stats['rejected_player'] += 1
                            self.trajectory_filter.record_miss()
                            continue

                if self.use_trajectory_filter:
                    accepted, reason = self.trajectory_filter.filter_detection(current_frame_idx, bbox)
                    if not accepted:
                        all_detections.append({})
                        self.detection_confidences.append(None)
                        self.stats['rejected_trajectory'] += 1
                        continue

                all_detections.append({1: bbox})
                self.detection_confidences.append(confidence)
                self.stats['accepted'] += 1

            frame_idx += actual_batch_size

            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

        self._print_stats(total_frames)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump((all_detections, self.detection_confidences), f)
            print(f"Saved detections to {stub_path}")

        return all_detections

    def _print_stats(self, total_frames: int):
        print(f"\n  === Detection Statistics ===")
        print(f"  Raw detections: {self.stats['total_raw']}")
        print(f"  Rejected - low confidence: {self.stats['rejected_confidence']}")
        print(f"  Rejected - exclusion zone: {self.stats['rejected_exclusion']}")
        print(f"  Rejected - outside court: {self.stats['rejected_inclusion']}")
        print(f"  Rejected - player distance: {self.stats['rejected_player']}")
        print(f"  Rejected - trajectory: {self.stats['rejected_trajectory']}")
        print(f"  ACCEPTED: {self.stats['accepted']} ({100*self.stats['accepted']/total_frames:.1f}%)")

    def draw_bboxes(self, video_frames: List[np.ndarray], ball_detections: List[Dict],
                    color: Tuple[int, int, int] = (0, 255, 0), trail_length: int = 10,
                    show_confidence: bool = True, draw_exclusion_zones: bool = False,
                    draw_court: bool = False) -> List[np.ndarray]:
        output_frames = []
        recent_positions = []

        for frame_idx, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            frame = frame.copy()

            if draw_exclusion_zones:
                frame = self.exclusion_filter.draw_zones(frame)

            if draw_court:
                frame = self.inclusion_filter.draw_court(frame)

            if ball_dict:
                bbox = list(ball_dict.values())[0]

                if any(np.isnan(v) for v in bbox):
                    output_frames.append(frame)
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                recent_positions.append((cx, cy))
                if len(recent_positions) > trail_length:
                    recent_positions.pop(0)

                for i, pos in enumerate(recent_positions[:-1]):
                    alpha = (i + 1) / len(recent_positions)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(frame, pos, 3, trail_color, -1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                if show_confidence and frame_idx < len(self.detection_confidences):
                    conf = self.detection_confidences[frame_idx]
                    label = f"Ball ({conf:.2f})" if conf is not None else "Ball"
                else:
                    label = "Ball"

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            output_frames.append(frame)

        return output_frames


# =============================================================================
# JENS'S STREAMING BALL TRACKER (from simple_inference.py) - For Shot Classification
# =============================================================================

class StreamingBallTracker:
    """
    Jens's TrackNet-based ball tracker using streaming approach.
    Used for shot classification mode.
    """

    def __init__(self, model_path, device, input_wh=(512, 288)):
        self.device = device
        self.w, self.h = input_wh
        self.bg = None
        self.model = None
        self.seq_len = 8
        self.bg_mode = 'concat'

        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                sd = ckpt['model']
                p = ckpt.get('param_dict', {})
                self.seq_len = p.get('seq_len', 8)
                self.bg_mode = p.get('bg_mode', 'concat')
            else:
                sd = ckpt

            in_c = (self.seq_len + 1) * 3 if self.bg_mode == 'concat' else self.seq_len * 3
            self.model = TrackNet(in_c, self.seq_len).to(self.device)
            
            # Handle state dict key mapping
            new_sd = OrderedDict()
            for k, v in sd.items():
                name = k.replace("module.", "")
                new_sd[name] = v
            
            self.model.load_state_dict(new_sd)
            self.model.eval()

            self.buf = deque(maxlen=self.seq_len)
            print(f"StreamingBallTracker loaded successfully on {self.device}")
        except Exception as e:
            print(f"StreamingBallTracker Load Error: {e}")
            self.model = None

    def compute_background(self, frames):
        """Compute background from a list of frames."""
        if not frames:
            return
        indices = np.linspace(0, len(frames) - 1, min(15, len(frames)), dtype=int)
        sampled = [cv2.resize(frames[i], (self.w, self.h)) for i in indices]
        med = np.median(sampled, axis=0).astype(np.uint8)
        self.bg = torch.from_numpy(cv2.cvtColor(med, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        self.bg = self.bg.permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.buf.clear()

    def predict(self, frame):
        """Predict ball heatmap for a single frame (streaming)."""
        if not self.model:
            return None
        rgb = cv2.cvtColor(cv2.resize(frame, (self.w, self.h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.buf.append(torch.from_numpy(rgb).permute(2, 0, 1).to(self.device))
        if len(self.buf) < self.seq_len:
            return None

        x_seq = torch.stack(list(self.buf)).reshape(1, -1, self.h, self.w)
        x = torch.cat([self.bg, x_seq], 1) if self.bg is not None else x_seq

        with torch.no_grad():
            return self.model(x)[0, -1].cpu().numpy()


class SmartBallTracker:
    """Tracks ball with constraints on sudden movement (Jens's implementation)."""

    def __init__(self, fps=30):
        self.last_pos = None
        self.missing_frames = 0
        self.fps = fps
        self.max_jump_dist = 100

    def update(self, detected_pos):
        if detected_pos is None:
            self.missing_frames += 1
            return None

        if self.last_pos is not None:
            if self.missing_frames > 3 * self.fps:
                self.last_pos = detected_pos
                self.missing_frames = 0
                return detected_pos

            dist = np.linalg.norm(np.array(detected_pos) - np.array(self.last_pos))
            allowed_dist = self.max_jump_dist * (self.missing_frames + 1)
            # Accept the position even if it's a jump (for shot attribution)

        self.last_pos = detected_pos
        self.missing_frames = 0
        return detected_pos


# =============================================================================
# JENS'S SHOT CLASSIFICATION MODEL (from simple_inference.py)
# =============================================================================

class EnhancedVideoClassifier(nn.Module):
    """Enhanced classifier combining video features (R(2+1)D-18) with pose keypoints (Bi-LSTM)."""

    def __init__(self, num_classes, pose_input_size=51, pose_hidden_size=256, pretrained_video=True):
        super().__init__()

        self.video_model = video_models.r2plus1d_18(
            weights=video_models.R2Plus1D_18_Weights.DEFAULT if pretrained_video else None
        )
        num_video_ftrs = self.video_model.fc.in_features
        self.video_model.fc = nn.Identity()

        self.pose_lstm = nn.LSTM(
            pose_input_size, pose_hidden_size,
            num_layers=3, batch_first=True, dropout=0.3,
            bidirectional=True
        )
        self.pose_attention = nn.Sequential(
            nn.Linear(pose_hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        fusion_size = num_video_ftrs + (pose_hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, video_x, pose_x):
        video_features = self.video_model(video_x)
        pose_out, _ = self.pose_lstm(pose_x)
        attention_weights = self.pose_attention(pose_out)
        pose_features = torch.sum(pose_out * attention_weights, dim=1)
        combined = torch.cat((video_features, pose_features), dim=1)
        return self.classifier(combined)


class SimpleTracker:
    """Simple IOU-based tracker to maintain player IDs (Jens's implementation)."""

    def __init__(self, max_missed=30):
        self.tracks = {}
        self.next_id = 0
        self.max_missed = max_missed

    def update(self, players):
        if not self.tracks:
            for p in players:
                p['id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': p['bbox_proc'], 'missed': 0}
                self.next_id += 1
            return

        track_ids = list(self.tracks.keys())
        matches = []

        for i, p in enumerate(players):
            for tid in track_ids:
                iou = get_bbox_iou(p['bbox_proc'], self.tracks[tid]['bbox'])
                if iou > 0.1:
                    matches.append((iou, tid, i))

        matches.sort(key=lambda x: x[0], reverse=True)

        assigned_tracks = set()
        assigned_players = set()

        for iou, tid, idx in matches:
            if tid not in assigned_tracks and idx not in assigned_players:
                players[idx]['id'] = tid
                self.tracks[tid]['bbox'] = players[idx]['bbox_proc']
                self.tracks[tid]['missed'] = 0
                assigned_tracks.add(tid)
                assigned_players.add(idx)

        for i, p in enumerate(players):
            if i not in assigned_players:
                p['id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': p['bbox_proc'], 'missed': 0}
                self.next_id += 1

        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]


def load_shot_model(model_path, device):
    """Load Jens's trained shot classification model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    labels_map = checkpoint['labels_map']
    num_classes = len(labels_map)
    config = checkpoint.get('config', {})

    model = EnhancedVideoClassifier(
        num_classes=num_classes,
        pose_input_size=51,
        pose_hidden_size=config.get('pose_hidden_size', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    idx_to_label = checkpoint.get('idx_to_label', {idx: name for name, idx in labels_map.items()})
    if idx_to_label and isinstance(list(idx_to_label.keys())[0], str):
        idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    other_class_idx = checkpoint.get('other_class_idx', labels_map.get('other', -1))
    confidence_threshold = config.get('confidence_threshold', 0.6)

    print(f"Shot model loaded: {model_path}")
    print(f"Classes: {', '.join([idx_to_label[i] for i in range(num_classes)])}")
    print(f"Confidence threshold: {confidence_threshold}")

    return model, idx_to_label, other_class_idx, confidence_threshold


def preprocess_clip(frames, clip_len=32, resolution=(112, 112)):
    """Preprocess video frames for model input."""
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    total = len(frames)
    if total >= clip_len:
        indices = np.linspace(0, total - 1, clip_len, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.pad(indices, (0, clip_len - total), 'edge')

    sampled = frames[indices]
    tensor = sampled.permute(1, 0, 2, 3).float() / 255.0
    tensor = transforms.functional.resize(tensor, resolution)
    tensor = (tensor - norm_mean) / norm_std

    return tensor.unsqueeze(0)


# =============================================================================
# HEATMAP FUNCTIONS (from teammate's heatmap_on_image.py)
# =============================================================================

def load_heatmap_points(csv_path, inplay_only=False, min_conf=None, players=None):
    df = pd.read_csv(csv_path)

    if "x_px" not in df.columns or "y_px" not in df.columns:
        raise SystemExit("CSV must contain x_px and y_px columns.")

    if "confidence" in df.columns and min_conf is not None:
        df = df[df["confidence"] >= min_conf]
    if inplay_only and "in_play" in df.columns:
        df = df[df["in_play"] == 1]
    if players:
        keep = set(players)
        df = df[df["track_id"].isin(keep)]

    pts = df[["x_px", "y_px", "track_id"]].dropna().to_numpy()
    return pts


def make_hist_on_image(pts_xy, img_w, img_h, bins_x, bins_y):
    x = np.clip(pts_xy[:, 0], 0, img_w - 1)
    y = np.clip(pts_xy[:, 1], 0, img_h - 1)

    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[0, img_w], [0, img_h]])
    H = H.T
    return H, (0, img_w, 0, img_h)


def get_court_blue_cmap():
    return LinearSegmentedColormap.from_list(
        "courtblue",
        [(0.0, "#0050ff"), (0.3, "#00a8ff"), (0.6, "#ffff66"), (1.0, "#ff3300")]
    )


def gaussian_blur_heatmap(H, ksize):
    if ksize is None or ksize < 3:
        return H
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    return cv2.GaussianBlur(H.astype(np.float32), (k, k), 0)


def save_heatmap_on_image(H, extent, img, out_png, title, cmap, heat_alpha=0.7, show_axes=False):
    xmin, xmax, ymin, ymax = extent
    img_h, img_w = img.shape[0], img.shape[1]
    aspect = img_w / img_h
    base_height = 8
    fig_width = base_height * aspect
    fig, ax = plt.subplots(figsize=(fig_width, base_height))

    H_plot = np.where(H > 0.3, H, np.nan)

    ax.imshow(img, origin="upper", extent=[xmin, xmax, ymin, ymax], aspect="auto")
    im = ax.imshow(H_plot, origin="upper", extent=[xmin, xmax, ymin, ymax],
                   aspect="auto", cmap=cmap, alpha=heat_alpha)

    plt.colorbar(im, ax=ax, label="Counts")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Pixels (x)")
    ax.set_ylabel("Pixels (y)")

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def generate_heatmap(csv_path, court_img_path, out_png, bins_x=200, bins_y=100,
                     gauss=9, inplay_only=False, min_conf=None, players=None,
                     heat_alpha=0.7, show_axes=False):
    pts = load_heatmap_points(csv_path, inplay_only, min_conf, players)
    if pts.size == 0:
        raise SystemExit("No points after filtering.")

    img = plt.imread(court_img_path)
    img_h, img_w = img.shape[0], img.shape[1]

    H, extent = make_hist_on_image(pts[:, :2], img_w, img_h, bins_x, bins_y)
    Hsmooth = gaussian_blur_heatmap(H, gauss)

    title = "Heatmap on court image"
    if inplay_only:
        title += " • in-play only"

    cmap = get_court_blue_cmap()
    save_heatmap_on_image(Hsmooth, extent, img, out_png, title,
                          cmap=cmap, heat_alpha=heat_alpha, show_axes=not show_axes)
    print(f"Saved heatmap: {out_png}")


# =============================================================================
# STANDALONE PLAYER TRACKER (from teammate's player_tracker.py)
# =============================================================================

def draw_player_annotations(frame, xyxy, ids, pts_px, pts_m=None):
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


def run_standalone_player_tracker(args):
    """Run standalone player tracking with YOLOv8 + ByteTrack."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    cap.release()

    ensure_dir(args.out_video)
    ensure_dir(args.out_csv)

    H = None
    if args.calib_csv:
        try:
            H = load_calib_csv(args.calib_csv)
            print(f"[INFO] Loaded homography from {args.calib_csv}")
        except Exception as e:
            print(f"[WARN] Failed to load {args.calib_csv}: {e}")
    elif args.calib:
        try:
            img_pts = parse_calib_points(args.calib)
            H = build_homography(img_pts, court_w=args.court_w, court_h=args.court_h)
            print("[INFO] Loaded homography from 4-corner string.")
        except Exception as e:
            print(f"[WARN] Failed to parse --calib: {e}")

    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    rows = []
    frame_idx = -1

    track_kwargs = dict(
        source=args.video, stream=True, conf=args.conf,
        classes=[0], tracker=args.tracker, device=args.device
    )
    if args.imgsz:
        track_kwargs["imgsz"] = args.imgsz

    print(f"[INFO] Running player tracking on {args.video}...")

    for result in model.track(**track_kwargs):
        frame_idx += 1
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
            in_court = (xm >= 0.0) & (xm <= args.court_w) & (ym >= 0.0) & (ym <= args.court_h)
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

            rows.append((frame_idx, int(ids[i]), float(feet_px[i, 0]), float(feet_px[i, 1]),
                         xm_val, ym_val, float(confs[i])))

        draw_player_annotations(frame, xyxy, ids, feet_px, feet_m)
        writer.write(frame)

    writer.release()

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x_px", "y_px", "x_m", "y_m", "confidence"])
    df.to_csv(args.out_csv, index=False)

    print("Done!")
    print(f"- Annotated video: {args.out_video}")
    print(f"- Tracks CSV:      {args.out_csv}")


# =============================================================================
# JENS'S SHOT CLASSIFICATION INFERENCE (from simple_inference.py)
# =============================================================================

def run_shot_inference(args):
    """Run Jens's shot classification inference with streaming ball tracker."""
    from ultralytics import YOLO

    colors = {
        'forehand': (0, 255, 0), 'backhand': (255, 0, 0), 'serve': (0, 165, 255),
        'overhead': (0, 255, 255), 'lob': (255, 0, 255), 'other': (128, 128, 128)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("\nLoading shot classifier...")
    model, idx_to_label, other_class_idx, model_conf_threshold = load_shot_model(args.shot_model, device)

    print("\nLoading YOLO pose model...")
    pose_model = YOLO(args.yolo_pose)

    confidence_threshold = args.confidence_threshold or model_conf_threshold

    # Open video
    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc_width, proc_height = args.processing_resolution
    yolo_width, yolo_height = args.yolo_resolution

    scale_orig_to_yolo_x = yolo_width / orig_width
    scale_orig_to_yolo_y = yolo_height / orig_height

    print(f"Video: {total_frames} frames, {fps:.1f} FPS, {orig_width}x{orig_height}")
    print(f"YOLO inference at: {yolo_width}x{yolo_height}")

    # Initialize streaming ball tracker
    ball_tracker = None
    smart_ball_tracker = None
    if args.tracknet and os.path.exists(args.tracknet):
        print(f"Loading TrackNet from {args.tracknet}...")
        ball_tracker = StreamingBallTracker(args.tracknet, device, input_wh=(512, 288))
        smart_ball_tracker = SmartBallTracker(fps=fps)

        # Compute background
        print("Computing background for ball tracking...")
        bg_frames = []
        for i in range(min(30, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * (total_frames // 30))
            ret, f = cap.read()
            if ret:
                bg_frames.append(f)
        ball_tracker.compute_background(bg_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        print("Warning: TrackNet model not found. Ball tracking disabled.")

    # Setup output
    output_path = args.output or f"{os.path.splitext(os.path.basename(args.input))[0]}_labeled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    # Buffers
    frame_buffer = deque(maxlen=args.window_size)
    pose_buffer = deque(maxlen=args.window_size)
    output_buffer = deque(maxlen=args.lookahead + args.window_size)
    predictions = {}
    frame_ball_map = {}
    frame_players_map = {}
    frame_active_map = {}

    tracker = SimpleTracker()
    player_stats = {}
    player_positions = {}
    prev_players = None
    last_shot_global = -9999

    def draw_stats(img, stats, positions):
        """Draw aggregated stats overlay."""
        h, w = img.shape[:2]
        center_x = w / 2
        left_stats, right_stats = {}, {}

        for pid, p_stats in stats.items():
            pos = positions.get(pid, center_x)
            target = left_stats if pos < center_x else right_stats
            for label, count in p_stats.items():
                target[label] = target.get(label, 0) + count

        def draw_side(side_stats, title, start_x, start_y):
            overlay = img.copy()
            box_h = 30 + (len(side_stats) + 1) * 20 if side_stats else 60
            cv2.rectangle(overlay, (start_x-10, start_y-30), (start_x+200, start_y + box_h - 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(img, title, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_off = 25
            total = sum(side_stats.values())
            cv2.putText(img, f"Total: {total}", (start_x, start_y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_off += 20
            for label, count in sorted(side_stats.items(), key=lambda x: x[1], reverse=True):
                c = colors.get(label, (255, 255, 255))
                cv2.putText(img, f"{label}: {count}", (start_x, start_y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
                y_off += 20

        draw_side(left_stats, "Player Left", 20, 40)
        draw_side(right_stats, "Player Right", w - 220, 40)

    print("Processing video...")
    pbar = tqdm(total=total_frames)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_yolo = cv2.resize(frame, (yolo_width, yolo_height))
        frame_rgb = cv2.cvtColor(frame_yolo, cv2.COLOR_BGR2RGB)

        results = pose_model(frame_yolo, verbose=False)

        players = []
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i in range(len(keypoints_data)):
                kp = keypoints_data[i]
                if np.mean(kp[:, 2]) > 0.3:
                    scaled_bbox = boxes[i].copy()
                    scaled_bbox[0] /= scale_orig_to_yolo_x
                    scaled_bbox[1] /= scale_orig_to_yolo_y
                    scaled_bbox[2] /= scale_orig_to_yolo_x
                    scaled_bbox[3] /= scale_orig_to_yolo_y

                    kp_proc = kp.copy()
                    if np.max(kp_proc) > 1.0:
                        kp_proc[:, 0] /= yolo_width
                        kp_proc[:, 1] /= yolo_height

                    bbox_proc = boxes[i].copy()

                    players.append({
                        'keypoints': kp_proc,
                        'bbox': scaled_bbox,
                        'bbox_proc': bbox_proc,
                        'raw_kp': kp
                    })

        tracker.update(players)

        for p in players:
            pid = p.get('id')
            if pid is not None:
                bbox = p['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                player_positions[pid] = cx

        # Ball tracking (streaming)
        ball_pos = None
        if ball_tracker:
            hm = ball_tracker.predict(frame)
            detected_ball = None
            if hm is not None:
                _, th = cv2.threshold(hm, 0.5, 1, 0)
                ctrs, _ = cv2.findContours((th * 255).astype(np.uint8), 0, 2)
                if ctrs:
                    c = max(ctrs, key=cv2.contourArea)
                    (cx, cy), _ = cv2.minEnclosingCircle(c)
                    detected_ball = (int(cx * (orig_width / 512)), int(cy * (orig_height / 288)))

            ball_pos = smart_ball_tracker.update(detected_ball) if smart_ball_tracker else detected_ball

        frame_ball_map[frame_idx] = ball_pos
        frame_players_map[frame_idx] = players

        # Find active player
        active_idx = None
        active_id = None

        if len(players) > 0:
            if len(players) == 1:
                active_idx = 0
            else:
                motion_scores = []
                for p in players:
                    prev_kp = None
                    if prev_players:
                        for pp in prev_players:
                            if pp.get('id') == p.get('id'):
                                prev_kp = pp['keypoints']
                                break
                    motion_scores.append(get_motion_score(p['keypoints'], prev_kp))
                active_idx = np.argmax(motion_scores)

        if active_idx is not None:
            active_kp = players[active_idx]['keypoints'].flatten()
            active_bbox = players[active_idx]['bbox']
            active_id = players[active_idx].get('id')
        else:
            active_kp = np.zeros(51)
            active_bbox = None
            active_id = None

        frame_active_map[frame_idx] = active_id

        frame_buffer.append(torch.from_numpy(frame_rgb))
        pose_buffer.append(torch.from_numpy(active_kp))

        output_buffer.append({
            'frame': frame.copy(),
            'players': players,
            'active_bbox': active_bbox,
            'frame_idx': frame_idx,
            'ball_pos': ball_pos
        })

        # Run prediction
        if len(frame_buffer) == args.window_size and frame_idx % args.stride == 0:
            clip = torch.stack(list(frame_buffer)).permute(0, 3, 1, 2)
            clip_tensor = preprocess_clip(clip, clip_len=args.window_size,
                                          resolution=(args.classifier_resolution, args.classifier_resolution)).to(device)
            pose_tensor = torch.stack(list(pose_buffer)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                outputs = model(clip_tensor, pose_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)

                pred_label = idx_to_label[pred_idx.item()]
                conf = confidence.item()

                if pred_label != 'other':
                    print(f"Frame {frame_idx}: Raw: {pred_label} | Conf: {conf:.4f}")

                if conf < confidence_threshold and other_class_idx >= 0:
                    pred_label = idx_to_label.get(other_class_idx, 'other')

                if pred_label != "other" and conf > args.threshold:
                    shot_frame_idx = frame_idx - args.window_size // 2

                    if shot_frame_idx - last_shot_global >= fps:
                        shooter_id = None

                        shot_ball_pos = frame_ball_map.get(shot_frame_idx)
                        shot_players = frame_players_map.get(shot_frame_idx)

                        if shot_ball_pos and shot_players:
                            min_dist = float('inf')
                            for p in shot_players:
                                bbox = p['bbox']
                                p_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                                dist = np.linalg.norm(np.array(shot_ball_pos) - np.array(p_center))
                                if dist < min_dist:
                                    min_dist = dist
                                    shooter_id = p.get('id')

                        if shooter_id is None:
                            shooter_id = frame_active_map.get(shot_frame_idx)

                        if shooter_id is not None:
                            last_shot_global = shot_frame_idx

                            if shooter_id not in player_stats:
                                player_stats[shooter_id] = {}
                            player_stats[shooter_id][pred_label] = player_stats[shooter_id].get(pred_label, 0) + 1

                            display_duration = args.stride * 3
                            for offset in range(display_duration):
                                target_idx = shot_frame_idx + offset
                                if target_idx >= 0:
                                    predictions[target_idx] = (pred_label, shooter_id)

        # Write frames from buffer
        while len(output_buffer) > args.lookahead:
            buf_entry = output_buffer.popleft()
            display_frame = buf_entry['frame'].copy()
            buf_players = buf_entry['players']
            buf_frame_idx = buf_entry['frame_idx']

            for p in buf_players:
                x1, y1, x2, y2 = map(int, p['bbox'])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

            pred_data = predictions.get(buf_frame_idx)
            current_prediction = None
            shooter_id = None

            if pred_data:
                if isinstance(pred_data, tuple):
                    current_prediction, shooter_id = pred_data
                else:
                    current_prediction = pred_data

            if buf_frame_idx in predictions:
                del predictions[buf_frame_idx]

            ball_pos = buf_entry.get('ball_pos')
            if ball_pos:
                cv2.circle(display_frame, ball_pos, 5, (0, 0, 255), -1)

            if current_prediction is not None:
                color = colors.get(current_prediction, (255, 255, 255))
                label = current_prediction.upper()

                for p in buf_players:
                    if p.get('id') == shooter_id:
                        x1, y1, x2, y2 = map(int, p['bbox'])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (tw, th), _ = cv2.getTextSize(label, font, 1.0, 2)
                        cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(display_frame, label, (x1 + 5, y1 - 5), font, 1.0, (255, 255, 255), 2)
                        break

            draw_stats(display_frame, player_stats, player_positions)
            out.write(display_frame)

        prev_players = players
        frame_idx += 1
        pbar.update(1)

    # Flush remaining frames
    while output_buffer:
        buf_entry = output_buffer.popleft()
        display_frame = buf_entry['frame'].copy()

        for p in buf_entry['players']:
            x1, y1, x2, y2 = map(int, p['bbox'])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

        ball_pos = buf_entry.get('ball_pos')
        if ball_pos:
            cv2.circle(display_frame, ball_pos, 5, (0, 0, 255), -1)

        draw_stats(display_frame, player_stats, player_positions)
        out.write(display_frame)

    cap.release()
    out.release()
    pbar.close()

    print(f"\nOutput saved to: {output_path}")
    print(f"\nPlayer Stats: {player_stats}")


# =============================================================================
# YOUR MAIN PIPELINE (jens_inference.py)
# =============================================================================

def run_main_pipeline(args):
    """Run your full padel analysis pipeline with BallTrackerTrackNet."""
    from utils import read_video, save_video, load_court_config, calibrate_exclusion_zones
    from trackers import PlayerTracker, RallyTracker

    print("=" * 60)
    print("PADEL MATCH ANALYSIS - MAIN PIPELINE")
    print("=" * 60)

    court_config = load_court_config(args.court_number)

    LEFT_EXCLUSION_ZONE = court_config['LEFT_EXCLUSION_ZONE']
    RIGHT_EXCLUSION_ZONE = court_config['RIGHT_EXCLUSION_ZONE']
    COURT_POLYGON = court_config['COURT_POLYGON']

    video_frames = read_video(args.input_video)
    frame_height, frame_width = video_frames[0].shape[:2]
    print(f"\nVideo: {len(video_frames)} frames, {frame_width}x{frame_height}, {args.fps} fps")

    # Player Detection
    print("\n" + "=" * 60)
    print("STEP 1: Player Detection (YOLO)")
    print("=" * 60)

    player_tracker = PlayerTracker(model_path=args.player_model)
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=args.player_stub
    )

    # Ball Detection with YOUR BallTrackerTrackNet
    print("\n" + "=" * 60)
    print("STEP 2: Ball Detection (TrackNet with Filters)")
    print("=" * 60)

    ball_tracker = BallTrackerTrackNet(
        tracknet_path=args.tracknet_model,
        detection_threshold=args.ball_threshold,
        min_heatmap_confidence=args.min_heatmap_conf,
    )

    ball_tracker.add_exclusion_zone(LEFT_EXCLUSION_ZONE, "left_glass")
    ball_tracker.add_exclusion_zone(RIGHT_EXCLUSION_ZONE, "right_glass")
    ball_tracker.set_court_polygon(COURT_POLYGON)

    ball_tracker.use_exclusion_filter = True
    ball_tracker.use_inclusion_filter = False
    ball_tracker.use_player_filter = True
    ball_tracker.use_trajectory_filter = True

    ball_detections = ball_tracker.detect_frames(
        video_frames,
        player_detections=player_detections,
        read_from_stub=args.use_stubs,
        stub_path=args.ball_stub,
    )

    detected_count = sum(1 for d in ball_detections if d)
    print(f"\nBall detected in {detected_count}/{len(video_frames)} frames "
          f"({100 * detected_count / len(video_frames):.1f}%)")

    # Rally Detection
    print("\n" + "=" * 60)
    print("STEP 3: Rally Detection")
    print("=" * 60)

    rally_tracker = RallyTracker(
        fps=args.fps,
        frame_height=frame_height,
        min_velocity=args.min_velocity,
        serve_velocity=args.serve_velocity,
        base_gap_during_rally=args.base_gap_during_rally,
        base_rally_end_gap=args.base_rally_end_gap,
        max_gap_extension=args.max_gap_extension,
        min_rally_frames=args.min_rally_frames,
        min_rally_distance=args.min_rally_distance,
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
                  f"({rally.duration_seconds(args.fps):.1f}s)")

    rally_csv = args.rally_csv or 'output_videos/rallies.csv'
    rally_tracker.export_rallies_csv(rally_csv)

    # Output Video
    print("\n" + "=" * 60)
    print("STEP 4: Generating Output Video")
    print("=" * 60)

    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    output_frames = ball_tracker.draw_bboxes(
        output_frames,
        ball_detections,
        trail_length=args.trail_length,
        show_confidence=True,
        draw_exclusion_zones=args.draw_exclusion_zones,
        draw_court=args.draw_court,
    )

    output_frames = rally_tracker.draw_overlay(output_frames)

    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 221), 2)

    save_video(output_frames, args.output_video, args.fps)
    print(f"\nOutput saved to: {args.output_video}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merged Padel Match Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", default="full",
                        choices=["full", "shots", "player_track", "heatmap", "calibrate"],
                        help="Pipeline mode: full (your pipeline), shots (Jens's), player_track, heatmap, calibrate")

    # Main Pipeline (mode=full) - YOUR jens_inference.py
    main_group = parser.add_argument_group("Main Pipeline Options (your jens_inference.py)")
    main_group.add_argument("--input_video", default="input_videos/video4_trimmed_1-1080p30.mp4")
    main_group.add_argument("--output_video", default="output_videos/output_merged.avi")
    main_group.add_argument("--tracknet_model", default="models/TrackNet_best.pt")
    main_group.add_argument("--player_model", default="models/yolov8x.pt")
    main_group.add_argument("--player_stub", default="tracker_stubs/player_detections5.pkl")
    main_group.add_argument("--ball_stub", default="tracker_stubs/ball_detections_v5.pkl")
    main_group.add_argument("--fps", type=float, default=30.0)
    main_group.add_argument("--court_number", type=int, default=11)
    main_group.add_argument("--use_stubs", action="store_true", default=True)
    main_group.add_argument("--no_stubs", dest="use_stubs", action="store_false")
    main_group.add_argument("--ball_threshold", type=float, default=0.5)
    main_group.add_argument("--min_heatmap_conf", type=float, default=0.5)
    main_group.add_argument("--min_velocity", type=float, default=3.0)
    main_group.add_argument("--serve_velocity", type=float, default=6.0)
    main_group.add_argument("--base_gap_during_rally", type=int, default=40)
    main_group.add_argument("--base_rally_end_gap", type=int, default=70)
    main_group.add_argument("--max_gap_extension", type=int, default=60)
    main_group.add_argument("--min_rally_frames", type=int, default=45)
    main_group.add_argument("--min_rally_distance", type=int, default=80)
    main_group.add_argument("--rally_csv", default=None)
    main_group.add_argument("--trail_length", type=int, default=10)
    main_group.add_argument("--draw_exclusion_zones", action="store_true", default=True)
    main_group.add_argument("--draw_court", action="store_true", default=False)

    # Shot Classification (mode=shots) - JENS's simple_inference.py
    shot_group = parser.add_argument_group("Shot Classification Options (Jens's simple_inference.py)")
    shot_group.add_argument("--input", type=str, help="Input video for shot classification")
    shot_group.add_argument("--output", type=str, default=None, help="Output video path")
    shot_group.add_argument("--shot_model", default="models/best_model.pth")
    shot_group.add_argument("--yolo_pose", default="models/yolov8n-pose.pt")
    shot_group.add_argument("--tracknet", default="models/TrackNet_best.pt")
    shot_group.add_argument("--threshold", type=float, default=0.5, help="Display confidence threshold")
    shot_group.add_argument("--confidence_threshold", type=float, default=None,
                            help="Below this confidence, default to 'other' class")
    shot_group.add_argument("--window_size", type=int, default=32, help="Sliding window size")
    shot_group.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    shot_group.add_argument("--processing_resolution", type=int, nargs=2, default=[1280, 720])
    shot_group.add_argument("--yolo_resolution", type=int, nargs=2, default=[640, 360])
    shot_group.add_argument("--classifier_resolution", type=int, default=128)
    shot_group.add_argument("--lookahead", type=int, default=16)

    # Standalone Player Tracker (mode=player_track) - TEAMMATE's player_tracker.py
    pt_group = parser.add_argument_group("Standalone Player Tracker Options (teammate's player_tracker.py)")
    pt_group.add_argument("--video", default=None)
    pt_group.add_argument("--out_video", default="outputs/padel_annotated.mp4")
    pt_group.add_argument("--out_csv", default="outputs/padel_tracks.csv")
    pt_group.add_argument("--model", default="yolov8s.pt")
    pt_group.add_argument("--conf", type=float, default=0.25)
    pt_group.add_argument("--imgsz", type=int, default=None)
    pt_group.add_argument("--device", default=None)
    pt_group.add_argument("--tracker", default="bytetrack.yaml")

    # Calibration
    calib_group = parser.add_argument_group("Calibration Options")
    calib_group.add_argument("--calib", default=None)
    calib_group.add_argument("--calib_csv", default=None)
    calib_group.add_argument("--court_w", type=float, default=20.0)
    calib_group.add_argument("--court_h", type=float, default=10.0)

    # Heatmap (mode=heatmap) - TEAMMATE's heatmap_on_image.py
    heat_group = parser.add_argument_group("Heatmap Options (teammate's heatmap_on_image.py)")
    heat_group.add_argument("--csv", default=None)
    heat_group.add_argument("--court_img", default=None)
    heat_group.add_argument("--heatmap_output", default="heatmap_on_image.png")
    heat_group.add_argument("--heatmap_bins_x", type=int, default=200)
    heat_group.add_argument("--heatmap_bins_y", type=int, default=100)
    heat_group.add_argument("--heatmap_gauss", type=int, default=9)
    heat_group.add_argument("--heatmap_inplay_only", action="store_true")
    heat_group.add_argument("--heatmap_min_conf", type=float, default=None)
    heat_group.add_argument("--heatmap_players", default="")
    heat_group.add_argument("--heatmap_alpha", type=float, default=0.7)
    heat_group.add_argument("--heatmap_show_axes", action="store_true")

    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    args = parse_args()

    if args.mode == "calibrate":
        from utils import calibrate_exclusion_zones
        video_path = args.video or args.input_video
        calibrate_exclusion_zones(video_path)

    elif args.mode == "heatmap":
        if not args.csv or not args.court_img:
            print("Error: --csv and --court_img required for heatmap mode")
            sys.exit(1)

        players = [int(s) for s in args.heatmap_players.split(",") if s.strip().isdigit()] if args.heatmap_players else None

        generate_heatmap(
            csv_path=args.csv,
            court_img_path=args.court_img,
            out_png=args.heatmap_output,
            bins_x=args.heatmap_bins_x,
            bins_y=args.heatmap_bins_y,
            gauss=args.heatmap_gauss,
            inplay_only=args.heatmap_inplay_only,
            min_conf=args.heatmap_min_conf,
            players=players,
            heat_alpha=args.heatmap_alpha,
            show_axes=args.heatmap_show_axes,
        )

    elif args.mode == "player_track":
        if not args.video:
            print("Error: --video required for player_track mode")
            sys.exit(1)
        run_standalone_player_tracker(args)

    elif args.mode == "shots":
        if not args.input:
            print("Error: --input required for shots mode")
            sys.exit(1)
        run_shot_inference(args)

    else:  # mode == "full"
        run_main_pipeline(args)


if __name__ == "__main__":
    main()