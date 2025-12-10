#!/usr/bin/env python3
"""
Padel Match Analysis Pipeline
==================================================

Pipeline Order (Full Mode):
1. Shot Classification - ball/shot detection with pose
2. Player Detection (YOLO)
3. Ball Detection (TrackNet with Filters) 
4. Rally Detection
5. Player Position Export & Heatmap Generation
6. Generate Combined Output Video

Features:
- Output files named based on input video (e.g., video2.mp4 -> video2_output.avi)
- Auto-generates heatmap using first frame as court image
- Combines all overlays (players, ball, shots, rallies) into single video
- Court calibration loaded from JSON with homography support

Usage Examples:
---------------
# Full pipeline with shot classification
python merged_padel_inference.py --mode full --input_video video.mp4

# Full pipeline with heatmap generation
python merged_padel_inference.py --mode full --input_video video.mp4 --generate_heatmap

# Shot classification only
python merged_padel_inference.py --mode shots --input video.mp4

# Standalone player tracking with auto heatmap
python merged_padel_inference.py --mode player_track --video input.mp4 --auto_heatmap

# Calibrate a court (exclusion zones + calibration points)
python merged_padel_inference.py --mode calibrate --video input.mp4 --court_number 9
"""

import argparse
import os
import sys
import csv
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
from ultralytics import YOLO


# Import from centralized ball tracker module
from trackers import (
    BallTrackerTrackNet,
    StreamingBallTracker,
    SmartBallTracker
)

# Import calibration utilities
from utils import (
    calibrate_court,
    load_court_config,
    load_court_calibration
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Ensure directory exists for a file path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def get_output_basename(input_video: str) -> str:
    """Get output base name from input video path."""
    basename = os.path.basename(input_video)
    name_without_ext = os.path.splitext(basename)[0]
    return name_without_ext


def get_output_paths(input_video: str, output_dir: str = "output_videos") -> Dict[str, str]:
    """Generate all output file paths based on input video name."""
    basename = get_output_basename(input_video)
    ensure_dir(os.path.join(output_dir, "dummy.txt"))
    
    return {
        'video': os.path.join(output_dir, f"{basename}_output.avi"),
        'rally_csv': os.path.join(output_dir, f"{basename}_rallies.csv"),
        'player_csv': os.path.join(output_dir, f"{basename}_player_positions.csv"),
        'heatmap': os.path.join(output_dir, f"{basename}_heatmap.png"),
        'court_frame': os.path.join(output_dir, f"{basename}_court_frame.png"),
        'shots_csv': os.path.join(output_dir, f"{basename}_shots.csv"),
    }


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
# SHOT CLASSIFICATION MODEL
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
    """Simple IOU-based tracker to maintain player IDs."""

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
    """Load trained shot classification model."""
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
# HEATMAP FUNCTIONS
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
    ensure_dir(out_png)
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
        print("Warning: No points after filtering for heatmap.")
        return

    img = plt.imread(court_img_path)
    img_h, img_w = img.shape[0], img.shape[1]

    H, extent = make_hist_on_image(pts[:, :2], img_w, img_h, bins_x, bins_y)
    Hsmooth = gaussian_blur_heatmap(H, gauss)

    title = "Player Position Heatmap"
    if inplay_only:
        title += " (in-play only)"

    cmap = get_court_blue_cmap()
    save_heatmap_on_image(Hsmooth, extent, img, out_png, title,
                          cmap=cmap, heat_alpha=heat_alpha, show_axes=not show_axes)
    print(f"Saved heatmap: {out_png}")


# =============================================================================
# PLAYER POSITION CSV EXPORT
# =============================================================================

def export_player_positions_csv(player_detections, output_csv, fps=30.0, homography=None, 
                                 court_w=20.0, court_h=10.0, rally_frames=None):
    """Export player detections to CSV with optional homography for meter conversion."""
    ensure_dir(output_csv)
    
    rows = []
    for frame_idx, detections in enumerate(player_detections):
        if not detections:
            continue
            
        for track_id, bbox in detections.items():
            x_px = (bbox[0] + bbox[2]) / 2.0
            y_px = bbox[3]
            
            x_m, y_m = np.nan, np.nan
            if homography is not None:
                pts = np.array([[x_px, y_px]], dtype=np.float32)
                pts_m = to_meters(homography, pts)
                if pts_m.size > 0 and np.all(np.isfinite(pts_m[0])):
                    x_m, y_m = pts_m[0]
                    if not (0 <= x_m <= court_w and 0 <= y_m <= court_h):
                        x_m, y_m = np.nan, np.nan
            
            in_play = 1 if rally_frames and frame_idx in rally_frames else 0
            conf = 1.0
            
            rows.append({
                'frame': frame_idx,
                'track_id': track_id,
                'x_px': x_px,
                'y_px': y_px,
                'x_m': x_m,
                'y_m': y_m,
                'confidence': conf,
                'in_play': in_play
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Exported player positions to: {output_csv}")
    return output_csv


# =============================================================================
# SHOT CLASSIFICATION PROCESSOR (for integration into full pipeline)
# =============================================================================

class ShotClassificationProcessor:
    """
    Processes video for shot classification and returns shot events.
    Can be integrated into the full pipeline.
    """
    
    SHOT_COLORS = {
        'forehand': (0, 255, 0), 'backhand': (255, 0, 0), 'serve': (0, 165, 255),
        'overhead': (0, 255, 255), 'lob': (255, 0, 255), 'other': (128, 128, 128)
    }
    
    def __init__(self, shot_model_path: str, yolo_pose_path: str, tracknet_path: str,
                 device: torch.device = None, confidence_threshold: float = None,
                 window_size: int = 32, stride: int = 8, classifier_resolution: int = 128):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.window_size = window_size
        self.stride = stride
        self.classifier_resolution = classifier_resolution
        
        # Load models
        print("\nLoading shot classifier...")
        self.model, self.idx_to_label, self.other_class_idx, model_conf = load_shot_model(shot_model_path, self.device)
        self.confidence_threshold = confidence_threshold or model_conf
        
        print("\nLoading YOLO pose model...")
        self.pose_model = YOLO(yolo_pose_path)
        
        # Ball tracker
        self.ball_tracker = None
        self.smart_ball_tracker = None
        if tracknet_path and os.path.exists(tracknet_path):
            print(f"Loading TrackNet for shot classification from {tracknet_path}...")
            self.ball_tracker = StreamingBallTracker(tracknet_path, self.device, input_wh=(512, 288))
            self.smart_ball_tracker = SmartBallTracker()
        
        self.tracker = SimpleTracker()
        self.shot_events = []  # List of (frame_idx, shot_type, player_id, confidence)
        self.player_stats = {}
        self.player_positions = {}
    
    def process_video(self, video_frames: List[np.ndarray], fps: float = 30.0,
                      yolo_resolution: Tuple[int, int] = (640, 360)) -> Dict:
        """
        Process video frames and return shot events and ball positions.
        
        Returns:
            Dict with 'shot_events', 'ball_positions', 'player_stats', 'frame_predictions'
        """
        print(f"\nProcessing {len(video_frames)} frames for shot classification...")
        
        orig_height, orig_width = video_frames[0].shape[:2]
        yolo_width, yolo_height = yolo_resolution
        scale_orig_to_yolo_x = yolo_width / orig_width
        scale_orig_to_yolo_y = yolo_height / orig_height
        
        # Compute background for ball tracker
        if self.ball_tracker:
            self.ball_tracker.compute_background(video_frames)
        
        frame_buffer = deque(maxlen=self.window_size)
        pose_buffer = deque(maxlen=self.window_size)
        
        frame_ball_map = {}
        frame_players_map = {}
        frame_active_map = {}
        predictions = {}
        
        prev_players = None
        last_shot_global = -9999
        
        for frame_idx, frame in enumerate(tqdm(video_frames, desc="Shot classification")):
            frame_yolo = cv2.resize(frame, (yolo_width, yolo_height))
            frame_rgb = cv2.cvtColor(frame_yolo, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = self.pose_model(frame_yolo, verbose=False)
            
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
            
            self.tracker.update(players)
            
            for p in players:
                pid = p.get('id')
                if pid is not None:
                    bbox = p['bbox']
                    cx = (bbox[0] + bbox[2]) / 2
                    self.player_positions[pid] = cx
            
            # Ball tracking
            ball_pos = None
            if self.ball_tracker:
                hm = self.ball_tracker.predict(frame)
                detected_ball = None
                if hm is not None:
                    _, th = cv2.threshold(hm, 0.5, 1, 0)
                    ctrs, _ = cv2.findContours((th * 255).astype(np.uint8), 0, 2)
                    if ctrs:
                        c = max(ctrs, key=cv2.contourArea)
                        (cx, cy), _ = cv2.minEnclosingCircle(c)
                        detected_ball = (int(cx * (orig_width / 512)), int(cy * (orig_height / 288)))
                
                ball_pos = self.smart_ball_tracker.update(detected_ball) if self.smart_ball_tracker else detected_ball
            
            frame_ball_map[frame_idx] = ball_pos
            frame_players_map[frame_idx] = players
            
            # Determine active player
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
                active_id = players[active_idx].get('id')
            else:
                active_kp = np.zeros(51)
            
            frame_active_map[frame_idx] = active_id
            
            frame_buffer.append(torch.from_numpy(frame_rgb))
            pose_buffer.append(torch.from_numpy(active_kp))
            
            # Classify when buffer is full
            if len(frame_buffer) == self.window_size and frame_idx % self.stride == 0:
                clip = torch.stack(list(frame_buffer)).permute(0, 3, 1, 2)
                clip_tensor = preprocess_clip(clip, clip_len=self.window_size,
                                              resolution=(self.classifier_resolution, self.classifier_resolution)).to(self.device)
                pose_tensor = torch.stack(list(pose_buffer)).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(clip_tensor, pose_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    
                    pred_label = self.idx_to_label[pred_idx.item()]
                    conf = confidence.item()
                    
                    if conf < self.confidence_threshold and self.other_class_idx >= 0:
                        pred_label = self.idx_to_label.get(self.other_class_idx, 'other')
                    
                    if pred_label != "other" and conf > 0.5:
                        shot_frame_idx = frame_idx - self.window_size // 2
                        
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
                                
                                if shooter_id not in self.player_stats:
                                    self.player_stats[shooter_id] = {}
                                self.player_stats[shooter_id][pred_label] = self.player_stats[shooter_id].get(pred_label, 0) + 1
                                
                                self.shot_events.append({
                                    'frame': shot_frame_idx,
                                    'shot_type': pred_label,
                                    'player_id': shooter_id,
                                    'confidence': conf
                                })
                                
                                # Store prediction for display
                                display_duration = self.stride * 3
                                for offset in range(display_duration):
                                    target_idx = shot_frame_idx + offset
                                    if target_idx >= 0:
                                        predictions[target_idx] = (pred_label, shooter_id, conf)
            
            prev_players = players
        
        print(f"Detected {len(self.shot_events)} shots")
        
        return {
            'shot_events': self.shot_events,
            'ball_positions': frame_ball_map,
            'player_stats': self.player_stats,
            'frame_predictions': predictions,
            'frame_players': frame_players_map,
        }
    
    def draw_shot_overlay(self, frames: List[np.ndarray], shot_data: Dict) -> List[np.ndarray]:
        """Draw shot classification overlay on frames."""
        output_frames = []
        predictions = shot_data.get('frame_predictions', {})
        ball_positions = shot_data.get('ball_positions', {})
        frame_players = shot_data.get('frame_players', {})
        
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            
            # Draw ball position
            ball_pos = ball_positions.get(frame_idx)
            if ball_pos:
                cv2.circle(frame, ball_pos, 5, (0, 0, 255), -1)
            
            # Draw shot prediction
            pred_data = predictions.get(frame_idx)
            if pred_data:
                shot_type, shooter_id, conf = pred_data
                color = self.SHOT_COLORS.get(shot_type, (255, 255, 255))
                
                # Find shooter and draw label
                players = frame_players.get(frame_idx, [])
                for p in players:
                    if p.get('id') == shooter_id:
                        bbox = p['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        label = shot_type.upper()
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (tw, th), _ = cv2.getTextSize(label, font, 1.0, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, 1.0, (255, 255, 255), 2)
                        break
            
            output_frames.append(frame)
        
        # Draw stats
        self._draw_stats_on_frames(output_frames, self.player_stats, self.player_positions)
        
        return output_frames
    
    def _draw_stats_on_frames(self, frames: List[np.ndarray], stats: Dict, positions: Dict):
        """Draw player stats overlay."""
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        center_x = w / 2
        
        for frame in frames:
            left_stats, right_stats = {}, {}
            
            for pid, p_stats in stats.items():
                pos = positions.get(pid, center_x)
                target = left_stats if pos < center_x else right_stats
                for label, count in p_stats.items():
                    target[label] = target.get(label, 0) + count
            
            def draw_side(side_stats, title, start_x, start_y):
                overlay = frame.copy()
                box_h = 30 + (len(side_stats) + 1) * 20 if side_stats else 60
                cv2.rectangle(overlay, (start_x-10, start_y-30), (start_x+200, start_y + box_h - 30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, title, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_off = 25
                total = sum(side_stats.values())
                cv2.putText(frame, f"Total: {total}", (start_x, start_y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_off += 20
                for label, count in sorted(side_stats.items(), key=lambda x: x[1], reverse=True):
                    c = self.SHOT_COLORS.get(label, (255, 255, 255))
                    cv2.putText(frame, f"{label}: {count}", (start_x, start_y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
                    y_off += 20
            
            draw_side(left_stats, "Player Left", 20, 40)
            draw_side(right_stats, "Player Right", w - 220, 40)
    
    def export_shots_csv(self, output_path: str):
        """Export shot events to CSV."""
        ensure_dir(output_path)
        df = pd.DataFrame(self.shot_events)
        df.to_csv(output_path, index=False)
        print(f"Exported shots to: {output_path}")


# =============================================================================
# STANDALONE PLAYER TRACKER
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get output paths based on input video name
    output_paths = get_output_paths(args.video)
    out_video = args.out_video or output_paths['video'].replace('_output.avi', '_players.mp4')
    out_csv = args.out_csv or output_paths['player_csv']
    
    ensure_dir(out_video)
    ensure_dir(out_csv)

    # Load homography from various sources
    H = None
    
    # Priority 1: Load from court JSON
    if args.court_number and args.court_json:
        try:
            court_config = load_court_config(args.court_number, args.court_json)
            H = court_config.get('HOMOGRAPHY')
            if H is not None:
                print(f"Loaded homography from court {args.court_number} in {args.court_json}")
        except Exception as e:
            print(f"Warning: Could not load court config: {e}")
    
    # Priority 2: CSV calibration file
    if H is None and args.calib_csv:
        try:
            H = load_calib_csv(args.calib_csv)
            print(f"Loaded homography from {args.calib_csv}")
        except Exception as e:
            print(f"Warning: Failed to load {args.calib_csv}: {e}")
    
    # Priority 3: Command line calibration string
    if H is None and args.calib:
        try:
            img_pts = parse_calib_points(args.calib)
            H = build_homography(img_pts, court_w=args.court_w, court_h=args.court_h)
            print("Loaded homography from 4-corner string.")
        except Exception as e:
            print(f"Warning: Failed to parse --calib: {e}")

    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    rows = []
    frame_idx = -1

    track_kwargs = dict(
        source=args.video, stream=True, conf=args.conf,
        classes=[0], tracker=args.tracker, device=args.device
    )
    if args.imgsz:
        track_kwargs["imgsz"] = args.imgsz

    print(f"Running player tracking on {args.video}...")

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
    cap.release()

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x_px", "y_px", "x_m", "y_m", "confidence"])
    df.to_csv(out_csv, index=False)

    print("Done!")
    print(f"- Annotated video: {out_video}")
    print(f"- Tracks CSV:      {out_csv}")
    
    # Auto-generate heatmap
    if args.auto_heatmap:
        print("\nGenerating heatmap automatically...")
        
        court_img_path = output_paths['court_frame']
        ensure_dir(court_img_path)
        cv2.imwrite(court_img_path, first_frame)
        
        heatmap_output = output_paths['heatmap']
        
        players_filter = None
        if args.heatmap_players:
            players_filter = [int(s) for s in args.heatmap_players.split(",") if s.strip().isdigit()]
        
        try:
            generate_heatmap(
                csv_path=out_csv,
                court_img_path=court_img_path,
                out_png=heatmap_output,
                bins_x=args.heatmap_bins_x,
                bins_y=args.heatmap_bins_y,
                gauss=args.heatmap_gauss,
                inplay_only=args.heatmap_inplay_only,
                min_conf=args.heatmap_min_conf,
                players=players_filter,
                heat_alpha=args.heatmap_alpha,
                show_axes=args.heatmap_show_axes,
            )
            print(f"- Heatmap:         {heatmap_output}")
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")


# =============================================================================
# STANDALONE RALLY DETECTION
# =============================================================================

def run_rally_detection(args):
    """Run standalone rally detection from ball detections pickle."""
    from trackers import RallyTracker
    
    print("=" * 60)
    print("STANDALONE RALLY DETECTION")
    print("=" * 60)
    
    if not args.ball_detections_pkl:
        print("Error: --ball_detections_pkl required for rally mode")
        sys.exit(1)
    
    print(f"Loading ball detections from {args.ball_detections_pkl}...")
    with open(args.ball_detections_pkl, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, tuple):
            ball_detections, _ = data
        else:
            ball_detections = data
    
    print(f"Loaded {len(ball_detections)} frames of ball detections")
    
    frame_height = args.frame_height or 720
    
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
    ensure_dir(rally_csv)
    rally_tracker.export_rallies_csv(rally_csv)
    print(f"\nRally CSV saved to: {rally_csv}")
    
    return rallies, rally_tracker


# =============================================================================
# STANDALONE SHOT INFERENCE
# =============================================================================

def run_shot_inference(args):
    """Run standalone shot classification inference."""
    from utils import read_video, save_video
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get output paths
    output_paths = get_output_paths(args.input)
    output_path = args.output or output_paths['video'].replace('_output.avi', '_shots.mp4')
    
    # Read video
    print(f"\nReading video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames at {fps:.1f} FPS")
    
    # Process
    processor = ShotClassificationProcessor(
        shot_model_path=args.shot_model,
        yolo_pose_path=args.yolo_pose,
        tracknet_path=args.tracknet,
        device=device,
        confidence_threshold=args.confidence_threshold,
        window_size=args.window_size,
        stride=args.stride,
        classifier_resolution=args.classifier_resolution,
    )
    
    shot_data = processor.process_video(frames, fps=fps, yolo_resolution=tuple(args.yolo_resolution))
    
    # Draw overlay
    output_frames = processor.draw_shot_overlay(frames, shot_data)
    
    # Save
    ensure_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    # Export shots CSV
    shots_csv = output_paths['shots_csv']
    processor.export_shots_csv(shots_csv)
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Shots CSV: {shots_csv}")
    print(f"\nPlayer Stats: {processor.player_stats}")


# =============================================================================
# FULL PIPELINE (INTEGRATED)
# =============================================================================

def run_main_pipeline(args):
    """Run full padel analysis pipeline with all features integrated."""
    from utils import read_video, save_video
    from trackers import PlayerTracker, RallyTracker

    print("=" * 60)
    print("PADEL MATCH ANALYSIS - FULL PIPELINE")
    print("=" * 60)

    # Get output paths based on input video
    output_paths = get_output_paths(args.input_video, args.output_dir)
    
    # Load court configuration from JSON
    court_config = load_court_config(args.court_number, args.court_json)
    LEFT_EXCLUSION_ZONE = court_config['LEFT_EXCLUSION_ZONE']
    RIGHT_EXCLUSION_ZONE = court_config['RIGHT_EXCLUSION_ZONE']
    homography = court_config.get('HOMOGRAPHY')  # May be None if no calibration points
    
    if homography is not None:
        print(f"Loaded homography from court {args.court_number}")
    else:
        print(f"No calibration points found for court {args.court_number}")

    video_frames = read_video(args.input_video)
    frame_height, frame_width = video_frames[0].shape[:2]
    print(f"\nVideo: {len(video_frames)} frames, {frame_width}x{frame_height}, {args.fps} fps")
    
    # Save first frame for heatmap
    court_frame_path = output_paths['court_frame']
    ensure_dir(court_frame_path)
    cv2.imwrite(court_frame_path, video_frames[0])
    print(f"Saved court frame: {court_frame_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    # Track shot data if enabled
    shot_data = None
    shot_processor = None

    # =========================================================================
    # STEP 1: Shot Classification (FIRST - as requested)
    # =========================================================================
    if args.enable_shot_classification:
        print("\n" + "=" * 60)
        print("STEP 1: Shot Classification")
        print("=" * 60)
        
        try:
            shot_processor = ShotClassificationProcessor(
                shot_model_path=args.shot_model,
                yolo_pose_path=args.yolo_pose,
                tracknet_path=args.tracknet_model,
                device=device,
                confidence_threshold=args.confidence_threshold,
                window_size=args.window_size,
                stride=args.stride,
                classifier_resolution=args.classifier_resolution,
            )
            
            shot_data = shot_processor.process_video(
                video_frames, 
                fps=args.fps, 
                yolo_resolution=tuple(args.yolo_resolution)
            )
            
            # Export shots CSV
            shot_processor.export_shots_csv(output_paths['shots_csv'])
            
            print(f"\nShot classification complete: {len(shot_data['shot_events'])} shots detected")
            print(f"Player stats: {shot_processor.player_stats}")
        except Exception as e:
            print(f"Warning: Shot classification failed: {e}")
            print("Continuing without shot classification...")
            shot_data = None
    else:
        print("\n[Skipping Step 1: Shot Classification - not enabled]")

    # =========================================================================
    # STEP 2: Player Detection (YOLO)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Player Detection (YOLO)")
    print("=" * 60)

    player_tracker = PlayerTracker(model_path=args.player_model)
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=args.player_stub if args.use_stubs else None
    )

    # =========================================================================
    # STEP 3: Ball Detection (TrackNet with Filters)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Ball Detection (TrackNet with Filters)")
    print("=" * 60)

    ball_tracker = BallTrackerTrackNet(
        tracknet_path=args.tracknet_model,
        detection_threshold=args.ball_threshold,
        min_heatmap_confidence=args.min_heatmap_conf,
    )

    # Add exclusion zones from court config
    if LEFT_EXCLUSION_ZONE:
        ball_tracker.add_exclusion_zone(LEFT_EXCLUSION_ZONE, "left_glass")
    if RIGHT_EXCLUSION_ZONE:
        ball_tracker.add_exclusion_zone(RIGHT_EXCLUSION_ZONE, "right_glass")

    ball_tracker.use_exclusion_filter = True
    ball_tracker.use_player_filter = True
    ball_tracker.use_trajectory_filter = True

    ball_detections = ball_tracker.detect_frames(
        video_frames,
        player_detections=player_detections,
        read_from_stub=args.use_stubs,
        stub_path=args.ball_stub if args.use_stubs else None,
    )

    detected_count = sum(1 for d in ball_detections if d)
    print(f"\nBall detected in {detected_count}/{len(video_frames)} frames "
          f"({100 * detected_count / len(video_frames):.1f}%)")

    # =========================================================================
    # STEP 4: Rally Detection
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Rally Detection")
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

    rally_frames = set()
    if summary['rally_count'] > 0:
        print(f"  Total rally time: {summary['total_rally_time_sec']:.1f}s")
        print(f"  Average duration: {summary['avg_duration_sec']:.1f}s")

        for rally in rallies:
            print(f"  Rally #{rally.rally_id}: "
                  f"Frames {rally.start_frame}-{rally.end_frame} "
                  f"({rally.duration_seconds(args.fps):.1f}s)")
            for f in range(rally.start_frame, rally.end_frame + 1):
                rally_frames.add(f)

    rally_tracker.export_rallies_csv(output_paths['rally_csv'])

    # =========================================================================
    # STEP 5: Player Position Export & Heatmap Generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Player Position Export & Heatmap")
    print("=" * 60)

    # Use homography from court config, or fall back to command line options
    if homography is None:
        if args.calib_csv:
            try:
                homography = load_calib_csv(args.calib_csv)
                print(f"Loaded homography from {args.calib_csv}")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
        elif args.calib:
            try:
                img_pts = parse_calib_points(args.calib)
                homography = build_homography(img_pts, court_w=args.court_w, court_h=args.court_h)
                print("Loaded homography from 4-corner calibration string")
            except Exception as e:
                print(f"Warning: Could not parse calibration: {e}")

    export_player_positions_csv(
        player_detections, 
        output_paths['player_csv'], 
        fps=args.fps,
        homography=homography,
        court_w=args.court_w,
        court_h=args.court_h,
        rally_frames=rally_frames
    )

    # Auto-generate heatmap (always, using first frame as court image)
    print("\nGenerating player heatmap...")
    
    players_filter = None
    if args.heatmap_players:
        players_filter = [int(s) for s in args.heatmap_players.split(",") if s.strip().isdigit()]
    
    try:
        generate_heatmap(
            csv_path=output_paths['player_csv'],
            court_img_path=court_frame_path,
            out_png=output_paths['heatmap'],
            bins_x=args.heatmap_bins_x,
            bins_y=args.heatmap_bins_y,
            gauss=args.heatmap_gauss,
            inplay_only=args.heatmap_inplay_only,
            min_conf=args.heatmap_min_conf,
            players=players_filter,
            heat_alpha=args.heatmap_alpha,
            show_axes=args.heatmap_show_axes,
        )
    except Exception as e:
        print(f"Warning: Could not generate heatmap: {e}")

    # =========================================================================
    # STEP 6: Generate Combined Output Video
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Generating Combined Output Video")
    print("=" * 60)

    # Start with original frames
    output_frames = [f.copy() for f in video_frames]
    
    # Layer 1: Player bboxes
    output_frames = player_tracker.draw_bboxes(output_frames, player_detections)

    # Layer 2: Ball tracking
    output_frames = ball_tracker.draw_bboxes(
        output_frames,
        ball_detections,
        trail_length=args.trail_length,
        show_confidence=True,
        draw_exclusion_zones=args.draw_exclusion_zones,
    )

    # Layer 3: Shot classification overlay (if enabled and available)
    if shot_data and shot_processor:
        print("Adding shot classification overlay...")
        output_frames = shot_processor.draw_shot_overlay(output_frames, shot_data)

    # Layer 4: Rally overlay
    output_frames = rally_tracker.draw_overlay(output_frames)

    # Add frame counter (bottom right)
    for i, frame in enumerate(output_frames):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Frame: {i}", (w - 200, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 221), 2)

    output_video = args.output_video or output_paths['video']
    ensure_dir(output_video)
    save_video(output_frames, output_video, args.fps)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - OUTPUT FILES")
    print("=" * 60)
    print(f"  Video:           {output_video}")
    print(f"  Rally CSV:       {output_paths['rally_csv']}")
    print(f"  Player CSV:      {output_paths['player_csv']}")
    print(f"  Heatmap:         {output_paths['heatmap']}")
    print(f"  Court Frame:     {court_frame_path}")
    if shot_data:
        print(f"  Shots CSV:       {output_paths['shots_csv']}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merged Padel Match Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", default="full",
                        choices=["full", "shots", "player_track", "rally", "heatmap", "calibrate"],
                        help="Pipeline mode")

    # Output directory
    parser.add_argument("--output_dir", default="output_videos",
                        help="Output directory for all generated files")

    # Court configuration
    parser.add_argument("--court_json", default="court_info/court_information.json",
                        help="Path to court configuration JSON file")

    # Main Pipeline (mode=full)
    main_group = parser.add_argument_group("Main Pipeline Options (mode=full)")
    main_group.add_argument("--input_video", default="input_videos/video2_trimmed.mp4")
    main_group.add_argument("--output_video", default=None,
                            help="Output video path (auto-generated from input if not specified)")
    main_group.add_argument("--tracknet_model", default="models/TrackNet_best.pt")
    main_group.add_argument("--player_model", default="models/yolov8s")
    main_group.add_argument("--player_stub", default="tracker_stubs/player_detections.pkl")
    main_group.add_argument("--ball_stub", default="tracker_stubs/ball_detections.pkl")
    main_group.add_argument("--fps", type=float, default=30.0)
    main_group.add_argument("--court_number", type=int, default=9)
    main_group.add_argument("--use_stubs", action="store_true", default=False)
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
    main_group.add_argument("--player_csv", default=None)
    main_group.add_argument("--generate_heatmap", action="store_true", default=True,
                            help="Generate heatmap (default: True)")

    # Shot Classification Integration
    shot_int_group = parser.add_argument_group("Shot Classification Integration (mode=full)")
    shot_int_group.add_argument("--enable_shot_classification", action="store_true", default=True,
                                help="Enable shot classification in full pipeline")
    shot_int_group.add_argument("--shot_model", default="models/best_model.pth")
    shot_int_group.add_argument("--yolo_pose", default="models/yolov8n-pose.pt")
    shot_int_group.add_argument("--confidence_threshold", type=float, default=None)
    shot_int_group.add_argument("--window_size", type=int, default=32)
    shot_int_group.add_argument("--stride", type=int, default=8)
    shot_int_group.add_argument("--yolo_resolution", type=int, nargs=2, default=[640, 360])
    shot_int_group.add_argument("--classifier_resolution", type=int, default=128)

    # Shot Classification Standalone (mode=shots)
    shot_group = parser.add_argument_group("Shot Classification Standalone (mode=shots)")
    shot_group.add_argument("--input", type=str, default="input_videos/video2_trimmed.mp4")
    shot_group.add_argument("--output", type=str, default=None)
    shot_group.add_argument("--tracknet", default="models/TrackNet_best.pt")
    shot_group.add_argument("--threshold", type=float, default=0.5)
    shot_group.add_argument("--processing_resolution", type=int, nargs=2, default=[1280, 720])
    shot_group.add_argument("--lookahead", type=int, default=16)

    # Standalone Player Tracker (mode=player_track)
    pt_group = parser.add_argument_group("Player Tracker Options (mode=player_track)")
    pt_group.add_argument("--video", default="input_videos/video2_trimmed.mp4")
    pt_group.add_argument("--out_video", default=None)
    pt_group.add_argument("--out_csv", default=None)
    pt_group.add_argument("--model", default="models/yolov8s.pt")
    pt_group.add_argument("--conf", type=float, default=0.25)
    pt_group.add_argument("--imgsz", type=int, default=None)
    pt_group.add_argument("--device", default=None)
    pt_group.add_argument("--tracker", default="bytetrack.yaml")
    pt_group.add_argument("--auto_heatmap", action="store_true", default=True,
                          help="Automatically generate heatmap after tracking (default: True)")

    # Rally Detection (mode=rally)
    rally_group = parser.add_argument_group("Rally Detection Options (mode=rally)")
    rally_group.add_argument("--ball_detections_pkl", default=None)
    rally_group.add_argument("--frame_height", type=int, default=None)

    # Calibration (can be used as fallback)
    calib_group = parser.add_argument_group("Calibration Options (fallback if not in JSON)")
    calib_group.add_argument("--calib", default=None,
                             help="4-corner calibration string: x1,y1;x2,y2;x3,y3;x4,y4")
    calib_group.add_argument("--calib_csv", default=None,
                             help="CSV file with calibration points (x_px,y_px,x_m,y_m)")
    calib_group.add_argument("--court_w", type=float, default=20.0,
                             help="Court width in meters")
    calib_group.add_argument("--court_h", type=float, default=10.0,
                             help="Court height in meters")

    # Heatmap Options
    heat_group = parser.add_argument_group("Heatmap Options")
    heat_group.add_argument("--csv", default=None)
    heat_group.add_argument("--court_img", default=None)
    heat_group.add_argument("--heatmap_output", default=None)
    heat_group.add_argument("--heatmap_court_img", default=None)
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
        # Run the new calibration utility
        video_path = args.video or args.input_video
        court_number = args.court_number
        json_path = args.court_json
        
        print(f"Starting calibration for court {court_number}")
        print(f"Video: {video_path}")
        print(f"JSON output: {json_path}")
        
        calibrate_court(video_path, court_number, json_path)

    elif args.mode == "heatmap":
        if not args.csv or not args.court_img:
            print("Error: --csv and --court_img required for heatmap mode")
            sys.exit(1)

        players = [int(s) for s in args.heatmap_players.split(",") if s.strip().isdigit()] if args.heatmap_players else None
        output = args.heatmap_output or get_output_paths(args.csv)['heatmap']

        generate_heatmap(
            csv_path=args.csv,
            court_img_path=args.court_img,
            out_png=output,
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

    elif args.mode == "rally":
        run_rally_detection(args)

    elif args.mode == "shots":
        if not args.input:
            print("Error: --input required for shots mode")
            sys.exit(1)
        run_shot_inference(args)

    else:  # mode == "full"
        run_main_pipeline(args)


if __name__ == "__main__":
    main()