import os
import numpy as np
import cv2
import csv
from typing import Dict

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
        'zones': os.path.join(output_dir, f"{basename}_zones.png"),
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