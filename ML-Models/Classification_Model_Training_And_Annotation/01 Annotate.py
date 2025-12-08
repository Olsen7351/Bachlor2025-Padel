#!/usr/bin/env python3
"""
Padel Shot Annotation Pipeline

Complete annotation workflow:
1. Semi-automatic shot detection using TrackNet + YOLO pose
2. Manual sorting and labeling of detected shots
3. Pose extraction for training data

Usage:
    # Step 1: Extract shots from video (semi-automatic detection)
    python annotate.py extract --video videos/game.mp4
    
    # Step 2: Sort and label the extracted clips
    python annotate.py sort
    
    # Step 3: Extract pose data for training
    python annotate.py prepare

Keyboard Controls (for sorting):
    1 - Mark as Overhead
    2 - Mark as Serve
    3 - Mark as Forehand
    4 - Mark as Backhand
    5 - Mark as Lob
    6 - Mark as Other (non-shot)
    D - Delete (not valid)
    SPACE - Skip (review later)
    Q - Quit and save progress
    R - Replay current video
"""

import argparse
import os
import cv2
import json
import shutil
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from collections import deque
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import video as video_models


# ==========================================
# SHOT CLASSIFIER MODEL
# ==========================================

class EnhancedVideoClassifier(nn.Module):
    """Enhanced classifier combining video features (R(2+1)D-18) with pose keypoints (Bi-LSTM)."""

    def __init__(self, num_classes, pose_input_size=51, pose_hidden_size=256, pretrained_video=True):
        super().__init__()

        # Video branch (R(2+1)D-18)
        self.video_model = video_models.r2plus1d_18(
            weights=video_models.R2Plus1D_18_Weights.DEFAULT if pretrained_video else None
        )
        num_video_ftrs = self.video_model.fc.in_features
        self.video_model.fc = nn.Identity()

        # Pose branch (Bi-LSTM with attention)
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

        # Fusion classifier
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
        # Video features
        video_features = self.video_model(video_x)

        # Pose features with attention
        pose_out, _ = self.pose_lstm(pose_x)
        attention_weights = self.pose_attention(pose_out)
        pose_features = torch.sum(pose_out * attention_weights, dim=1)

        # Fusion and classification
        combined = torch.cat((video_features, pose_features), dim=1)
        return self.classifier(combined)


class ShotClassifier:
    """Wrapper for shot classification model"""

    def __init__(self, model_path, pose_model, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pose_model = pose_model
        self.model = None
        self.idx_to_label = None
        self.clip_len = 32

        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False)
                labels_map = checkpoint['labels_map']
                num_classes = len(labels_map)
                config = checkpoint.get('config', {})

                self.model = EnhancedVideoClassifier(
                    num_classes=num_classes,
                    pose_input_size=51,
                    pose_hidden_size=config.get('pose_hidden_size', 128)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                self.idx_to_label = {
                    idx: name for name, idx in labels_map.items()}
                print(f"Shot classifier loaded: {model_path}")
                print(
                    f"  Classes: {', '.join([self.idx_to_label[i] for i in range(num_classes)])}")
            except Exception as e:
                print(f"Warning: Could not load classifier model: {e}")
                self.model = None

    def classify(self, frames, keypoints_seq=None):
        """Classify a clip. Returns (class_name, confidence) or (None, 0) if no model."""
        if self.model is None or len(frames) < 8:
            return None, 0.0

        try:
            # Preprocess video
            clip_tensor = self._preprocess_video(frames)

            # Get or compute poses
            if keypoints_seq is None:
                keypoints_seq = self._extract_poses(frames)

            pose_tensor = self._preprocess_poses(keypoints_seq)

            # Classify
            with torch.no_grad():
                outputs = self.model(clip_tensor.to(
                    self.device), pose_tensor.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                max_prob, pred_class = torch.max(probs, 1)

                label = self.idx_to_label[pred_class.item()]
                conf = max_prob.item()

                return label, conf
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0.0

    def _preprocess_video(self, frames, resolution=(112, 112)):
        """Preprocess video frames for model input."""
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        # Convert BGR to RGB
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        frames_array = np.array(rgb_frames)

        frames_tensor = torch.from_numpy(
            frames_array).float().permute(0, 3, 1, 2)

        total = frames_tensor.shape[0]
        if total >= self.clip_len:
            indices = np.linspace(0, total - 1, self.clip_len, dtype=int)
        else:
            indices = np.pad(np.arange(total),
                             (0, self.clip_len - total), 'edge')

        sampled = frames_tensor[indices].permute(1, 0, 2, 3) / 255.0
        sampled = transforms.functional.resize(sampled, resolution)
        sampled = (sampled - norm_mean) / norm_std

        return sampled.unsqueeze(0)

    def _extract_poses(self, frames):
        """Extract pose keypoints from frames."""
        poses = []
        for frame in frames:
            results = self.pose_model(frame, verbose=False)
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # Take first person's keypoints
                kp = results[0].keypoints.data[0].cpu().numpy().flatten()
                poses.append(kp)
            else:
                poses.append(np.zeros(51))
        return np.array(poses)

    def _preprocess_poses(self, keypoints_seq):
        """Preprocess pose sequence for model input."""
        poses = keypoints_seq.copy()

        # Normalize to 0-1 (assuming 1280x720 input resolution for annotation)
        # Safety check: only normalize if values are > 1.0
        if np.max(poses) > 1.0:
            if len(poses.shape) == 3:  # (T, 17, 3)
                poses[:, :, 0] /= 1280.0
                poses[:, :, 1] /= 720.0
                poses = poses.reshape(poses.shape[0], -1)
            elif len(poses.shape) == 2:  # (T, 51)
                # Reshape, normalize, flatten
                p_reshaped = poses.reshape(-1, 17, 3)
                p_reshaped[:, :, 0] /= 1280.0
                p_reshaped[:, :, 1] /= 720.0
                poses = p_reshaped.reshape(poses.shape[0], -1)

        total = len(poses)
        if total >= self.clip_len:
            indices = np.linspace(0, total - 1, self.clip_len, dtype=int)
            poses = poses[indices]
        else:
            poses = np.pad(poses, ((0, self.clip_len - total), (0, 0)), 'edge')

        return torch.from_numpy(poses).unsqueeze(0).float()

    def predict_from_video(self, video_path, pose_data=None):
        """Load video and classify. Returns (class_name, confidence) or (None, 0)."""
        if self.model is None:
            return None, 0.0

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return None, 0.0

        return self.classify(frames, pose_data)


# ==========================================
# TRACKNET MODEL ARCHITECTURE
# ==========================================

class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.mp = nn.MaxPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.down_block_1(x)
        x = self.mp(x1)
        x2 = self.down_block_2(x)
        x = self.mp(x2)
        x3 = self.down_block_3(x)
        x = self.mp(x3)
        x = self.bottleneck(x)
        x = torch.cat([self.up(x), x3], 1)
        x = self.up_block_1(x)
        x = torch.cat([self.up(x), x2], 1)
        x = self.up_block_2(x)
        x = torch.cat([self.up(x), x1], 1)
        x = self.up_block_3(x)
        return self.sigmoid(self.predictor(x))


# ==========================================
# BALL TRACKING
# ==========================================

class BallPositionTracker:
    """Tracks ball positions and fills gaps when detection is lost"""

    def __init__(self, max_gap_frames=10, min_velocity=2.0):
        self.positions = deque(maxlen=30)
        self.detection_sources = deque(maxlen=30)
        self.gap_count = 0
        self.max_gap = max_gap_frames
        self.min_vel = min_velocity
        self.tracking_confidence = 0
        self.min_confidence_for_hit = 5
        self.velocity = np.array([0.0, 0.0])
        self.last_detected = None

    def update(self, detected_ball):
        if detected_ball is not None:
            if len(self.positions) > 0:
                last_pos = np.array(self.positions[-1], dtype=np.float32)
                current_pos = np.array(detected_ball, dtype=np.float32)
                new_velocity = current_pos - last_pos
                self.velocity = new_velocity if self.gap_count == 0 else 0.5 * \
                    new_velocity + 0.5 * self.velocity

            self.gap_count = 0
            self.last_detected = detected_ball
            self.positions.append(detected_ball)
            self.detection_sources.append(True)
            self.tracking_confidence += 1
            return detected_ball, True, self.tracking_confidence >= self.min_confidence_for_hit
        else:
            self.gap_count += 1
            if self.gap_count > self.max_gap:
                self.tracking_confidence = 0
                return None, False, False

            predicted = self._predict_position()
            if predicted is not None:
                self.positions.append(predicted)
                self.detection_sources.append(False)
                return predicted, False, self.tracking_confidence >= self.min_confidence_for_hit
            return None, False, False

    def _predict_position(self):
        if len(self.positions) < 1 or np.linalg.norm(self.velocity) < self.min_vel:
            return None
        current_pos = np.array(self.positions[-1], dtype=np.float32)
        if self.gap_count > 2:
            gravity = np.array([0.0, 2.0])
            predicted_velocity = self.velocity + gravity * (self.gap_count - 2)
        else:
            predicted_velocity = self.velocity
        predicted_pos = current_pos + predicted_velocity
        self.velocity = predicted_velocity
        return (int(predicted_pos[0]), int(predicted_pos[1]))

    def get_history(self):
        return list(self.positions)

    def get_detection_sources(self):
        return list(self.detection_sources)


class BallTracker:
    """TrackNet-based ball tracker"""

    def __init__(self, model_path, video_path=None, input_wh=(512, 288)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w, self.h = input_wh
        self.bg = None
        self.model = None
        self.seq_len = 8
        self.bg_mode = 'concat'

        try:
            ckpt = torch.load(
                model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                sd = ckpt['model']
                p = ckpt.get('param_dict', {})
                self.seq_len = p.get('seq_len', 8)
                self.bg_mode = p.get('bg_mode', 'concat')
            else:
                sd = ckpt

            in_c = (self.seq_len + 1) * \
                3 if self.bg_mode == 'concat' else self.seq_len * 3
            self.model = TrackNet(in_c, self.seq_len).to(self.device)
            self.model.load_state_dict(sd)
            self.model.eval()

            self.buf = deque(maxlen=self.seq_len)

            if video_path and self.bg_mode == 'concat':
                self._compute_bg(video_path)

            print(f"TrackNet loaded successfully on {self.device}")
        except Exception as e:
            print(f"TrackNet Load Error: {e}")

    def reset(self, video_path=None):
        """Reset buffer and optionally compute new background for a different video"""
        self.buf.clear()
        if video_path and self.bg_mode == 'concat':
            self._compute_bg(video_path)

    def set_background_from_frames(self, frames):
        """Compute background from a list of frames (faster than reading video)"""
        if not frames or self.bg_mode != 'concat':
            return
        # Sample frames for median background
        indices = np.linspace(0, len(frames) - 1,
                              min(15, len(frames)), dtype=int)
        sampled = [cv2.resize(frames[i], (self.w, self.h)) for i in indices]
        med = np.median(sampled, axis=0).astype(np.uint8)
        self.bg = torch.from_numpy(cv2.cvtColor(
            med, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        self.bg = self.bg.permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.buf.clear()

    def _compute_bg(self, v_path):
        cap = cv2.VideoCapture(v_path)
        frames = []
        for i in np.linspace(0, min(int(cap.get(7)) - 1, 500), 30, dtype=int):
            cap.set(1, i)
            r, f = cap.read()
            if r:
                frames.append(cv2.resize(f, (self.w, self.h)))
        cap.release()
        if frames:
            med = np.median(frames, axis=0).astype(np.uint8)
            self.bg = torch.from_numpy(cv2.cvtColor(
                med, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            self.bg = self.bg.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def predict(self, frame):
        if not self.model:
            return None
        rgb = cv2.cvtColor(cv2.resize(frame, (self.w, self.h)),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.buf.append(torch.from_numpy(rgb).permute(2, 0, 1).to(self.device))
        if len(self.buf) < self.seq_len:
            return None

        x_seq = torch.stack(list(self.buf)).reshape(1, -1, self.h, self.w)
        x = torch.cat([self.bg, x_seq], 1) if self.bg is not None else x_seq

        with torch.no_grad():
            return self.model(x)[0, -1].cpu().numpy()


# ==========================================
# HIT DETECTION LOGIC
# ==========================================

class GameLogic:
    """Detects hits based on ball-wrist proximity and trajectory changes"""

    def __init__(self, cooldown_frames=15, hit_threshold_ratio=0.040, min_confidence=0.1, ball_tracker=None):
        self.hits = 0
        self.cooldown = 0
        self.limit = cooldown_frames
        self.hit_threshold = hit_threshold_ratio
        self.min_conf = min_confidence
        self.ball_history = deque(maxlen=10)
        self.pending_hit = None
        self.validation_timer = 0
        self.validation_frames = 8
        self.ball_tracker = ball_tracker
        self.validation_start_idx = 0

    def check_hit(self, ball, keypoints, h, w):
        if self.pending_hit is not None:
            self.validation_timer += 1
            if ball is not None:
                self.ball_history.append(ball)
            if self.validation_timer >= self.validation_frames:
                if self._validate_hit():
                    self.hits += 1
                    self.cooldown = self.limit
                    result = (True, self.hits, self.pending_hit)
                    self.pending_hit = None
                    self.validation_timer = 0
                    return result
                else:
                    self.pending_hit = None
                    self.validation_timer = 0
            return False, self.hits, None

        if self.cooldown > 0:
            self.cooldown -= 1
            return False, self.hits, None

        if ball is None or len(keypoints) < 11:
            return False, self.hits, None

        self.ball_history.append(ball)
        ball_pos = np.array(ball, dtype=np.float32)
        base_threshold = w * self.hit_threshold
        has_conf = keypoints.shape[1] >= 3

        # Check wrists and elbows
        joints = [
            (10, 'right', 'wrist', 1.0),
            (9, 'left', 'wrist', 1.0),
            (8, 'right', 'elbow', 0.8),
            (7, 'left', 'elbow', 0.8)
        ]

        hit_detected = False
        hit_info = None

        for idx, side, joint_name, threshold_mult in joints:
            joint = keypoints[idx]
            joint_pos = (joint[0] * w, joint[1] * h)
            joint_conf = joint[2] if has_conf else 1.0

            if joint_conf >= self.min_conf:
                dist = np.linalg.norm(ball_pos - np.array(joint_pos))
                threshold = base_threshold * threshold_mult
                if dist < threshold:
                    hit_detected = True
                    hit_info = {'side': side, 'joint': joint_name,
                                'distance': dist, 'threshold': threshold}
                    break

        if hit_detected and len(self.ball_history) >= 4:
            if not self._check_trajectory_change():
                hit_detected = False

        if hit_detected:
            self.pending_hit = hit_info
            self.validation_timer = 0
            if self.ball_tracker is not None:
                self.validation_start_idx = len(
                    self.ball_tracker.get_detection_sources())
            return False, self.hits, {'pending': True, **hit_info}

        return False, self.hits, None

    def _check_trajectory_change(self):
        if len(self.ball_history) < 5:
            return True
        positions = list(self.ball_history)
        v_before = (np.array(positions[-3]) - np.array(positions[-5])) / 2.0
        v_after = np.array(positions[-1]) - np.array(positions[-2])
        v_before_mag = np.linalg.norm(v_before)
        v_after_mag = np.linalg.norm(v_after)
        if v_before_mag < 2 or v_after_mag < 2:
            return True
        cos_angle = np.dot(v_before, v_after) / (v_before_mag * v_after_mag)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        speed_ratio = v_after_mag / v_before_mag
        return angle_deg > 40 or speed_ratio < 0.5 or speed_ratio > 2.0

    def _validate_hit(self):
        if len(self.ball_history) < self.validation_frames + 2:
            return True
        positions = list(self.ball_history)
        hit_idx = len(positions) - self.validation_frames
        if hit_idx < 3:
            return True
        before = positions[hit_idx - 3:hit_idx]
        after = positions[hit_idx:hit_idx + 4]
        if len(before) < 2 or len(after) < 2:
            return True
        vel_before = np.array(before[-1]) - np.array(before[0])
        vel_after = np.array(after[-1]) - np.array(after[0])
        mag_before = np.linalg.norm(vel_before)
        mag_after = np.linalg.norm(vel_after)
        if mag_before < 5 or mag_after < 5:
            return True
        cos_angle = np.dot(vel_before, vel_after) / (mag_before * mag_after)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        speed_ratio = mag_after / mag_before
        return angle_deg > 30 or speed_ratio < 0.6 or speed_ratio > 1.7


# ==========================================
# DATASET RECORDER
# ==========================================

class DatasetRecorder:
    def __init__(self, output_dir="dataset", output_resolution=(1280, 720), classifier=None):
        self.raw_dir = os.path.join(output_dir, "raw_clips")
        self.pose_dir = os.path.join(output_dir, "pose_data")
        self.csv_path = os.path.join(output_dir, "labels.csv")
        self.predictions_path = os.path.join(output_dir, "predictions.json")
        self.buffer = deque(maxlen=30)
        self.recording = False
        self.timer = 0
        self.clip = []
        self.idx = self._get_next_idx()
        self.output_w, self.output_h = output_resolution
        self.classifier = classifier
        self.predictions = self._load_predictions()

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ["filename", "pose_file", "timestamp", "resolution", "predicted_class", "confidence"])

    def _get_next_idx(self):
        existing = [f for f in os.listdir(self.raw_dir) if f.startswith(
            "shot_") and f.endswith(".mp4")]
        if not existing:
            return 0
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing]
        return max(indices) + 1 if indices else 0

    def _load_predictions(self):
        if os.path.exists(self.predictions_path):
            with open(self.predictions_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_predictions(self):
        with open(self.predictions_path, 'w') as f:
            json.dump(self.predictions, f, indent=2)

    def process(self, frame, keypoints, trigger):
        self.buffer.append((frame, keypoints))
        if trigger and not self.recording:
            self.recording = True
            self.timer = 30
            self.clip = list(self.buffer)
        if self.recording:
            if not trigger:
                self.clip.append((frame, keypoints))
            self.timer -= 1
            if self.timer <= 0:
                self._save()

    def _save(self):
        if not self.clip:
            return
        self.recording = False
        name = f"shot_{self.idx:04d}"

        # Save video
        out = cv2.VideoWriter(
            os.path.join(self.raw_dir, f"{name}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.output_w, self.output_h)
        )
        frames_list = []
        for f, _ in self.clip:
            resized = cv2.resize(f, (self.output_w, self.output_h))
            out.write(resized)
            frames_list.append(resized)
        out.release()

        # Save keypoints
        kpts_list = [x[1] for x in self.clip]
        valid_kpts = [k for k in kpts_list if isinstance(
            k, np.ndarray) and k.size > 0]
        shape = valid_kpts[0].shape if valid_kpts else (17, 2)
        padded = [k if isinstance(k, np.ndarray) and k.shape == shape else np.full(
            shape, np.nan) for k in kpts_list]
        np.save(os.path.join(self.pose_dir, f"{name}.npy"), np.array(padded))

        # Classify if model available
        pred_class, confidence = None, 0.0
        if self.classifier is not None:
            pred_class, confidence = self.classifier.classify(frames_list)
            if pred_class:
                self.predictions[f"{name}.mp4"] = {
                    "class": pred_class,
                    "confidence": confidence
                }
                self._save_predictions()

        # Update CSV
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                f"{name}.mp4", f"{name}.npy",
                datetime.now().strftime("%H:%M:%S"),
                f"{self.output_w}x{self.output_h}",
                pred_class or "",
                f"{confidence:.3f}" if pred_class else ""
            ])

        if pred_class:
            print(
                f"  Saved: {name}.mp4 -> {pred_class} ({confidence*100:.1f}%)")
        else:
            print(f"  Saved: {name}.mp4")

        self.idx += 1
        self.clip = []


# ==========================================
# EXTRACTION (Semi-automatic annotation)
# ==========================================

def extract_shots(args):
    """Semi-automatic shot extraction from video"""
    print("="*60)
    print("Shot Extraction Mode")
    print("="*60)

    resolution = tuple(map(int, args.resolution.split('x')))
    print(f"Resolution: {resolution[0]}x{resolution[1]}")

    # Load models
    print("\nLoading models...")
    pose_model = YOLO(args.pose_model)
    ball_tracker = BallTracker(args.tracknet_model, args.video)
    position_tracker = BallPositionTracker()
    logic = GameLogic(cooldown_frames=30, ball_tracker=position_tracker)

    # Load classifier if available
    classifier = None
    if hasattr(args, 'classifier_model') and args.classifier_model:
        if os.path.exists(args.classifier_model):
            classifier = ShotClassifier(args.classifier_model, pose_model)
        else:
            print(
                f"Warning: Classifier model not found: {args.classifier_model}")

    recorder = DatasetRecorder(
        args.dataset_dir, resolution, classifier=classifier)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resolution)
        h, w = frame.shape[:2]

        # Pose detection
        p_res = pose_model(frame, verbose=False)[0]
        vis = p_res.plot()
        all_kpts = []
        if len(p_res.keypoints) > 0:
            for i in range(len(p_res.keypoints.xyn)):
                all_kpts.append(p_res.keypoints.xyn[i].cpu().numpy())

        # Ball detection
        hm = ball_tracker.predict(frame)
        detected_ball = None
        if hm is not None:
            _, th = cv2.threshold(hm, 0.5, 1, 0)
            ctrs, _ = cv2.findContours((th * 255).astype(np.uint8), 0, 2)
            if ctrs:
                c = max(ctrs, key=cv2.contourArea)
                (cx, cy), _ = cv2.minEnclosingCircle(c)
                detected_ball = (int(cx * (w / 512)), int(cy * (h / 288)))

        ball, is_detected, is_reliable = position_tracker.update(detected_ball)

        # Visualize ball
        if ball is not None:
            color = (0, 255, 255) if is_detected else (0, 165, 255)
            cv2.circle(vis, ball, 8, color, -1)

        # Hit detection
        is_hit = False
        for kpts in all_kpts:
            hit, _, debug = logic.check_hit(
                ball, kpts, h, w) if is_reliable else (False, 0, None)
            if hit:
                is_hit = True
                break

        recorder.process(frame, all_kpts[0] if all_kpts else [], is_hit)

        # UI
        cv2.putText(vis, f"HITS: {logic.hits}",
                    (40, 60), 0, 1.5, (0, 255, 0), 3)
        if recorder.recording:
            cv2.circle(vis, (30, 100), 15, (0, 0, 255), -1)
            cv2.putText(vis, "REC", (60, 110), 0, 1, (0, 0, 255), 2)

        cv2.imshow("Shot Extraction", vis)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1
        pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    pbar.close()

    print(f"\nExtracted {logic.hits} shots to {args.dataset_dir}/raw_clips/")
    print("Next step: python annotate.py sort")


# ==========================================
# SORTING
# ==========================================

def sort_dataset(args):
    """Sort and label extracted clips"""
    print("="*60)
    print("Dataset Sorting Mode")
    print("="*60)

    dataset_dir = Path(args.dataset_dir)
    raw_dir = dataset_dir / "raw_clips"
    pose_dir = dataset_dir / "pose_data"
    sorted_dir = dataset_dir / "sorted"
    predictions_file = dataset_dir / "predictions.json"
    resolution = tuple(map(int, args.resolution.split('x')))

    categories = {
        '1': 'overhead', '2': 'serve', '3': 'forehand',
        '4': 'backhand', '5': 'lob', '6': 'other'
    }

    # Reverse map for auto-suggestion
    category_keys = {v: k for k, v in categories.items()}

    for cat in categories.values():
        (sorted_dir / cat / "clips").mkdir(parents=True, exist_ok=True)
        (sorted_dir / cat / "pose").mkdir(parents=True, exist_ok=True)

    progress_file = dataset_dir / "sorting_progress.json"
    progress = json.load(open(progress_file)) if progress_file.exists() else {
        'current_idx': 0, 'sorted': {}, 'deleted': [], 'skipped': []}

    # Load predictions if available
    predictions = {}
    if predictions_file.exists():
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions)} cached model predictions")

    # Load classifier for live predictions if no cached predictions
    classifier = None
    if hasattr(args, 'classifier_model') and args.classifier_model:
        if os.path.exists(args.classifier_model):
            print(
                f"Loading classifier for live predictions: {args.classifier_model}")
            pose_model = YOLO('yolo11x-pose.pt')
            classifier = ShotClassifier(args.classifier_model, pose_model)
        else:
            print(f"Classifier model not found: {args.classifier_model}")

    videos = sorted(raw_dir.glob("*.mp4"))

    # Filter out already processed videos
    unsorted_videos = [v for v in videos
                       if str(v.name) not in progress['sorted']
                       and str(v.name) not in progress['deleted']
                       and str(v.name) not in progress.get('skipped', [])]

    print(
        f"Found {len(videos)} total videos, {len(unsorted_videos)} need sorting")
    print(
        f"Already sorted: {len(progress['sorted'])}, deleted: {len(progress['deleted'])}")

    if not unsorted_videos:
        print("\nAll videos have been sorted!")
        return

    current_idx = 0
    while current_idx < len(unsorted_videos):
        video_path = unsorted_videos[current_idx]

        # Get model prediction (cached first, then live if available)
        pred = predictions.get(video_path.name, {})
        pred_class = pred.get('class', None)
        pred_conf = pred.get('confidence', 0)

        # Run live classification if no cached prediction and classifier available
        if not pred_class and classifier:
            pose_file = pose_dir / (video_path.stem + ".npy")
            pose_data = np.load(str(pose_file)) if pose_file.exists() else None
            pred_class, pred_conf = classifier.predict_from_video(
                str(video_path), pose_data)
            # Cache the prediction
            predictions[video_path.name] = {
                'class': pred_class, 'confidence': pred_conf}
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)

        # Play video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            current_idx += 1
            continue

        frame_idx = 0
        choice = None

        while choice is None:
            frame = frames[frame_idx].copy()
            h, w = frame.shape[:2]

            # Draw UI background
            cv2.rectangle(frame, (10, 10), (w - 10, 155), (0, 0, 0), -1)
            cv2.putText(frame, f"[{current_idx + 1}/{len(unsorted_videos)}] {video_path.name}",
                        (20, 35), 0, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "1:Overhead 2:Serve 3:Forehand 4:Backhand 5:Lob 6:Other",
                        (20, 60), 0, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "D:Delete  SPACE:Skip  R:Replay  Q:Quit  ENTER:Accept prediction",
                        (20, 85), 0, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Sorted:{len(progress['sorted'])} Del:{len(progress['deleted'])}",
                        (20, 110), 0, 0.5, (200, 200, 200), 1)

            # Show model prediction
            if pred_class:
                pred_key = category_keys.get(pred_class, '?')
                color = (0, 255, 0) if pred_conf > 0.7 else (
                    0, 200, 255) if pred_conf > 0.5 else (0, 100, 255)
                cv2.putText(frame, f"Model: {pred_class.upper()} ({pred_conf*100:.1f}%) - Press {pred_key} or ENTER",
                            (20, 135), 0, 0.6, color, 2)
            else:
                cv2.putText(frame, "Model: No prediction available",
                            (20, 135), 0, 0.5, (128, 128, 128), 1)

            cv2.imshow("Dataset Sorter", frame)
            key = cv2.waitKey(30) & 0xFF

            if chr(key) in categories:
                choice = categories[chr(key)]
            elif key == 13 and pred_class:  # ENTER key - accept prediction
                choice = pred_class
            elif key == ord('d') or key == ord('D'):
                choice = 'delete'
            elif key == ord(' '):
                choice = 'skip'
            elif key == ord('q') or key == ord('Q'):
                choice = 'quit'
            elif key == ord('r') or key == ord('R'):
                frame_idx = 0
            else:
                frame_idx = (frame_idx + 1) % len(frames)

        cv2.destroyAllWindows()

        if choice == 'quit':
            break
        elif choice == 'skip':
            if str(video_path.name) not in progress['skipped']:
                progress['skipped'].append(str(video_path.name))
        elif choice == 'delete':
            video_path.unlink()
            pose_file = pose_dir / (video_path.stem + ".npy")
            if pose_file.exists():
                pose_file.unlink()
            progress['deleted'].append(str(video_path.name))
            print(f"  Deleted: {video_path.name}")
        else:
            # Move to category folder with consistent resolution
            dest_video = sorted_dir / choice / "clips" / video_path.name
            dest_pose = sorted_dir / choice / \
                "pose" / (video_path.stem + ".npy")

            cap = cv2.VideoCapture(str(video_path))
            out = cv2.VideoWriter(
                str(dest_video), cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(cv2.resize(frame, resolution))
            cap.release()
            out.release()
            video_path.unlink()

            pose_file = pose_dir / (video_path.stem + ".npy")
            if pose_file.exists():
                shutil.move(str(pose_file), str(dest_pose))

            progress['sorted'][str(video_path.name)] = choice
            print(f"  Sorted: {video_path.name} -> {choice}")

        current_idx += 1
        # Save progress (sorted/deleted lists are the source of truth)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SORTING SUMMARY")
    print("="*60)
    print(f"Sorted: {len(progress['sorted'])}")
    print(f"Deleted: {len(progress['deleted'])}")
    print(f"Skipped: {len(progress.get('skipped', []))}")
    for cat in categories.values():
        count = sum(1 for v in progress['sorted'].values() if v == cat)
        print(f"  {cat}: {count}")

    # Check remaining
    remaining = [v for v in sorted(raw_dir.glob("*.mp4"))
                 if str(v.name) not in progress['sorted']
                 and str(v.name) not in progress['deleted']
                 and str(v.name) not in progress.get('skipped', [])]
    if remaining:
        print(f"\nRemaining to sort: {len(remaining)}")
    else:
        print("\nAll videos sorted!")
    print("\nNext step: python annotate.py prepare")


# ==========================================
# PREPARE (Extract pose data for training)
# ==========================================

def prepare_dataset(args):
    """Extract YOLO pose keypoints for training - with manual player selection (FAST)"""
    print("="*60)
    print("Dataset Preparation Mode (Optimized)")
    print("="*60)

    input_dir = Path(args.dataset_dir) / "sorted"
    output_dir = Path(args.dataset_dir) / "player_enhanced"

    # Progress tracking for manual selection
    progress_file = Path(args.dataset_dir) / "prepare_progress.json"
    # Handle --reset flag
    if hasattr(args, 'reset') and args.reset:
        if progress_file.exists():
            progress_file.unlink()
        progress = {'completed': []}
        print("Progress reset!")
    else:
        progress = json.load(open(progress_file)) if progress_file.exists() else {
            'completed': []}

    print("Loading YOLO pose model...")
    pose_model = YOLO(args.pose_model)

    all_classes = [d for d in os.listdir(
        input_dir) if (input_dir / d).is_dir()]

    # Filter classes if --classes specified
    if hasattr(args, 'classes') and args.classes:
        filter_classes = [c.strip() for c in args.classes.split(',')]
        classes = [c for c in all_classes if c in filter_classes]
        print(f"Filtering to classes: {classes}")
    else:
        classes = all_classes

    print(f"Found classes: {classes}")
    print(f"\nPlayer selection: MANUAL - Click player or press 1-4")
    print("Controls: 1-4=Select player | S=Skip | Q=Quit | R=Replay")

    stats = {'total': 0, 'processed': 0, 'skipped': 0}

    for class_name in classes:
        print(f"\nProcessing: {class_name}")
        clip_dir = input_dir / class_name / "clips"
        if not clip_dir.exists():
            continue

        out_pose_dir = output_dir / class_name / "pose_yolo"
        out_clips_dir = output_dir / class_name / "clips"
        out_pose_dir.mkdir(parents=True, exist_ok=True)
        out_clips_dir.mkdir(parents=True, exist_ok=True)

        clips = sorted(clip_dir.glob("*.mp4"))
        stats['total'] += len(clips)

        for clip_idx, clip_path in enumerate(clips):
            clip_key = f"{class_name}/{clip_path.name}"

            # Skip already processed
            if clip_key in progress['completed']:
                stats['processed'] += 1
                continue

            # Skip if output already exists
            if (out_pose_dir / f"{clip_path.stem}.npy").exists():
                progress['completed'].append(clip_key)
                stats['processed'] += 1
                continue

            try:
                # Load video frames
                cap = cv2.VideoCapture(str(clip_path))
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                if not frames:
                    continue

                # RUN YOLO ONCE ON ALL FRAMES (much faster!)
                print(
                    f"  Detecting poses in {clip_path.name}...", end=" ", flush=True)
                all_results = pose_model(frames, verbose=False, stream=True)

                # Cache all results
                cached_keypoints = []
                cached_boxes = []
                for result in all_results:
                    if result.keypoints is not None and len(result.keypoints.data) > 0:
                        cached_keypoints.append(
                            result.keypoints.data.cpu().numpy())
                        cached_boxes.append(result.boxes.xyxy.cpu(
                        ).numpy() if result.boxes is not None else None)
                    else:
                        cached_keypoints.append(None)
                        cached_boxes.append(None)
                print("done")

                # Use middle frame for player selection
                mid_idx = len(frames) // 2
                if cached_keypoints[mid_idx] is None:
                    # Find nearest frame with detections
                    for offset in range(1, len(frames)//2):
                        if mid_idx + offset < len(frames) and cached_keypoints[mid_idx + offset] is not None:
                            mid_idx = mid_idx + offset
                            break
                        if mid_idx - offset >= 0 and cached_keypoints[mid_idx - offset] is not None:
                            mid_idx = mid_idx - offset
                            break
                    else:
                        print(f"  No players detected, skipping")
                        stats['skipped'] += 1
                        continue

                kp_data = cached_keypoints[mid_idx]
                boxes = cached_boxes[mid_idx]
                num_players = len(kp_data)

                # Pre-compute stable player tracking for all frames
                # Use frame-by-frame IoU tracking to prevent ID jumps
                # Players don't move much between consecutive frames
                stable_player_map = {}  # frame_idx -> {detection_idx: stable_player_id}

                # Start from mid frame - assign IDs by x-position (left-to-right)
                if boxes is not None and len(boxes) > 0:
                    mid_centers = [(i, (boxes[i][0] + boxes[i][2]) / 2)
                                   for i in range(len(boxes))]
                    mid_centers.sort(key=lambda x: x[1])  # Sort by x position
                    stable_player_map[mid_idx] = {
                        orig_idx: stable_id for stable_id, (orig_idx, _) in enumerate(mid_centers)}

                    # Track FORWARD from mid frame
                    prev_boxes = boxes.copy()
                    prev_map = stable_player_map[mid_idx].copy()
                    for f_idx in range(mid_idx + 1, len(frames)):
                        stable_player_map[f_idx] = {}
                        if cached_boxes[f_idx] is None:
                            continue
                        curr_boxes = cached_boxes[f_idx]

                        # Match each current detection to previous frame by IoU
                        used_stable_ids = set()
                        matches = []
                        for det_idx in range(len(curr_boxes)):
                            best_iou = 0
                            best_prev_idx = -1
                            for prev_idx in range(len(prev_boxes)):
                                iou = _bbox_iou(
                                    curr_boxes[det_idx], prev_boxes[prev_idx])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_prev_idx = prev_idx
                            matches.append((det_idx, best_prev_idx, best_iou))

                        # Assign stable IDs based on matches (highest IoU first)
                        matches.sort(key=lambda x: -x[2])
                        for det_idx, prev_idx, iou in matches:
                            if iou > 0.2 and prev_idx in prev_map and prev_map[prev_idx] not in used_stable_ids:
                                stable_player_map[f_idx][det_idx] = prev_map[prev_idx]
                                used_stable_ids.add(prev_map[prev_idx])

                        # For unmatched, assign remaining IDs
                        for det_idx in range(len(curr_boxes)):
                            if det_idx not in stable_player_map[f_idx]:
                                for sid in range(num_players):
                                    if sid not in used_stable_ids:
                                        stable_player_map[f_idx][det_idx] = sid
                                        used_stable_ids.add(sid)
                                        break

                        prev_boxes = curr_boxes.copy()
                        prev_map = stable_player_map[f_idx].copy()

                    # Track BACKWARD from mid frame
                    prev_boxes = boxes.copy()
                    prev_map = stable_player_map[mid_idx].copy()
                    for f_idx in range(mid_idx - 1, -1, -1):
                        stable_player_map[f_idx] = {}
                        if cached_boxes[f_idx] is None:
                            continue
                        curr_boxes = cached_boxes[f_idx]

                        # Match each current detection to previous frame by IoU
                        used_stable_ids = set()
                        matches = []
                        for det_idx in range(len(curr_boxes)):
                            best_iou = 0
                            best_prev_idx = -1
                            for prev_idx in range(len(prev_boxes)):
                                iou = _bbox_iou(
                                    curr_boxes[det_idx], prev_boxes[prev_idx])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_prev_idx = prev_idx
                            matches.append((det_idx, best_prev_idx, best_iou))

                        # Assign stable IDs based on matches (highest IoU first)
                        matches.sort(key=lambda x: -x[2])
                        for det_idx, prev_idx, iou in matches:
                            if iou > 0.2 and prev_idx in prev_map and prev_map[prev_idx] not in used_stable_ids:
                                stable_player_map[f_idx][det_idx] = prev_map[prev_idx]
                                used_stable_ids.add(prev_map[prev_idx])

                        # For unmatched, assign remaining IDs
                        for det_idx in range(len(curr_boxes)):
                            if det_idx not in stable_player_map[f_idx]:
                                for sid in range(num_players):
                                    if sid not in used_stable_ids:
                                        stable_player_map[f_idx][det_idx] = sid
                                        used_stable_ids.add(sid)
                                        break

                        prev_boxes = curr_boxes.copy()
                        prev_map = stable_player_map[f_idx].copy()
                else:
                    for f_idx in range(len(frames)):
                        stable_player_map[f_idx] = {
                            i: i for i in range(num_players)}

                # Auto-select player if --auto flag is set
                selected_player = None

                if hasattr(args, 'auto') and args.auto:
                    # Strategy: Pick player with most motion
                    max_motion = -1
                    best_p_idx = 0

                    # Calculate motion for each stable ID
                    for p_idx in range(num_players):
                        motion = 0
                        prev_kpts = None

                        for f_idx in range(len(frames)):
                            if cached_keypoints[f_idx] is None:
                                continue

                            # Find detection index for this stable ID
                            det_idx = -1
                            for d_idx, s_id in stable_player_map.get(f_idx, {}).items():
                                if s_id == p_idx:
                                    det_idx = d_idx
                                    break

                            if det_idx != -1:
                                curr_kpts = cached_keypoints[f_idx][det_idx]
                                if prev_kpts is not None:
                                    motion += _motion_score(curr_kpts.reshape(
                                        1, 17, 3), prev_kpts.reshape(1, 17, 3))
                                prev_kpts = curr_kpts

                        if motion > max_motion:
                            max_motion = motion
                            best_p_idx = p_idx

                    selected_player = best_p_idx
                    print(
                        f"  Auto-selected Player {selected_player + 1} (Motion: {max_motion:.1f})")

                # Manual player selection UI
                frame_idx = mid_idx

                while selected_player is None:
                    display = frames[frame_idx].copy()
                    h, w = display.shape[:2]

                    # Use cached data for current frame if available
                    curr_kp = cached_keypoints[frame_idx] if cached_keypoints[frame_idx] is not None else kp_data
                    curr_boxes = cached_boxes[frame_idx] if cached_boxes[frame_idx] is not None else boxes
                    player_map = stable_player_map.get(frame_idx, {})

                    # Draw header
                    cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
                    cv2.putText(display, f"[{clip_idx+1}/{len(clips)}] {class_name}: {clip_path.name}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display, f"Press 1-{num_players} to select player | S=Skip Q=Quit R=Replay",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(display, f"Frame {frame_idx+1}/{len(frames)} | Players: {len(curr_kp) if curr_kp is not None else 0}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Draw player bounding boxes with STABLE numbers
                    player_centers = []
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

                    if curr_kp is not None:
                        for det_idx in range(len(curr_kp)):
                            # Get stable player ID from tracking
                            stable_id = player_map.get(det_idx, det_idx)
                            kpts = curr_kp[det_idx]
                            color = colors[stable_id % len(colors)]

                            # Draw box if available
                            if curr_boxes is not None and det_idx < len(curr_boxes):
                                x1, y1, x2, y2 = curr_boxes[det_idx].astype(
                                    int)
                                cv2.rectangle(display, (x1, y1),
                                              (x2, y2), color, 2)
                                cv2.putText(
                                    display, f"{stable_id+1}", (x1+5, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                                player_centers.append(
                                    (stable_id, x1, y1, x2, y2))

                            # Draw keypoints
                            for kp in kpts:
                                if kp[2] > 0.3:
                                    cv2.circle(
                                        display, (int(kp[0]), int(kp[1])), 4, color, -1)

                    cv2.imshow("Select Hitting Player", display)

                    # Handle mouse click
                    def mouse_callback(event, x, y, flags, param):
                        nonlocal selected_player
                        if event == cv2.EVENT_LBUTTONDOWN:
                            for p_idx, x1, y1, x2, y2 in player_centers:
                                if x1 <= x <= x2 and y1 <= y <= y2:
                                    selected_player = p_idx
                                    break

                    cv2.setMouseCallback(
                        "Select Hitting Player", mouse_callback)

                    key = cv2.waitKey(50) & 0xFF

                    # Number keys 1-9 for quick selection
                    if ord('1') <= key <= ord('9'):
                        p_idx = key - ord('1')
                        if p_idx < num_players:
                            selected_player = p_idx
                    elif key == ord('s') or key == ord('S'):
                        selected_player = -1  # Skip
                    elif key == ord('q') or key == ord('Q'):
                        cv2.destroyAllWindows()
                        with open(progress_file, 'w') as f:
                            json.dump(progress, f)
                        print(
                            f"\nProgress saved. Processed: {stats['processed']}, Skipped: {stats['skipped']}")
                        return
                    elif key == ord('r') or key == ord('R'):
                        frame_idx = 0
                    else:
                        frame_idx = (frame_idx + 1) % len(frames)

                cv2.destroyAllWindows()

                if selected_player == -1:
                    stats['skipped'] += 1
                    continue

                # Extract keypoints for selected player using cached data + tracking
                tracked_box = boxes[selected_player] if boxes is not None and selected_player < len(
                    boxes) else None
                keypoints_seq = []

                for f_idx in range(len(frames)):
                    if cached_keypoints[f_idx] is not None:
                        kp_data = cached_keypoints[f_idx]
                        frame_boxes = cached_boxes[f_idx]

                        # Track player by bounding box IoU
                        if tracked_box is not None and frame_boxes is not None and len(frame_boxes) > 0:
                            best_iou = 0
                            best_idx = 0
                            for p_idx in range(len(frame_boxes)):
                                iou = _bbox_iou(
                                    tracked_box, frame_boxes[p_idx])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = p_idx

                            if best_iou > 0.1:
                                tracked_box = frame_boxes[best_idx]
                                active_kpts = kp_data[best_idx]
                            else:
                                active_kpts = kp_data[0] if len(
                                    kp_data) > 0 else np.zeros((17, 3))
                        elif selected_player < len(kp_data):
                            active_kpts = kp_data[selected_player]
                        else:
                            active_kpts = kp_data[0] if len(
                                kp_data) > 0 else np.zeros((17, 3))
                    else:
                        active_kpts = np.zeros((17, 3))

                    keypoints_seq.append(active_kpts)

                # Save
                np.save(
                    str(out_pose_dir / f"{clip_path.stem}.npy"), np.array(keypoints_seq))
                if not (out_clips_dir / clip_path.name).exists():
                    shutil.copy2(str(clip_path), str(
                        out_clips_dir / clip_path.name))

                progress['completed'].append(clip_key)
                stats['processed'] += 1
                print(f"  ✓ Player {selected_player + 1} selected")

                if stats['processed'] % 10 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)

            except Exception as e:
                print(f"    Error: {clip_path.name}: {e}")
                import traceback
                traceback.print_exc()

    cv2.destroyAllWindows()

    with open(progress_file, 'w') as f:
        json.dump(progress, f)

    print("\n" + "="*60)
    print(
        f"Total: {stats['total']}, Processed: {stats['processed']}, Skipped: {stats['skipped']}")
    print(f"Output: {output_dir}")
    print("\nNext step: python train.py --data dataset/player_enhanced")


def _bbox_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 < x1 or y2 < y1:
        return 0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def _motion_score(kpts, prev_kpts):
    if prev_kpts is None:
        return 0
    disp = np.linalg.norm(kpts[:, :2] - prev_kpts[:, :2], axis=1)
    return np.sum(disp * kpts[:, 2] * prev_kpts[:, 2]) / (np.sum(kpts[:, 2]) + 1e-6)


# ==========================================
# AUTO ANNOTATION
# ==========================================

def auto_annotate(args):
    """Automatically detect, classify, and sort shots from a video"""
    print("="*60)
    print("Auto-Annotation Mode")
    print("="*60)

    dataset_dir = Path(args.dataset_dir)
    sorted_dir = dataset_dir / "sorted"
    resolution = tuple(map(int, args.resolution.split('x')))

    # Ensure directories exist
    categories = {
        '1': 'overhead', '2': 'serve', '3': 'forehand',
        '4': 'backhand', '5': 'lob', '6': 'other'
    }
    for cat in categories.values():
        (sorted_dir / cat / "clips").mkdir(parents=True, exist_ok=True)
        (sorted_dir / cat / "pose").mkdir(parents=True, exist_ok=True)

    # Load models
    print("\nLoading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO(args.pose_model)

    if not os.path.exists(args.classifier_model):
        print(f"Error: Classifier model not found at {args.classifier_model}")
        return

    classifier = ShotClassifier(
        args.classifier_model, pose_model, device=device)

    # Video setup
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    # Sliding window parameters
    window_size = 32
    stride = 4  # Check every 4 frames for better temporal resolution
    confidence_threshold = args.conf_threshold

    frame_buffer = deque(maxlen=window_size)
    pose_buffer = deque(maxlen=window_size)

    # Find next global index for naming
    existing_indices = []
    for cat in categories.values():
        clips = list((sorted_dir / cat / "clips").glob("shot_*.mp4"))
        for c in clips:
            try:
                idx = int(c.stem.split('_')[1])
                existing_indices.append(idx)
            except:
                pass
    next_idx = max(existing_indices) + 1 if existing_indices else 0

    print(f"Starting auto-annotation from index {next_idx}...")
    print(f"Confidence threshold: {confidence_threshold}")

    pbar = tqdm(total=total_frames)
    frame_idx = 0
    shots_found = 0
    last_shot_frame = -999

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for processing
        frame_proc = cv2.resize(frame, resolution)

        # Extract pose
        results = pose_model(frame_proc, verbose=False)
        kp = np.zeros(51)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Get first person
            kp = results[0].keypoints.data[0].cpu().numpy().flatten()

        frame_buffer.append(frame_proc)
        pose_buffer.append(kp)

        # Run inference
        if len(frame_buffer) == window_size and frame_idx % stride == 0:
            # Check cooldown (1 second)
            if frame_idx - last_shot_frame > fps:
                label, conf = classifier.classify(
                    list(frame_buffer), np.array(list(pose_buffer)))

                if label and label != 'other' and conf >= confidence_threshold:
                    # Found a shot!
                    shots_found += 1
                    last_shot_frame = frame_idx

                    # Save clip
                    shot_name = f"shot_{next_idx:04d}"
                    save_path = sorted_dir / label / \
                        "clips" / f"{shot_name}.mp4"
                    pose_path = sorted_dir / label / \
                        "pose" / f"{shot_name}.npy"

                    # Save video
                    out = cv2.VideoWriter(
                        str(save_path),
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution
                    )
                    for f in frame_buffer:
                        out.write(f)
                    out.release()

                    # Save pose
                    np.save(str(pose_path), np.array(list(pose_buffer)))

                    pbar.write(
                        f"  Found {label.upper()} ({conf:.2f}) -> {shot_name}")
                    next_idx += 1

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"\nAuto-annotation complete. Found {shots_found} shots.")
    print("Next step: python annotate.py validate")


# ==========================================
# VALIDATION
# ==========================================

def validate_dataset(args):
    """Manually validate and correct auto-sorted clips"""
    print("="*60)
    print("Dataset Validation Mode")
    print("="*60)

    dataset_dir = Path(args.dataset_dir)
    sorted_dir = dataset_dir / "sorted"

    categories = {
        '1': 'overhead', '2': 'serve', '3': 'forehand',
        '4': 'backhand', '5': 'lob', '6': 'other'
    }

    # Collect all clips
    all_clips = []
    for cat in categories.values():
        clip_dir = sorted_dir / cat / "clips"
        if clip_dir.exists():
            for f in clip_dir.glob("*.mp4"):
                all_clips.append({
                    'path': f,
                    'category': cat,
                    'pose_path': sorted_dir / cat / "pose" / (f.stem + ".npy")
                })

    # Sort by filename to keep order
    all_clips.sort(key=lambda x: x['path'].name)

    print(f"Found {len(all_clips)} clips to validate.")
    print("Controls: 1-6=Move to Class | SPACE=Confirm/Next | D=Delete | Q=Quit | R=Replay")

    idx = 0
    while idx < len(all_clips):
        clip_info = all_clips[idx]
        current_cat = clip_info['category']
        video_path = clip_info['path']

        if not video_path.exists():
            idx += 1
            continue

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            idx += 1
            continue

        frame_idx = 0
        action = None

        while action is None:
            display = frames[frame_idx].copy()
            h, w = display.shape[:2]

            # UI
            cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(display, f"[{idx+1}/{len(all_clips)}] {video_path.name}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Current: {current_cat.upper()}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "1:Overhead 2:Serve 3:Forehand 4:Backhand 5:Lob 6:Other",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Validator", display)

            key = cv2.waitKey(30) & 0xFF

            if chr(key) in categories:
                new_cat = categories[chr(key)]
                if new_cat != current_cat:
                    # Move file
                    new_dir = sorted_dir / new_cat / "clips"
                    new_pose_dir = sorted_dir / new_cat / "pose"

                    shutil.move(str(video_path), str(
                        new_dir / video_path.name))
                    if clip_info['pose_path'].exists():
                        shutil.move(str(clip_info['pose_path']), str(
                            new_pose_dir / clip_info['pose_path'].name))

                    print(
                        f"  Moved {video_path.name}: {current_cat} -> {new_cat}")
                    clip_info['category'] = new_cat
                    clip_info['path'] = new_dir / video_path.name
                    clip_info['pose_path'] = new_pose_dir / \
                        clip_info['pose_path'].name
                    action = 'next'
            elif key == ord(' '):
                action = 'next'
            elif key == ord('d') or key == ord('D'):
                # Delete
                video_path.unlink()
                if clip_info['pose_path'].exists():
                    clip_info['pose_path'].unlink()
                print(f"  Deleted {video_path.name}")
                action = 'next'
            elif key == ord('q') or key == ord('Q'):
                action = 'quit'
            elif key == ord('r') or key == ord('R'):
                frame_idx = 0
            else:
                frame_idx = (frame_idx + 1) % len(frames)

        if action == 'quit':
            break

        idx += 1

    cv2.destroyAllWindows()
    print("\nValidation complete.")


# ==========================================
# FINAL VERIFICATION
# ==========================================

def verify_final_dataset(args):
    """Verify the final training data (clips + pose) in player_enhanced"""
    print("="*60)
    print("Final Dataset Verification Mode")
    print("="*60)

    dataset_dir = Path(args.dataset_dir)
    enhanced_dir = dataset_dir / "player_enhanced"

    if not enhanced_dir.exists():
        print(f"Error: {enhanced_dir} does not exist. Run 'prepare' first.")
        return

    # Collect all samples
    samples = []
    categories = [d.name for d in enhanced_dir.iterdir() if d.is_dir()]

    categories_map = {
        '1': 'overhead', '2': 'serve', '3': 'forehand',
        '4': 'backhand', '5': 'lob', '6': 'other'
    }

    for cat in categories:
        clip_dir = enhanced_dir / cat / "clips"
        pose_dir = enhanced_dir / cat / "pose_yolo"

        if clip_dir.exists():
            for f in clip_dir.glob("*.mp4"):
                pose_file = pose_dir / (f.stem + ".npy")
                if pose_file.exists():
                    samples.append({
                        'path': f,
                        'category': cat,
                        'pose_path': pose_file
                    })

    samples.sort(key=lambda x: (x['category'], x['path'].name))

    # Load progress
    progress_file = dataset_dir / "verify_progress.json"
    verified_files = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                verified_files = set(json.load(f))
            print(
                f"Loaded progress: {len(verified_files)} clips already verified.")
        except:
            print("Could not load progress file.")

    # Filter out verified samples
    samples = [s for s in samples if str(s['path'].name) not in verified_files]

    print(f"Found {len(samples)} samples to verify.")
    print("Controls: SPACE=Next | D=Delete | Q=Quit | R=Replay")
    print("          1-6=Move to Category | P=Change Player")

    pose_model_loaded = None

    idx = 0
    while idx < len(samples):
        sample = samples[idx]
        video_path = sample['path']
        pose_path = sample['pose_path']
        category = sample['category']

        # Load pose data
        try:
            pose_data = np.load(str(pose_path))
        except:
            print(f"Error loading pose: {pose_path}")
            idx += 1
            continue

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            idx += 1
            continue

        frame_idx = 0
        action = None

        while action is None:
            display = frames[frame_idx].copy()
            h, w = display.shape[:2]

            # Draw pose
            if frame_idx < len(pose_data):
                kpts = pose_data[frame_idx]
                # Reshape if flat
                if kpts.ndim == 1:
                    kpts = kpts.reshape(-1, 3)

                # Draw keypoints
                for kp in kpts:
                    # Check if normalized (0-1) or pixel coords
                    if np.max(kpts[:, :2]) <= 1.0:
                        x, y = int(kp[0] * w), int(kp[1] * h)
                    else:
                        x, y = int(kp[0]), int(kp[1])

                    if x > 0 and y > 0:
                        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)

            # UI

            cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(display, f"[{idx+1}/{len(samples)}] {video_path.name}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"LABEL: {category.upper()}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: Next | D: Delete | Q: Quit | P: Change Player",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, "1:Overhead 2:Serve 3:Forehand 4:Backhand 5:Lob 6:Other",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("Final Verification", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):
                action = 'next'
                verified_files.add(video_path.name)
            elif key == ord('d') or key == ord('D'):
                video_path.unlink()
                pose_path.unlink()
                print(f"  Deleted {video_path.name}")
                action = 'next'
                verified_files.add(video_path.name)
            elif key == ord('q') or key == ord('Q'):
                action = 'quit'
            elif key == ord('r') or key == ord('R'):
                frame_idx = 0
            elif chr(key) in categories_map:
                new_cat = categories_map[chr(key)]
                if new_cat != category:
                    new_clip_dir = enhanced_dir / new_cat / "clips"
                    new_pose_dir = enhanced_dir / new_cat / "pose_yolo"
                    new_clip_dir.mkdir(parents=True, exist_ok=True)
                    new_pose_dir.mkdir(parents=True, exist_ok=True)

                    new_video_path = new_clip_dir / video_path.name
                    new_pose_path = new_pose_dir / pose_path.name

                    shutil.move(str(video_path), str(new_video_path))
                    shutil.move(str(pose_path), str(new_pose_path))

                    print(f"  Moved {video_path.name} -> {new_cat}")
                    sample['category'] = new_cat
                    sample['path'] = new_video_path
                    sample['pose_path'] = new_pose_path
                    video_path = new_video_path
                    pose_path = new_pose_path
                    category = new_cat
                    verified_files.add(video_path.name)
            elif key == ord('p') or key == ord('P'):
                if pose_model_loaded is None:
                    print("\nLoading pose model for re-selection...")
                    pose_model_loaded = YOLO(args.pose_model)

                print("  Reselecting player...")
                cv2.destroyAllWindows()
                new_pose = reselect_player(video_path, pose_model_loaded)
                if new_pose is not None:
                    np.save(str(pose_path), new_pose)
                    pose_data = new_pose
                    print("  Player updated!")
                else:
                    print("  Selection cancelled.")
            else:
                frame_idx = (frame_idx + 1) % len(frames)

        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(list(verified_files), f)

        if action == 'quit':
            break

        idx += 1

    cv2.destroyAllWindows()
    print("\nVerification complete.")


# ==========================================
# HELPERS
# ==========================================


def reselect_player(video_path, pose_model):
    """Interactive player selection for a single video (reused logic)"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    print(f"  Reprocessing {video_path.name}...", end=" ", flush=True)
    results = pose_model(frames, verbose=False, stream=True)

    cached_keypoints = []
    cached_boxes = []
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            cached_keypoints.append(r.keypoints.data.cpu().numpy())
            cached_boxes.append(r.boxes.xyxy.cpu().numpy())
        else:
            cached_keypoints.append(None)
            cached_boxes.append(None)
    print("done")

    # Tracking logic (simplified from prepare_dataset)
    mid_idx = len(frames) // 2
    kp_data = cached_keypoints[mid_idx] if cached_keypoints[mid_idx] is not None else [
    ]
    boxes = cached_boxes[mid_idx] if cached_boxes[mid_idx] is not None else []
    num_players = len(kp_data)

    # UI for selection
    selected_player = None
    frame_idx = mid_idx

    while selected_player is None:
        display = frames[frame_idx].copy()
        h, w = display.shape[:2]

        curr_kp = cached_keypoints[frame_idx]
        curr_boxes = cached_boxes[frame_idx]

        cv2.putText(display, "Select Player (Click or 1-4)",
                    (10, 30), 0, 0.7, (0, 255, 255), 2)

        player_centers = []
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

        if curr_kp is not None:
            for i in range(len(curr_kp)):
                color = colors[i % len(colors)]
                if curr_boxes is not None and i < len(curr_boxes):
                    x1, y1, x2, y2 = map(int, curr_boxes[i])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, str(i+1), (x1, y1-5), 0, 1, color, 2)
                    player_centers.append((i, x1, y1, x2, y2))

                for kp in curr_kp[i]:
                    if kp[2] > 0.3:
                        cv2.circle(
                            display, (int(kp[0]), int(kp[1])), 3, color, -1)

        cv2.imshow("Reselect Player", display)

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_player
            if event == cv2.EVENT_LBUTTONDOWN:
                for p_idx, x1, y1, x2, y2 in player_centers:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_player = p_idx
                        break
        cv2.setMouseCallback("Reselect Player", mouse_callback)

        key = cv2.waitKey(0) & 0xFF
        if ord('1') <= key <= ord('9'):
            p_idx = key - ord('1')
            if p_idx < num_players:
                selected_player = p_idx
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Extract keypoints for selected player (simple tracking)
    keypoints_seq = []
    tracked_box = boxes[selected_player] if boxes is not None and selected_player < len(
        boxes) else None

    for i in range(len(frames)):
        if cached_keypoints[i] is not None:
            frame_boxes = cached_boxes[i]
            kp_data = cached_keypoints[i]

            if tracked_box is not None and frame_boxes is not None:
                best_iou = 0
                best_idx = 0
                for j, box in enumerate(frame_boxes):
                    iou = _bbox_iou(tracked_box, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_iou > 0.1:
                    tracked_box = frame_boxes[best_idx]
                    keypoints_seq.append(kp_data[best_idx])
                else:
                    keypoints_seq.append(np.zeros((17, 3)))
            elif selected_player < len(kp_data):
                keypoints_seq.append(kp_data[selected_player])
            else:
                keypoints_seq.append(np.zeros((17, 3)))
        else:
            keypoints_seq.append(np.zeros((17, 3)))

    return np.array(keypoints_seq)


def main():
    parser = argparse.ArgumentParser(
        description="Padel Dataset Annotation Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Extract
    extract_parser = subparsers.add_parser(
        'extract', help='Extract shots from video')
    extract_parser.add_argument(
        '--video', type=str, required=True, help='Path to input video')
    extract_parser.add_argument(
        '--dataset_dir', type=str, default='dataset', help='Dataset root directory')
    extract_parser.add_argument('--tracknet_model', type=str,
                                default='TrackNet_best.pt', help='Path to TrackNet model')
    extract_parser.add_argument(
        '--pose_model', type=str, default='yolov8n-pose.pt', help='Path to YOLO pose model')
    extract_parser.add_argument(
        '--resolution', type=str, default='1280x720', help='Processing resolution')
    extract_parser.add_argument('--classifier_model', type=str,
                                default=None, help='Optional classifier for live prediction')

    # Sort
    sort_parser = subparsers.add_parser('sort', help='Manually sort clips')
    sort_parser.add_argument('--dataset_dir', type=str,
                             default='dataset', help='Dataset root directory')
    sort_parser.add_argument('--resolution', type=str,
                             default='1280x720', help='Display resolution')
    sort_parser.add_argument('--classifier_model', type=str,
                             default=None, help='Optional classifier for live prediction')

    # Auto Annotate
    auto_parser = subparsers.add_parser(
        'auto_annotate', help='Automatically sort clips using classifier')
    auto_parser.add_argument(
        '--video', type=str, required=True, help='Path to input video')
    auto_parser.add_argument('--dataset_dir', type=str,
                             default='dataset', help='Dataset root directory')
    auto_parser.add_argument('--classifier_model', type=str,
                             default='models/best_model.pth', help='Path to classifier model')
    auto_parser.add_argument(
        '--pose_model', type=str, default='yolov8n-pose.pt', help='Path to YOLO pose model')
    auto_parser.add_argument('--resolution', type=str,
                             default='1280x720', help='Processing resolution')
    auto_parser.add_argument('--conf_threshold', type=float,
                             default=0.8, help='Confidence threshold')

    # Validate
    validate_parser = subparsers.add_parser(
        'validate', help='Validate sorted dataset')
    validate_parser.add_argument(
        '--dataset_dir', type=str, default='dataset', help='Dataset root directory')

    # Prepare
    prepare_parser = subparsers.add_parser(
        'prepare', help='Prepare final dataset (extract pose)')
    prepare_parser.add_argument(
        '--dataset_dir', type=str, default='dataset', help='Dataset root directory')
    prepare_parser.add_argument(
        '--pose_model', type=str, default='yolov8x-pose.pt', help='Path to YOLO pose model')
    prepare_parser.add_argument(
        '--auto', action='store_true', help='Automatically select player based on motion')
    prepare_parser.add_argument(
        '--reset', action='store_true', help='Reset progress')
    prepare_parser.add_argument(
        '--classes', type=str, default=None, help='Comma-separated list of classes to process')

    # Verify Final
    verify_parser = subparsers.add_parser(
        'verify_final', help='Verify final dataset')
    verify_parser.add_argument(
        '--dataset_dir', type=str, default='dataset', help='Dataset root directory')
    verify_parser.add_argument('--pose_model', type=str, default='yolov8x-pose.pt',
                               help='Path to YOLO pose model (for re-selection)')

    args = parser.parse_args()

    if args.command == 'extract':
        extract_shots(args)
    elif args.command == 'sort':
        sort_dataset(args)
    elif args.command == 'auto_annotate':
        auto_annotate(args)
    elif args.command == 'validate':
        validate_dataset(args)
    elif args.command == 'prepare':
        prepare_dataset(args)
    elif args.command == 'verify_final':
        verify_final_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
