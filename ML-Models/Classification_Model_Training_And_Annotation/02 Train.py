#!/usr/bin/env python3
"""
Padel Shot & Side Classification Training

Trains an enhanced video + pose classifier for padel shot classification.
Now includes Multi-Task Learning to predict:
1. Shot Type (Groundstroke, Lob, etc.)
2. Player Side (Left/Right)

Features:
- Multi-Task Loss (CrossEntropy for Shot + BCE for Side)
- Class weight balancing for imbalanced datasets
- Per-class accuracy monitoring
- Confusion Matrix logging
- Early stopping
- Learning rate scheduling

Usage:
    python train.py --data dataset/player_enhanced --epochs 50
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import video as video_models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import cv2
from collections import defaultdict, Counter


# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


class EnhancedVideoClassifier(nn.Module):
    """
    Enhanced classifier combining video features (R(2+1)D-18) with pose keypoints (Bi-LSTM).

    Heads:
    1. shot_head: Classifies the shot type (num_classes)
    2. side_head: Classifies player side (1 output, sigmoid)
    """

    def __init__(self, num_classes, pose_input_size=51, pose_hidden_size=256, pretrained_video=True, freeze_backbone=False):
        super().__init__()

        # Video branch: Upgrade to R(2+1)D-18
        self.video_model = video_models.r2plus1d_18(
            weights=video_models.R2Plus1D_18_Weights.DEFAULT if pretrained_video else None
        )

        if freeze_backbone and pretrained_video:
            print("Freezing video backbone layers...")
            for param in self.video_model.parameters():
                param.requires_grad = False
        else:
            print("Video backbone layers trainable (fine-tuning enabled)")

        num_video_ftrs = self.video_model.fc.in_features
        self.video_model.fc = nn.Identity()

        # Pose branch: Bidirectional LSTM for better temporal context
        self.pose_lstm = nn.LSTM(
            pose_input_size, pose_hidden_size,
            num_layers=3, batch_first=True, dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.pose_attention = nn.Sequential(
            nn.Linear(pose_hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Shared Fusion features
        fusion_size = num_video_ftrs + (pose_hidden_size * 2)
        self.shared_features = nn.Sequential(
            nn.Linear(fusion_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Multi-Heads (Two separate outputs)
        self.shot_head = nn.Linear(512, num_classes)
        self.side_head = nn.Linear(512, 1)

    def forward(self, video_x, pose_x):
        # Video features
        video_features = self.video_model(video_x)

        # Pose features with attention
        pose_out, _ = self.pose_lstm(pose_x)
        attention_weights = self.pose_attention(pose_out)
        pose_features = torch.sum(pose_out * attention_weights, dim=1)

        # Fusion
        combined = torch.cat((video_features, pose_features), dim=1)
        shared = self.shared_features(combined)

        # Return TUPLE of (Shot, Side)
        return self.shot_head(shared), self.side_head(shared)


# ==========================================
# DATASET
# ==========================================

class PadelShotDataset(Dataset):
    """
    Dataset for loading padel shot video clips with YOLO pose keypoints.
    Calculates Player Side (Left/Right) automatically from pose data.
    """

    def __init__(self, file_list, labels_map, clip_len=32, resolution=(112, 112), augment=False, augment_level='strong'):
        self.file_list = file_list
        self.labels_map = labels_map
        self.clip_len = clip_len
        self.resolution = resolution
        self.augment = augment
        self.augment_level = augment_level

        # ImageNet normalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        # COCO keypoint indices for horizontal flip (swap left/right)
        self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8),
                           (9, 10), (11, 12), (13, 14), (15, 16)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        clip_path, pose_path, label_name = self.file_list[idx]
        label = self.labels_map[label_name]

        # Load video
        try:
            video_frames = self._load_video(clip_path)
        except Exception:
            # Silently handle error to prevent spam, return black frames
            video_frames = torch.zeros((3, self.clip_len, *self.resolution))

        # Load pose (Safe loading)
        pose_data = np.zeros((self.clip_len, 51))
        try:
            raw_pose = np.load(pose_path, allow_pickle=True)
            if raw_pose.ndim == 3 and raw_pose.shape[1:] == (17, 3):
                raw_pose = np.nan_to_num(raw_pose)
                raw_pose = self._sample_sequence(raw_pose, self.clip_len)

                # Normalize pose to 0-1 range if needed
                if np.max(raw_pose) > 1.0:
                    raw_pose[:, :, 0] /= 1280.0
                    raw_pose[:, :, 1] /= 720.0

                pose_data = raw_pose.reshape(self.clip_len, -1)
        except Exception:
            pass  # Keep zero pose

        # Apply augmentation (Critical: Must happen BEFORE side calculation)
        if self.augment:
            video_frames, pose_data = self._augment(video_frames, pose_data)

        # --- AUTO-LABEL SIDE ---
        # Calculate side based on average X-coordinate of the player
        if isinstance(pose_data, torch.Tensor):
            pose_check = pose_data.numpy()
        else:
            pose_check = pose_data

        pose_reshaped = pose_check.reshape(self.clip_len, 17, 3)
        # Filter out missing keypoints (0.0)
        x_coords = pose_reshaped[:, :, 0]
        valid_x = x_coords[x_coords > 0.01]

        if len(valid_x) > 5:  # Require at least 5 valid keypoints in clip
            mean_x = np.mean(valid_x)
            # 0.0 = Left, 1.0 = Right
            side_val = 1.0 if mean_x >= 0.5 else 0.0
        else:
            side_val = 0.5  # Unknown/Missing Pose

        side_label = torch.tensor([side_val], dtype=torch.float32)

        return video_frames, torch.from_numpy(pose_data).float() if isinstance(pose_data, np.ndarray) else pose_data, label, side_label

    def _augment(self, video, pose):
        """Apply comprehensive data augmentation to video and pose data."""
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()

        # ===== TEMPORAL AUGMENTATIONS =====
        # Random temporal shift
        if random.random() > 0.4:
            shift = random.randint(-5, 5)
            if shift != 0:
                video = torch.roll(video, shift, dims=1)
                pose = np.roll(pose, shift, axis=0)

        # Random temporal speed
        if random.random() > 0.5:
            speed = random.choice([0.7, 0.85, 1.0, 1.15, 1.3])
            t = video.shape[1]
            new_indices = np.clip(np.arange(0, t, speed).astype(int), 0, t-1)
            if len(new_indices) >= self.clip_len:
                new_indices = new_indices[:self.clip_len]
            else:
                new_indices = np.pad(
                    new_indices, (0, self.clip_len - len(new_indices)), 'edge')
            video = video[:, new_indices]
            pose = pose[new_indices]

        # Random temporal reversal
        if random.random() > 0.85:
            video = torch.flip(video, dims=[1])
            pose = np.flip(pose, axis=0).copy()

        # ===== SPATIAL AUGMENTATIONS =====
        # Horizontal flip (Critical: Also swaps Left/Right side label implicitly via coordinate flip)
        if random.random() > 0.5:
            video = torch.flip(video, dims=[3])  # Flip width
            pose_reshaped = pose.reshape(-1, 17, 3)
            pose_reshaped[:, :, 0] = 1.0 - pose_reshaped[:, :, 0]  # Flip X
            # Swap Keypoints
            for left, right in self.flip_pairs:
                pose_reshaped[:, [left, right]
                              ] = pose_reshaped[:, [right, left]]
            pose = pose_reshaped.reshape(-1, 51)

        # Random crop/zoom
        if random.random() > 0.6:
            scale = random.uniform(0.85, 1.0)
            _, t, h, w = video.shape
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            video = video[:, :, top:top+new_h, left:left+new_w]
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3),
                size=(h, w), mode='bilinear', align_corners=False
            ).permute(1, 0, 2, 3)

        # ===== COLOR AUGMENTATIONS =====
        if random.random() > 0.4:  # Brightness
            video = video * random.uniform(0.7, 1.3)

        if random.random() > 0.5:  # Contrast
            mean = video.mean()
            video = (video - mean) * random.uniform(0.8, 1.2) + mean

        if random.random() > 0.6:  # Saturation
            gray = video.mean(dim=0, keepdim=True)
            video = gray + random.uniform(0.8, 1.2) * (video - gray)

        if random.random() > 0.7:  # Hue
            hue_shift = random.uniform(-0.1, 0.1)
            video = torch.roll(video, shifts=int(
                hue_shift * video.shape[0]), dims=0)

        # ===== BLUR & NOISE =====
        if random.random() > 0.7:  # Blur
            sigma = random.uniform(0.1, 2.0)
            video = transforms.functional.gaussian_blur(
                video, kernel_size=5, sigma=sigma)

        if random.random() > 0.7:  # Noise
            noise = torch.randn_like(video) * 0.05
            video = video + noise

        video = torch.clamp(video, -2.5, 2.5)

        # ===== CUTOUT =====
        if self.augment_level == 'strong' and random.random() > 0.7:
            _, t, h, w = video.shape
            mask_h = random.randint(h // 8, h // 4)
            mask_w = random.randint(w // 8, w // 4)
            top = random.randint(0, h - mask_h)
            left = random.randint(0, w - mask_w)
            frame_start = random.randint(0, max(0, t - t//4))
            frame_end = min(t, frame_start + random.randint(t//8, t//4))
            video[:, frame_start:frame_end,
                  top:top+mask_h, left:left+mask_w] = 0

        # ===== POSE AUGMENTATIONS =====
        if random.random() > 0.3:  # Noise
            noise_scale = 0.03 if self.augment_level == 'light' else 0.05
            noise = np.random.normal(0, noise_scale, pose.shape)
            pose = pose + noise

        # Dropout Keypoints
        if self.augment_level in ['medium', 'strong'] and random.random() > 0.7:
            pose_reshaped = pose.reshape(-1, 17, 3)
            n_drop = random.randint(1, 3)
            drop_kpts = random.sample(range(17), n_drop)
            drop_frames = random.sample(
                range(len(pose_reshaped)), min(len(pose_reshaped)//3, 8))
            for f in drop_frames:
                for k in drop_kpts:
                    pose_reshaped[f, k, :] = 0
            pose = pose_reshaped.reshape(-1, 51)

        # Scaling
        if random.random() > 0.6:
            scale = random.uniform(0.9, 1.1)
            pose_reshaped = pose.reshape(-1, 17, 3)
            pose_reshaped[:, :, :2] *= scale
            pose = pose_reshaped.reshape(-1, 51)

        return video, pose

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames in {video_path}")

        indices = self._sample_indices(len(frames), self.clip_len)
        frames = [frames[i] for i in indices]
        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).float() / 255.0
        frames_tensor = (frames_tensor - self.norm_mean) / self.norm_std
        return frames_tensor

    def _sample_indices(self, total, target):
        if total >= target:
            return np.linspace(0, total - 1, target, dtype=int)
        else:
            indices = np.arange(total)
            return np.pad(indices, (0, target - total), 'edge')

    def _sample_sequence(self, seq, target_len):
        if len(seq) >= target_len:
            indices = np.linspace(0, len(seq) - 1, target_len, dtype=int)
            return seq[indices]
        else:
            return np.pad(seq, ((0, target_len - len(seq)), (0, 0), (0, 0)), 'edge')


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, loader, criterion_shot, criterion_side, optimizer, device, scaler=None):
    """Train for one epoch with Multi-Task Loss."""
    model.train()
    metrics = {'loss': 0, 'shot_correct': 0, 'side_correct': 0, 'total': 0}
    per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for videos, poses, labels, side_labels in tqdm(loader, desc="Training"):
        videos = videos.to(device, non_blocking=True)
        poses = poses.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        side_labels = side_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Training
        if scaler:
            with torch.cuda.amp.autocast():
                # Expecting TUPLE output here
                shot_out, side_out = model(videos, poses)

                loss_shot = criterion_shot(shot_out, labels)
                loss_side = criterion_side(side_out, side_labels)

                # Weighted loss: 1.0 for Shot, 0.5 for Side
                total_loss = loss_shot + (0.5 * loss_side)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            shot_out, side_out = model(videos, poses)
            loss_shot = criterion_shot(shot_out, labels)
            loss_side = criterion_side(side_out, side_labels)
            total_loss = loss_shot + (0.5 * loss_side)
            total_loss.backward()
            optimizer.step()

        metrics['loss'] += total_loss.item()

        # Calculate Accuracies
        _, shot_pred = torch.max(shot_out.data, 1)
        side_pred = (torch.sigmoid(side_out) > 0.5).float()

        metrics['total'] += labels.size(0)
        metrics['shot_correct'] += (shot_pred == labels).sum().item()
        metrics['side_correct'] += (side_pred == side_labels).sum().item()

        for l, p in zip(labels, shot_pred):
            l_item = l.item()
            per_class_stats[l_item]['total'] += 1
            if p.item() == l_item:
                per_class_stats[l_item]['correct'] += 1

    return {
        'loss': metrics['loss'] / len(loader),
        'shot_acc': 100 * metrics['shot_correct'] / metrics['total'],
        'side_acc': 100 * metrics['side_correct'] / metrics['total'],
        'per_class': {k: 100 * v['correct'] / v['total'] for k, v in per_class_stats.items()}
    }


def validate_epoch(model, loader, criterion_shot, criterion_side, device):
    """Validate for one epoch."""
    model.eval()
    metrics = {'loss': 0, 'shot_correct': 0,
               'side_correct': 0, 'total': 0, 'side_total': 0}
    all_preds, all_labels = [], []
    per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        for videos, poses, labels, side_labels in tqdm(loader, desc="Validating"):
            videos = videos.to(device, non_blocking=True)
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            side_labels = side_labels.to(device, non_blocking=True)

            shot_out, side_out = model(videos, poses)
            loss_shot = criterion_shot(shot_out, labels)
            loss_side = criterion_side(side_out, side_labels)
            total_loss = loss_shot + (0.5 * loss_side)

            metrics['loss'] += total_loss.item()

            # Shot Acc
            _, shot_pred = torch.max(shot_out.data, 1)
            metrics['total'] += labels.size(0)
            metrics['shot_correct'] += (shot_pred == labels).sum().item()

            # Side Acc (Filtering out 0.5 "Unknown" labels)
            side_pred = (torch.sigmoid(side_out) > 0.5).float()

            valid_side_mask = (side_labels != 0.5).squeeze()
            # If batch has any valid sides
            if valid_side_mask.sum() > 0:
                valid_preds = side_pred.squeeze()[valid_side_mask]
                valid_labels = side_labels.squeeze()[valid_side_mask]
                metrics['side_correct'] += (valid_preds ==
                                            valid_labels).sum().item()
                metrics['side_total'] += valid_side_mask.sum().item()

            all_preds.extend(shot_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for l, p in zip(labels, shot_pred):
                l_item = l.item()
                per_class_stats[l_item]['total'] += 1
                if p.item() == l_item:
                    per_class_stats[l_item]['correct'] += 1

    # Calculate valid side accuracy
    side_acc = 100 * metrics['side_correct'] / \
        metrics['side_total'] if metrics['side_total'] > 0 else 0.0

    return (
        metrics['loss'] / len(loader),
        100 * metrics['shot_correct'] / metrics['total'],
        side_acc,
        {k: 100 * v['correct'] / v['total']
            for k, v in per_class_stats.items()},
        all_labels,
        all_preds
    )


# ==========================================
# MAIN
# ==========================================

def balance_dataset(files, method='oversample'):
    """Balance dataset by oversampling minority classes or undersampling majority classes."""
    class_files = defaultdict(list)
    for f in files:
        class_files[f[2]].append(f)

    class_counts = {k: len(v) for k, v in class_files.items()}
    print("\nOriginal class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:15s}: {count:4d}")

    if method == 'oversample':
        max_count = max(class_counts.values())
        balanced_files = []
        for class_samples in class_files.values():
            multiplier = max_count // len(class_samples)
            remainder = max_count % len(class_samples)
            balanced_files.extend(class_samples * multiplier)
            balanced_files.extend(random.sample(class_samples, remainder))
        print(f"\nAfter oversampling: {len(balanced_files)} samples")
    elif method == 'undersample':
        min_count = min(class_counts.values())
        balanced_files = []
        for class_samples in class_files.values():
            balanced_files.extend(random.sample(class_samples, min_count))
        print(f"\nAfter undersampling: {len(balanced_files)} samples")
    else:
        balanced_files = files
        print(f"\nNo balancing applied: {len(balanced_files)} samples")

    random.shuffle(balanced_files)
    return balanced_files


def find_pose_file(clip_path, class_dir):
    """Find the corresponding pose file for a clip."""
    shot_name = os.path.splitext(os.path.basename(clip_path))[0]
    pose_file = f"{shot_name}.npy"
    pose_path = os.path.join(class_dir, 'pose', pose_file)
    if os.path.exists(pose_path):
        return pose_path
    pose_path = os.path.join(class_dir, 'pose_yolo', pose_file)
    if os.path.exists(pose_path):
        return pose_path
    return None


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Find classes
    classes = sorted([d for d in os.listdir(args.data)
                     if os.path.isdir(os.path.join(args.data, d))])
    labels_map = {name: i for i, name in enumerate(classes)}
    idx_to_label = {i: name for name, i in labels_map.items()}
    num_classes = len(classes)

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Epochs: {args.epochs}")
    print(f"Balance: {args.balance}")
    print(f"Augment: {args.augment_level}")

    # Collect files
    all_files = []
    class_counts = defaultdict(int)

    for class_name in classes:
        class_dir = os.path.join(args.data, class_name)
        clip_dir = os.path.join(class_dir, 'clips')
        if not os.path.exists(clip_dir):
            continue
        for clip_file in os.listdir(clip_dir):
            if not clip_file.endswith('.mp4'):
                continue
            clip_path = os.path.join(clip_dir, clip_file)
            pose_path = find_pose_file(clip_path, class_dir)
            if pose_path:
                all_files.append((clip_path, pose_path, class_name))
                class_counts[class_name] += 1
            else:
                # Log warning but continue
                print(f"Warning: No pose file for {clip_file}")

    if len(all_files) < 10:
        print("Error: Not enough samples.")
        return

    # Split
    train_files, val_files = train_test_split(
        all_files, test_size=0.2, random_state=args.seed,
        stratify=[f[2] for f in all_files]
    )
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Balance
    if args.balance != 'none':
        train_files = balance_dataset(train_files, method=args.balance)

    # Weights
    train_labels = [labels_map[f[2]] for f in train_files]
    unique_labels = np.unique(train_labels)
    class_weights_computed = compute_class_weight(
        'balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for i, label_idx in enumerate(unique_labels):
        class_weights[label_idx] = class_weights_computed[i]

    # Apply biases
    other_idx = labels_map.get('other', -1)
    if other_idx >= 0 and args.other_bias > 0:
        class_weights[other_idx] *= args.other_bias
        print(f"Applied 'other' bias. Weight: {class_weights[other_idx]:.3f}")

    if args.groundstroke_boost > 1.0:
        for shot in ['groundstrokes', 'forehand', 'backhand']:
            idx = labels_map.get(shot, -1)
            if idx >= 0:
                class_weights[idx] *= args.groundstroke_boost

    class_weights = class_weights.to(device)

    # Datasets
    train_dataset = PadelShotDataset(train_files, labels_map, clip_len=args.clip_len, resolution=(
        args.resolution, args.resolution), augment=True, augment_level=args.augment_level)
    val_dataset = PadelShotDataset(val_files, labels_map, clip_len=args.clip_len, resolution=(
        args.resolution, args.resolution), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = EnhancedVideoClassifier(
        num_classes=num_classes, pose_hidden_size=args.pose_hidden_size, freeze_backbone=args.freeze_backbone).to(device)

    # Criteria
    if args.use_focal_loss:
        print("Using Focal Loss")
        criterion_shot = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        criterion_shot = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing)

    criterion_side = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Training Loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        t_metrics = train_epoch(
            model, train_loader, criterion_shot, criterion_side, optimizer, device, scaler)
        v_loss, v_acc, v_side_acc, v_per_class, v_labels, v_preds = validate_epoch(
            model, val_loader, criterion_shot, criterion_side, device)

        print(
            f"Train - Loss: {t_metrics['loss']:.4f}, Shot Acc: {t_metrics['shot_acc']:.2f}%, Side Acc: {t_metrics['side_acc']:.2f}%")
        print(
            f"Val   - Loss: {v_loss:.4f}, Shot Acc: {v_acc:.2f}%, Side Acc: {v_side_acc:.2f}%")

        print("\nPer-Class Val Accuracy:")
        for idx in sorted(v_per_class.keys()):
            print(f"  {idx_to_label[idx]:15s}: {v_per_class[idx]:5.1f}%")

        # Confusion Matrix
        if (epoch + 1) % 5 == 0 or v_acc > best_val_acc:
            print("\nConfusion Matrix:")
            cm = confusion_matrix(v_labels, v_preds)
            print(f"{'':15s}", end="")
            for i in range(num_classes):
                print(f"{idx_to_label[i][:4]:>6s}", end="")
            print()
            for i in range(num_classes):
                print(f"{idx_to_label[i]:15s}", end="")
                for j in range(num_classes):
                    print(f"{cm[i, j]:6d}", end="")
                print()

        scheduler.step(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_val_acc,
                'labels_map': labels_map,
                'idx_to_label': idx_to_label,
                'other_class_idx': other_idx,
                'config': {
                    'clip_len': args.clip_len,
                    'pose_hidden_size': args.pose_hidden_size,
                    'confidence_threshold': args.confidence_threshold,
                    'other_bias': args.other_bias
                }
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"\n✓ New best model saved! (Val Acc: {v_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping after {epoch + 1} epochs")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Padel Shot Classifier")
    parser.add_argument('--data', type=str,
                        default='dataset/player_enhanced', help="Dataset path")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs")
    parser.add_argument('--batch-size', type=int,
                        default=32, help="Batch size")
    parser.add_argument('--workers', type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument('--clip-len', type=int,
                        default=32, help="Frames per clip")
    parser.add_argument('--resolution', type=int,
                        default=128, help="Input resolution")
    parser.add_argument('--pose-hidden-size', type=int,
                        default=256, help="LSTM hidden size")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--balance', type=str, default='oversample',
                        choices=['oversample', 'undersample', 'none'])
    parser.add_argument('--augment-level', type=str,
                        default='strong', choices=['light', 'medium', 'strong'])
    parser.add_argument('--weight-decay', type=float,
                        default=0.02, help="Weight decay")
    parser.add_argument('--label-smoothing', type=float,
                        default=0.15, help="Label smoothing")
    parser.add_argument('--other-bias', type=float,
                        default=0.5, help="Bias towards 'other' class")
    parser.add_argument('--groundstroke-boost', type=float,
                        default=1.0, help="Boost weight for groundstrokes")
    parser.add_argument('--confidence-threshold', type=float,
                        default=0.6, help="Confidence threshold")
    parser.add_argument('--freeze-backbone',
                        action='store_true', help="Freeze video backbone")
    parser.add_argument('--use-focal-loss',
                        action='store_true', help="Use Focal Loss")
    parser.add_argument('--focal-gamma', type=float,
                        default=2.0, help="Gamma for Focal Loss")
    parser.add_argument('--save-dir', type=str, default='models',
                        help="Directory to save checkpoints")

    args = parser.parse_args()
    main(args)
