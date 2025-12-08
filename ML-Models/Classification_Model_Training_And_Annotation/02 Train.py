#!/usr/bin/env python3
"""
Padel Shot Classification Training

Trains an enhanced video + pose classifier for padel shot classification.
Uses R3D-18 backbone for video features and LSTM for pose sequences.

Features:
- Class weight balancing for imbalanced datasets
- Per-class accuracy monitoring
- Early stopping
- Learning rate scheduling

Usage:
    python train.py --data dataset/player_enhanced --epochs 50
    python train.py --data dataset/player_enhanced --epochs 100 --batch-size 4

Classes: backhand, forehand, lob, other, overhead, serve
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
from sklearn.metrics import confusion_matrix, classification_report
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

    Architecture:
    - Video branch: R(2+1)D-18 pretrained on Kinetics-400 (Better motion modeling than R3D)
    - Pose branch: 3-layer Bidirectional LSTM with attention
    - Fusion: Concatenation + Wider MLP classifier
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

        # Attention mechanism (handles bidirectional output = hidden_size * 2)
        self.pose_attention = nn.Sequential(
            nn.Linear(pose_hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Fusion classifier (Wider and deeper)
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


# ==========================================
# DATASET
# ==========================================

class PadelShotDataset(Dataset):
    """
    Dataset for loading padel shot video clips with YOLO pose keypoints.

    Expected structure:
        dataset/player_enhanced/
            backhand/
                clips/shot_0001.mp4
                pose_yolo/shot_0001.npy
            forehand/
                ...
    """

    def __init__(self, file_list, labels_map, clip_len=32, resolution=(112, 112), augment=False, augment_level='strong'):
        self.file_list = file_list
        self.labels_map = labels_map
        self.clip_len = clip_len
        self.resolution = resolution
        self.augment = augment
        self.augment_level = augment_level  # 'light', 'medium', 'strong'

        # ImageNet normalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        # COCO keypoint indices for horizontal flip (swap left/right)
        # 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
        # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
        # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
        # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
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
        except Exception as e:
            print(f"Error loading video {clip_path}: {e}")
            video_frames = torch.zeros((3, self.clip_len, *self.resolution))

        # Load pose
        try:
            pose_data = np.load(pose_path, allow_pickle=True)
            if pose_data.ndim == 3 and pose_data.shape[1:] == (17, 3):
                pose_data = np.nan_to_num(pose_data)
                pose_data = self._sample_sequence(pose_data, self.clip_len)

                # Normalize pose to 0-1 range (assuming 1280x720 source)
                # Only normalize if values are > 1 (pixels)
                if np.max(pose_data) > 1.0:
                    pose_data[:, :, 0] /= 1280.0  # x
                    pose_data[:, :, 1] /= 720.0   # y

                pose_data = pose_data.reshape(self.clip_len, -1)  # (T, 51)
            else:
                pose_data = np.zeros((self.clip_len, 51))
        except Exception as e:
            print(f"Error loading pose {pose_path}: {e}")
            pose_data = np.zeros((self.clip_len, 51))

        # Apply augmentation during training
        if self.augment:
            video_frames, pose_data = self._augment(video_frames, pose_data)

        return video_frames, torch.from_numpy(pose_data).float() if isinstance(pose_data, np.ndarray) else pose_data, label

    def _augment(self, video, pose):
        """Apply comprehensive data augmentation to video and pose data."""
        # Ensure pose is numpy array for manipulation
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()

        # ===== TEMPORAL AUGMENTATIONS =====

        # Random temporal shift (start from different frame)
        if random.random() > 0.4:
            shift = random.randint(-5, 5)
            if shift != 0:
                video = torch.roll(video, shift, dims=1)
                pose = np.roll(pose, shift, axis=0)

        # Random temporal speed (slightly faster/slower)
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

        # Random temporal reversal (play backwards) - rare
        if random.random() > 0.85:
            video = torch.flip(video, dims=[1])
            pose = np.flip(pose, axis=0).copy()

        # ===== SPATIAL AUGMENTATIONS =====

        # Horizontal flip with pose keypoint swap
        if random.random() > 0.5:
            video = torch.flip(video, dims=[3])  # Flip width dimension
            # Reshape pose to (T, 17, 3) for keypoint manipulation
            pose_reshaped = pose.reshape(-1, 17, 3)
            # Flip x-coordinates (assuming normalized 0-1)
            pose_reshaped[:, :, 0] = 1.0 - pose_reshaped[:, :, 0]
            # Swap left/right keypoints
            for left, right in self.flip_pairs:
                pose_reshaped[:, [left, right]
                              ] = pose_reshaped[:, [right, left]]
            pose = pose_reshaped.reshape(-1, 51)

        # Random crop/zoom (crop and resize back)
        if random.random() > 0.6:
            scale = random.uniform(0.85, 1.0)
            _, t, h, w = video.shape
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            video = video[:, :, top:top+new_h, left:left+new_w]
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3),  # (T, C, H, W)
                size=(h, w), mode='bilinear', align_corners=False
            ).permute(1, 0, 2, 3)  # Back to (C, T, H, W)

        # ===== COLOR AUGMENTATIONS =====

        # Random brightness
        if random.random() > 0.4:
            brightness = random.uniform(0.7, 1.3)
            video = video * brightness

        # Random contrast
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            mean = video.mean()
            video = (video - mean) * contrast + mean

        # Random saturation (approximate by scaling color channels)
        if random.random() > 0.6:
            saturation = random.uniform(0.8, 1.2)
            gray = video.mean(dim=0, keepdim=True)
            video = gray + saturation * (video - gray)

        # Random hue shift (approximate)
        if random.random() > 0.7:
            hue_shift = random.uniform(-0.1, 0.1)
            video = torch.roll(video, shifts=int(
                hue_shift * video.shape[0]), dims=0)

        # ===== BLUR & NOISE =====

        # Gaussian Blur
        if random.random() > 0.7:
            sigma = random.uniform(0.1, 2.0)
            video = transforms.functional.gaussian_blur(
                video, kernel_size=5, sigma=sigma)

        # Gaussian Noise
        if random.random() > 0.7:
            noise = torch.randn_like(video) * 0.05
            video = video + noise

        # Clamp video values
        video = torch.clamp(video, -2.5, 2.5)  # Allow normalized range

        # ===== CUTOUT / RANDOM ERASING =====
        if self.augment_level == 'strong' and random.random() > 0.7:
            _, t, h, w = video.shape
            # Random rectangular mask
            mask_h = random.randint(h // 8, h // 4)
            mask_w = random.randint(w // 8, w // 4)
            top = random.randint(0, h - mask_h)
            left = random.randint(0, w - mask_w)
            # Apply to random subset of frames
            frame_start = random.randint(0, max(0, t - t//4))
            frame_end = min(t, frame_start + random.randint(t//8, t//4))
            video[:, frame_start:frame_end,
                  top:top+mask_h, left:left+mask_w] = 0

        # ===== POSE AUGMENTATIONS =====

        # Add noise to pose keypoints (stronger than before)
        if random.random() > 0.3:
            noise_scale = 0.03 if self.augment_level == 'light' else 0.05
            noise = np.random.normal(0, noise_scale, pose.shape)
            pose = pose + noise

        # Random dropout of keypoints (simulate occlusion)
        if self.augment_level in ['medium', 'strong'] and random.random() > 0.7:
            pose_reshaped = pose.reshape(-1, 17, 3)
            # Zero out 1-3 random keypoints for some frames
            n_drop = random.randint(1, 3)
            drop_kpts = random.sample(range(17), n_drop)
            drop_frames = random.sample(
                range(len(pose_reshaped)), min(len(pose_reshaped)//3, 8))
            for f in drop_frames:
                for k in drop_kpts:
                    pose_reshaped[f, k, :] = 0  # Zero out (x, y, conf)
            pose = pose_reshaped.reshape(-1, 51)

        # Random scaling of pose (simulate distance variation)
        if random.random() > 0.6:
            scale = random.uniform(0.9, 1.1)
            pose_reshaped = pose.reshape(-1, 17, 3)
            # Scale x, y but not confidence
            pose_reshaped[:, :, :2] *= scale
            pose = pose_reshaped.reshape(-1, 51)

        return video, pose

    def _load_video(self, video_path):
        """Load and preprocess video frames using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize immediately to save memory and compute
            # cv2.resize expects (width, height)
            frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames in {video_path}")

        # Sample frames
        indices = self._sample_indices(len(frames), self.clip_len)
        frames = [frames[i] for i in indices]

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).float() / 255.0

        # Normalize (Resize is already done)
        frames_tensor = (frames_tensor - self.norm_mean) / self.norm_std

        return frames_tensor

    def _sample_indices(self, total, target):
        """Sample or pad frame indices to target length."""
        if total >= target:
            return np.linspace(0, total - 1, target, dtype=int)
        else:
            indices = np.arange(total)
            return np.pad(indices, (0, target - total), 'edge')

    def _sample_sequence(self, seq, target_len):
        """Sample or pad sequence to target length."""
        if len(seq) >= target_len:
            indices = np.linspace(0, len(seq) - 1, target_len, dtype=int)
            return seq[indices]
        else:
            return np.pad(seq, ((0, target_len - len(seq)), (0, 0), (0, 0)), 'edge')


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for videos, poses, labels in tqdm(loader, desc="Training"):
        videos = videos.to(device, non_blocking=True)
        poses = poses.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Training
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(videos, poses)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos, poses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            l = label.item()
            per_class_total[l] += 1
            if pred.item() == l:
                per_class_correct[l] += 1

    accuracy = 100 * correct / total
    per_class_acc = {
        k: 100 * per_class_correct[k] / per_class_total[k] for k in per_class_total}
    return total_loss / len(loader), accuracy, per_class_acc


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, poses, labels in tqdm(loader, desc="Validating"):
            videos = videos.to(device, non_blocking=True)
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(videos, poses)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for label, pred in zip(labels, predicted):
                l = label.item()
                per_class_total[l] += 1
                if pred.item() == l:
                    per_class_correct[l] += 1

    accuracy = 100 * correct / total
    per_class_acc = {
        k: 100 * per_class_correct[k] / per_class_total[k] for k in per_class_total}
    return total_loss / len(loader), accuracy, per_class_acc, all_labels, all_preds


# ==========================================
# MAIN
# ==========================================

def balance_dataset(files, method='oversample'):
    """
    Balance dataset by oversampling minority classes or undersampling majority classes.

    Args:
        files: List of (clip_path, pose_path, class_name) tuples
        method: 'oversample' to duplicate minority samples, 'undersample' to limit majority

    Returns:
        Balanced list of files
    """
    # Group files by class
    class_files = defaultdict(list)
    for f in files:
        class_files[f[2]].append(f)

    class_counts = {k: len(v) for k, v in class_files.items()}
    print(f"\nOriginal class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:15s}: {count:4d}")

    if method == 'oversample':
        # Oversample to match the largest class
        max_count = max(class_counts.values())
        balanced_files = []
        for class_name, class_samples in class_files.items():
            if len(class_samples) < max_count:
                # Duplicate samples to reach max_count
                multiplier = max_count // len(class_samples)
                remainder = max_count % len(class_samples)
                balanced_files.extend(class_samples * multiplier)
                balanced_files.extend(random.sample(class_samples, remainder))
            else:
                balanced_files.extend(class_samples)
        print(
            f"\nAfter oversampling: {len(balanced_files)} samples (target: {max_count} per class)")

    elif method == 'undersample':
        # Undersample to match the smallest class
        min_count = min(class_counts.values())
        balanced_files = []
        for class_name, class_samples in class_files.items():
            balanced_files.extend(random.sample(class_samples, min_count))
        print(
            f"\nAfter undersampling: {len(balanced_files)} samples (target: {min_count} per class)")

    else:  # 'none' or invalid
        balanced_files = files
        print(f"\nNo balancing applied: {len(balanced_files)} samples")

    # Verify balance
    new_counts = Counter(f[2] for f in balanced_files)
    print(f"\nBalanced class distribution:")
    for cls, count in sorted(new_counts.items()):
        print(f"  {cls:15s}: {count:4d}")

    random.shuffle(balanced_files)
    return balanced_files


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
    print(f"Classes ({num_classes}): {', '.join(classes)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Balance method: {args.balance}")
    print(f"Augmentation level: {args.augment_level}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Other class bias: {args.other_bias}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Collect files
    all_files = []
    class_counts = defaultdict(int)

    for class_name in classes:
        clip_dir = os.path.join(args.data, class_name, 'clips')
        pose_dir = os.path.join(args.data, class_name, 'pose_yolo')

        if not os.path.exists(clip_dir) or not os.path.exists(pose_dir):
            print(f"Warning: Missing directories for {class_name}")
            continue

        for clip_file in os.listdir(clip_dir):
            if not clip_file.endswith('.mp4'):
                continue

            shot_name = os.path.splitext(clip_file)[0]
            pose_file = f"{shot_name}.npy"
            clip_path = os.path.join(clip_dir, clip_file)
            pose_path = os.path.join(pose_dir, pose_file)

            if os.path.exists(pose_path):
                all_files.append((clip_path, pose_path, class_name))
                class_counts[class_name] += 1

    print(f"\nDataset Distribution:")
    for class_name in classes:
        count = class_counts[class_name]
        pct = 100 * count / len(all_files) if all_files else 0
        print(f"  {class_name:15s}: {count:4d} clips ({pct:5.1f}%)")
    print(f"\nTotal samples: {len(all_files)}")

    if len(all_files) < 10:
        print("Error: Not enough samples to train!")
        return

    # Split data (before balancing to keep validation set realistic)
    train_files, val_files = train_test_split(
        all_files, test_size=0.2, random_state=args.seed,
        stratify=[f[2] for f in all_files]
    )
    print(
        f"\nOriginal split - Train: {len(train_files)}, Val: {len(val_files)}")

    # Balance training set only (keep validation set as-is for realistic evaluation)
    if args.balance != 'none':
        train_files = balance_dataset(train_files, method=args.balance)
    print(f"Final Train: {len(train_files)}, Val: {len(val_files)}")

    # Compute class weights (only for classes present in training)
    train_labels = [labels_map[f[2]] for f in train_files]
    unique_labels = np.unique(train_labels)

    # Create class weights for ALL classes (even if not in training)
    class_weights_computed = compute_class_weight(
        'balanced', classes=unique_labels, y=train_labels)

    # Map to full class weight tensor
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for i, label_idx in enumerate(unique_labels):
        class_weights[label_idx] = class_weights_computed[i]
    class_weights = class_weights.to(device)

    print(f"\nClass Weights:")
    for i, class_name in enumerate(classes):
        if i in unique_labels:
            print(f"  {class_name:15s}: {class_weights[i]:.3f}")
        else:
            print(f"  {class_name:15s}: N/A (no samples)")

    # Create dataloaders with strong augmentation to combat overfitting
    train_dataset = PadelShotDataset(
        train_files, labels_map, clip_len=args.clip_len,
        resolution=(args.resolution, args.resolution),
        augment=True, augment_level=args.augment_level)  # Enable augmentation
    val_dataset = PadelShotDataset(
        # No augment for validation
        val_files, labels_map, clip_len=args.clip_len,
        resolution=(args.resolution, args.resolution),
        augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None)    # Model
    model = EnhancedVideoClassifier(
        num_classes=num_classes,
        pose_hidden_size=args.pose_hidden_size,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Loss, optimizer, scheduler
    # Apply "other" class bias - reduce its weight so model learns to be conservative
    # This makes "other" the default assumption when uncertain
    other_class_idx = labels_map.get('other', -1)
    if other_class_idx >= 0 and args.other_bias > 0:
        # Reduce the loss penalty for predicting "other" - this biases model towards it
        class_weights[other_class_idx] *= args.other_bias
        print(
            f"\nApplied 'other' class bias: weight reduced by {args.other_bias}x")
        print(f"  other class weight: {class_weights[other_class_idx]:.3f}")

    # Label smoothing helps prevent overconfidence and improves generalization
    if args.use_focal_loss:
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing)

    # AdamW with stronger regularization to combat overfitting
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    best_val_acc = 0
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc, train_per_class = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_per_class, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        print("\nPer-Class Val Accuracy:")
        for idx in sorted(val_per_class.keys()):
            print(f"  {idx_to_label[idx]:15s}: {val_per_class[idx]:5.1f}%")

        # Print Confusion Matrix
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print("\nConfusion Matrix:")
            cm = confusion_matrix(val_labels, val_preds)
            print(f"{'':15s}", end="")
            for i in range(num_classes):
                print(f"{idx_to_label[i][:4]:>6s}", end="")
            print()
            for i in range(num_classes):
                print(f"{idx_to_label[i]:15s}", end="")
                for j in range(num_classes):
                    print(f"{cm[i, j]:6d}", end="")
                print()

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'labels_map': labels_map,
                'idx_to_label': idx_to_label,
                'other_class_idx': other_class_idx,
                'config': {
                    'clip_len': args.clip_len,
                    'pose_hidden_size': args.pose_hidden_size,
                    'confidence_threshold': args.confidence_threshold,
                    'other_bias': args.other_bias
                }
            }, os.path.join(args.save_dir, 'best_model.pth'))

            print(f"\n✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping after {epoch + 1} epochs")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print("\nNext step: python inference.py --input videos/test.mp4")


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
                        default=128, help="Input resolution (default: 128)")
    parser.add_argument('--pose-hidden-size', type=int,
                        default=256, help="LSTM hidden size")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--balance', type=str, default='oversample',
                        choices=['oversample', 'undersample', 'none'],
                        help="Dataset balancing: oversample (default), undersample, or none")
    parser.add_argument('--augment-level', type=str, default='strong',
                        choices=['light', 'medium', 'strong'],
                        help="Augmentation intensity: light, medium, or strong (default)")
    parser.add_argument('--weight-decay', type=float, default=0.02,
                        help="Weight decay for AdamW optimizer (default: 0.02)")
    parser.add_argument('--label-smoothing', type=float, default=0.15,
                        help="Label smoothing factor (default: 0.15)")
    parser.add_argument('--other-bias', type=float, default=0.5,
                        help="Bias towards 'other' class (0.5 = halve its loss weight, 1.0 = no bias)")
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                        help="Confidence threshold below which prediction defaults to 'other' (default: 0.6)")
    parser.add_argument('--freeze-backbone', action='store_true',
                        help="Freeze video backbone layers to prevent overfitting")
    parser.add_argument('--use-focal-loss', action='store_true',
                        help="Use Focal Loss instead of CrossEntropy")
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help="Gamma for Focal Loss")
    parser.add_argument('--save-dir', type=str, default='models',
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    main(args)
