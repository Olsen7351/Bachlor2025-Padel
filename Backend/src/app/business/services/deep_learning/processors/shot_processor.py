"""
Shot Classification Processor.

Processes video for shot classification by combining:
- YOLO pose detection for player keypoints
- TrackNet ball detection  
- R(2+1)D + LSTM model for shot type classification
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from collections import deque
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from ultralytics import YOLO

from ..architectures import load_shot_model, preprocess_clip
from ..trackers import StreamingBallTracker, SmartBallTracker, SimpleTracker
from ..utils import ensure_dir, get_motion_score


# Shot type colors for visualization
SHOT_COLORS = {
    'forehand': (0, 255, 0),
    'backhand': (255, 0, 0),
    'serve': (0, 165, 255),
    'overhead': (0, 255, 255),
    'lob': (255, 0, 255),
    'other': (128, 128, 128)
}


class ShotClassificationProcessor:
    """
    Processes video for shot classification and returns shot events.
    
    Integrates pose detection, ball tracking, and shot classification
    into a unified processing pipeline.
    """
    
    def __init__(
        self, 
        shot_model_path: str, 
        yolo_pose_path: str, 
        tracknet_path: str,
        device: Optional[torch.device] = None, 
        confidence_threshold: Optional[float] = None,
        window_size: int = 32, 
        stride: int = 8, 
        classifier_resolution: int = 128
    ):
        """
        Initialize the processor.
        
        Args:
            shot_model_path: Path to shot classification model
            yolo_pose_path: Path to YOLO pose model
            tracknet_path: Path to TrackNet model for ball detection
            device: Compute device (auto-detected if None)
            confidence_threshold: Classification confidence threshold
            window_size: Number of frames in sliding window
            stride: Window stride for classification
            classifier_resolution: Resolution for classifier input
        """
        self.device = device or self._get_device()
        self.window_size = window_size
        self.stride = stride
        self.classifier_resolution = classifier_resolution
        
        # Load shot classifier
        print("\nLoading shot classifier...")
        self.model, self.idx_to_label, self.other_class_idx, model_conf = load_shot_model(
            shot_model_path, self.device
        )
        self.confidence_threshold = confidence_threshold or model_conf
        
        # Load pose model
        print("\nLoading YOLO pose model...")
        self.pose_model = YOLO(yolo_pose_path)
        
        # Ball tracker (optional)
        self.ball_tracker: Optional[StreamingBallTracker] = None
        self.smart_ball_tracker: Optional[SmartBallTracker] = None
        if tracknet_path and os.path.exists(tracknet_path):
            print(f"Loading TrackNet for shot classification from {tracknet_path}...")
            self.ball_tracker = StreamingBallTracker(tracknet_path, self.device, input_wh=(512, 288))
            self.smart_ball_tracker = SmartBallTracker()
        
        # Player tracking
        self.tracker = SimpleTracker()
        
        # Results storage
        self.shot_events: List[Dict] = []
        self.player_stats: Dict[int, Dict[str, int]] = {}
        self.player_positions: Dict[int, float] = {}
    
    @staticmethod
    def _get_device() -> torch.device:
        """Get best available compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def process_video(
        self, 
        video_frames: List[np.ndarray], 
        fps: float = 30.0,
        yolo_resolution: Tuple[int, int] = (640, 360)
    ) -> Dict:
        """
        Process video frames and return shot events.
        
        Args:
            video_frames: List of BGR frames
            fps: Video frame rate
            yolo_resolution: Resolution for YOLO inference
            
        Returns:
            Dict with 'shot_events', 'ball_positions', 'player_stats', 'frame_predictions', 'frame_players'
        """
        print(f"\nProcessing {len(video_frames)} frames for shot classification...")
        
        orig_height, orig_width = video_frames[0].shape[:2]
        yolo_width, yolo_height = yolo_resolution
        scale_orig_to_yolo_x = yolo_width / orig_width
        scale_orig_to_yolo_y = yolo_height / orig_height
        
        # Compute background for ball tracker
        if self.ball_tracker:
            self.ball_tracker.compute_background(video_frames)
        
        # Buffers
        frame_buffer = deque(maxlen=self.window_size)
        pose_buffer = deque(maxlen=self.window_size)
        
        # Frame-level storage
        frame_ball_map: Dict[int, Optional[Tuple[int, int]]] = {}
        frame_players_map: Dict[int, List[Dict]] = {}
        frame_active_map: Dict[int, Optional[int]] = {}
        predictions: Dict[int, Tuple[str, int, float]] = {}
        
        prev_players = None
        last_shot_global = -9999
        
        for frame_idx, frame in enumerate(tqdm(video_frames, desc="Shot classification")):
            # Resize and convert
            frame_yolo = cv2.resize(frame, (yolo_width, yolo_height))
            frame_rgb = cv2.cvtColor(frame_yolo, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = self.pose_model(frame_yolo, verbose=False)
            
            players = self._extract_players(
                results, 
                scale_orig_to_yolo_x, 
                scale_orig_to_yolo_y,
                yolo_width, 
                yolo_height
            )
            
            self.tracker.update(players)
            self._update_player_positions(players)
            
            # Ball tracking
            ball_pos = self._track_ball(frame, orig_width, orig_height)
            frame_ball_map[frame_idx] = ball_pos
            frame_players_map[frame_idx] = players
            
            # Determine active player
            active_idx, active_id = self._get_active_player(players, prev_players)
            frame_active_map[frame_idx] = active_id
            
            if active_idx is not None:
                active_kp = players[active_idx]['keypoints'].flatten()
            else:
                active_kp = np.zeros(51)
            
            frame_buffer.append(torch.from_numpy(frame_rgb))
            pose_buffer.append(torch.from_numpy(active_kp))
            
            # Classify when buffer is full
            if len(frame_buffer) == self.window_size and frame_idx % self.stride == 0:
                shot_info = self._classify_window(
                    frame_buffer, pose_buffer, frame_idx, fps,
                    last_shot_global, frame_ball_map, frame_players_map, frame_active_map
                )
                
                if shot_info:
                    shot_frame, pred_label, shooter_id, conf = shot_info
                    last_shot_global = shot_frame
                    
                    # Update stats
                    if shooter_id not in self.player_stats:
                        self.player_stats[shooter_id] = {}
                    self.player_stats[shooter_id][pred_label] = \
                        self.player_stats[shooter_id].get(pred_label, 0) + 1
                    
                    # Record event
                    self.shot_events.append({
                        'frame': shot_frame,
                        'shot_type': pred_label,
                        'player_id': shooter_id,
                        'confidence': conf
                    })
                    
                    # Store prediction for display
                    display_duration = self.stride * 3
                    for offset in range(display_duration):
                        target_idx = shot_frame + offset
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
    
    def _extract_players(
        self, 
        results, 
        scale_x: float, 
        scale_y: float,
        yolo_w: int, 
        yolo_h: int
    ) -> List[Dict]:
        """Extract player data from YOLO pose results."""
        players = []
        
        if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return players
        
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for i in range(len(keypoints_data)):
            kp = keypoints_data[i]
            if np.mean(kp[:, 2]) > 0.3:  # Confidence threshold
                # Scale bbox to original resolution
                scaled_bbox = boxes[i].copy()
                scaled_bbox[0] /= scale_x
                scaled_bbox[1] /= scale_y
                scaled_bbox[2] /= scale_x
                scaled_bbox[3] /= scale_y
                
                # Normalize keypoints
                kp_proc = kp.copy()
                if np.max(kp_proc) > 1.0:
                    kp_proc[:, 0] /= yolo_w
                    kp_proc[:, 1] /= yolo_h
                
                players.append({
                    'keypoints': kp_proc,
                    'bbox': scaled_bbox,
                    'bbox_proc': boxes[i].copy(),
                    'raw_kp': kp
                })
        
        return players
    
    def _update_player_positions(self, players: List[Dict]) -> None:
        """Update player position tracking (left/right side)."""
        for p in players:
            pid = p.get('id')
            if pid is not None:
                bbox = p['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                self.player_positions[pid] = cx
    
    def _track_ball(
        self, 
        frame: np.ndarray, 
        orig_w: int, 
        orig_h: int
    ) -> Optional[Tuple[int, int]]:
        """Track ball in frame using TrackNet."""
        if not self.ball_tracker:
            return None
        
        hm = self.ball_tracker.predict(frame)
        detected_ball = None
        
        if hm is not None:
            _, th = cv2.threshold(hm, 0.5, 1, 0)
            ctrs, _ = cv2.findContours((th * 255).astype(np.uint8), 0, 2)
            if ctrs:
                c = max(ctrs, key=cv2.contourArea)
                (cx, cy), _ = cv2.minEnclosingCircle(c)
                detected_ball = (int(cx * (orig_w / 512)), int(cy * (orig_h / 288)))
        
        if self.smart_ball_tracker:
            return self.smart_ball_tracker.update(detected_ball)
        return detected_ball
    
    def _get_active_player(
        self, 
        players: List[Dict], 
        prev_players: Optional[List[Dict]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Determine the most active player based on motion."""
        if not players:
            return None, None
        
        if len(players) == 1:
            return 0, players[0].get('id')
        
        # Score by motion
        motion_scores = []
        for p in players:
            prev_kp = None
            if prev_players:
                for pp in prev_players:
                    if pp.get('id') == p.get('id'):
                        prev_kp = pp['keypoints']
                        break
            motion_scores.append(get_motion_score(p['keypoints'], prev_kp))
        
        active_idx = int(np.argmax(motion_scores))
        return active_idx, players[active_idx].get('id')
    
    def _classify_window(
        self,
        frame_buffer: deque,
        pose_buffer: deque,
        frame_idx: int,
        fps: float,
        last_shot_global: int,
        frame_ball_map: Dict,
        frame_players_map: Dict,
        frame_active_map: Dict
    ) -> Optional[Tuple[int, str, int, float]]:
        """Classify shot from current window."""
        clip = torch.stack(list(frame_buffer)).permute(0, 3, 1, 2)
        clip_tensor = preprocess_clip(
            clip, 
            clip_len=self.window_size,
            resolution=(self.classifier_resolution, self.classifier_resolution)
        ).to(self.device)
        
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
                
                # Minimum gap between shots
                if shot_frame_idx - last_shot_global >= fps:
                    shooter_id = self._find_shooter(
                        shot_frame_idx, frame_ball_map, frame_players_map, frame_active_map
                    )
                    
                    if shooter_id is not None:
                        return (shot_frame_idx, pred_label, shooter_id, conf)
        
        return None
    
    def _find_shooter(
        self,
        shot_frame_idx: int,
        frame_ball_map: Dict,
        frame_players_map: Dict,
        frame_active_map: Dict
    ) -> Optional[int]:
        """Find the player who hit the shot based on ball proximity."""
        shot_ball_pos = frame_ball_map.get(shot_frame_idx)
        shot_players = frame_players_map.get(shot_frame_idx)
        
        if shot_ball_pos and shot_players:
            min_dist = float('inf')
            shooter_id = None
            
            for p in shot_players:
                bbox = p['bbox']
                p_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                dist = np.linalg.norm(np.array(shot_ball_pos) - np.array(p_center))
                if dist < min_dist:
                    min_dist = dist
                    shooter_id = p.get('id')
            
            if shooter_id is not None:
                return shooter_id
        
        # Fallback to active player
        return frame_active_map.get(shot_frame_idx)
    
    def draw_shot_overlay(
        self, 
        frames: List[np.ndarray], 
        shot_data: Dict
    ) -> List[np.ndarray]:
        """Draw shot classification overlay on frames."""
        output_frames = []
        predictions = shot_data.get('frame_predictions', {})
        ball_positions = shot_data.get('ball_positions', {})
        frame_players = shot_data.get('frame_players', {})
        
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            
            # Draw ball
            ball_pos = ball_positions.get(frame_idx)
            if ball_pos:
                cv2.circle(frame, ball_pos, 5, (0, 0, 255), -1)
            
            # Draw shot prediction
            pred_data = predictions.get(frame_idx)
            if pred_data:
                shot_type, shooter_id, conf = pred_data
                color = SHOT_COLORS.get(shot_type, (255, 255, 255))
                
                # Find and highlight shooter
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
        
        # Draw stats overlay
        self._draw_stats_on_frames(output_frames)
        
        return output_frames
    
    def _draw_stats_on_frames(self, frames: List[np.ndarray]) -> None:
        """Draw player stats overlay on frames."""
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        center_x = w / 2
        
        for frame in frames:
            left_stats, right_stats = {}, {}
            
            # Split stats by player position
            for pid, p_stats in self.player_stats.items():
                pos = self.player_positions.get(pid, center_x)
                target = left_stats if pos < center_x else right_stats
                for label, count in p_stats.items():
                    target[label] = target.get(label, 0) + count
            
            self._draw_stats_box(frame, left_stats, "Player Left", 20, 40)
            self._draw_stats_box(frame, right_stats, "Player Right", w - 220, 40)
    
    def _draw_stats_box(
        self, 
        frame: np.ndarray, 
        stats: Dict[str, int], 
        title: str, 
        x: int, 
        y: int
    ) -> None:
        """Draw a stats box on the frame."""
        box_h = 30 + (len(stats) + 1) * 20 if stats else 60
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-30), (x+200, y + box_h - 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_off = 25
        total = sum(stats.values())
        cv2.putText(frame, f"Total: {total}", (x, y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_off += 20
        
        for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            c = SHOT_COLORS.get(label, (255, 255, 255))
            cv2.putText(frame, f"{label}: {count}", (x, y + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
            y_off += 20
    
    def export_shots_csv(self, output_path: str) -> None:
        """Export shot events to CSV."""
        ensure_dir(output_path)
        df = pd.DataFrame(self.shot_events)
        df.to_csv(output_path, index=False)
        print(f"Exported shots to: {output_path}")
    
    def reset(self) -> None:
        """Reset processor state for new video."""
        self.tracker.reset()
        self.shot_events.clear()
        self.player_stats.clear()
        self.player_positions.clear()