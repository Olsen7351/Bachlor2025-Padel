import torch
import torch.nn as nn
import cv2
import numpy as np
import pickle
from collections import OrderedDict, deque
from typing import List, Dict, Tuple, Optional


# ============================================================================
# TRACKNET MODEL ARCHITECTURE
# ============================================================================

class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    

class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


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


# ============================================================================
# FILTER CLASSES
# ============================================================================

class PolygonExclusionFilter:
    """
    Rejects ball detections in polygon-shaped exclusion zones.
    Supports tilted zones for angled glass panels in perspective view.
    """
    
    def __init__(self):
        self.exclusion_zones: List[np.ndarray] = []
        self.zone_names: List[str] = []
    
    def add_polygon_zone(self, points: List[Tuple[float, float]], name: str = "zone"):
        """Add a polygon exclusion zone."""
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
        """Check if point is in any exclusion zone."""
        for zone, name in zip(self.exclusion_zones, self.zone_names):
            if cv2.pointPolygonTest(zone, (x, y), False) >= 0:
                return True, name
        return False, ""
    
    def filter_detection(self, bbox: List[float]) -> Tuple[bool, str]:
        """Returns (should_accept, rejection_reason)"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        is_excluded, zone_name = self.is_excluded(cx, cy)
        if is_excluded:
            return False, f"in_{zone_name}"
        return True, ""
    
    def draw_zones(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 180),
                   alpha: float = 0.35) -> np.ndarray:
        """Draw exclusion zones with transparency."""
        overlay = frame.copy()
        for zone in self.exclusion_zones:
            pts = zone.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


class SpatialPlayerFilter:
    """
    Filters ball detections based on player positions WITHOUT relying on IDs.
    Ball should be within reasonable distance of players on/near the court.
    """
    
    def __init__(self, max_distance_any_player: float = 500,):
        self.max_distance_any_player = max_distance_any_player
    
    def filter_detection(self, ball_bbox: List[float], player_detections: Dict) -> Tuple[bool, str]:
        """Returns (should_accept, reason)"""
        if not player_detections:
            return True, "no_players"
        
        ball_cx = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_cy = (ball_bbox[1] + ball_bbox[3]) / 2
        
        min_dist_any_player = float('inf')
        
        for player_id, player_bbox in player_detections.items():
            player_cx = (player_bbox[0] + player_bbox[2]) / 2
            player_cy = (player_bbox[1] + player_bbox[3]) / 2
            dist = np.sqrt((ball_cx - player_cx)**2 + (ball_cy - player_cy)**2)
            min_dist_any_player = min(min_dist_any_player, dist)
        
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
        """Returns (is_valid, reason)"""
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


# ============================================================================
# STREAMING BALL TRACKER (For Shot Classification)
# ============================================================================

class StreamingBallTracker:
    """
    TrackNet-based ball tracker using streaming approach.
    Used for shot classification - processes frames one at a time.
    """
    
    def __init__(self, model_path: str, device: torch.device, input_wh: Tuple[int, int] = (512, 288)):
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
    
    def compute_background(self, frames: List[np.ndarray]):
        """Compute background from a list of frames."""
        if not frames:
            return
        indices = np.linspace(0, len(frames) - 1, min(15, len(frames)), dtype=int)
        sampled = [cv2.resize(frames[i], (self.w, self.h)) for i in indices]
        med = np.median(sampled, axis=0).astype(np.uint8)
        self.bg = torch.from_numpy(cv2.cvtColor(med, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        self.bg = self.bg.permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.buf.clear()
    
    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
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
    """Tracks ball with constraints on sudden movement."""
    
    def __init__(self, fps: float = 30.0):
        self.last_pos = None
        self.missing_frames = 0
        self.fps = fps
        self.max_jump_dist = 100
    
    def update(self, detected_pos: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if detected_pos is None:
            self.missing_frames += 1
            return None
        
        if self.last_pos is not None:
            if self.missing_frames > 3 * self.fps:
                self.last_pos = detected_pos
                self.missing_frames = 0
                return detected_pos
        
        self.last_pos = detected_pos
        self.missing_frames = 0
        return detected_pos


# ============================================================================
# BATCH BALL TRACKER (Main Pipeline)
# ============================================================================

class BallTrackerTrackNet:
    """
    Ball tracker with:
    - Tilted polygon exclusion zones
    - No player ID dependency
    - Confidence score display
    - Batch processing for efficiency
    """
    
    MODEL_WIDTH = 512
    MODEL_HEIGHT = 288
    SEQ_LEN = 8
    
    def __init__(self, tracknet_path: str = 'models/TrackNet_best.pt',
                 detection_threshold: float = 0.5,
                 min_ball_radius: int = 2,
                 max_ball_radius: int = 50,
                 min_heatmap_confidence: float = 0.5):
        self.tracknet_path = tracknet_path
        self.detection_threshold = detection_threshold
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius
        self.min_heatmap_confidence = min_heatmap_confidence
        
        # Filters
        self.exclusion_filter = PolygonExclusionFilter()
        self.player_filter = SpatialPlayerFilter()
        self.trajectory_filter = TrajectoryFilter()
        
        # Filter enable flags
        self.use_exclusion_filter = True
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
            'rejected_player': 0, 'rejected_trajectory': 0, 'accepted': 0,
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
        """Add a polygon exclusion zone."""
        self.exclusion_filter.add_polygon_zone(points, name)
        self.use_exclusion_filter = True
    
    def setup_glass_panel_exclusions(self, left_zone: List[Tuple[float, float]], 
                                      right_zone: List[Tuple[float, float]]):
        """Set up left and right glass panel exclusion zones."""
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
        """Returns (cx, cy, radius, confidence) or None."""
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
        print(f"  Rejected - player distance: {self.stats['rejected_player']}")
        print(f"  Rejected - trajectory: {self.stats['rejected_trajectory']}")
        print(f"  ACCEPTED: {self.stats['accepted']} ({100*self.stats['accepted']/total_frames:.1f}%)")
    
    def draw_bboxes(self, video_frames: List[np.ndarray], ball_detections: List[Dict],
                    color: Tuple[int, int, int] = (0, 255, 0), trail_length: int = 10,
                    show_confidence: bool = True, draw_exclusion_zones: bool = False) -> List[np.ndarray]:
        """Draw ball detections with confidence scores."""
        output_frames = []
        recent_positions = []
        
        for frame_idx, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            frame = frame.copy()
            
            if draw_exclusion_zones:
                frame = self.exclusion_filter.draw_zones(frame)
            
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