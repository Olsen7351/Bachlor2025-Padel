import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class RallyState(Enum):
    IDLE = "idle"          # Between rallies
    SERVE = "serve"        # Serve detected, rally starting
    ACTIVE = "active"      # Rally in progress
    ENDING = "ending"      # Potential rally end (confirming)


@dataclass
class Rally:
    """Stores information about a single rally."""
    rally_id: int
    start_frame: int
    end_frame: Optional[int] = None
    ball_positions: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def duration_frames(self) -> int:
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame
    
    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames / fps


class RallyTracker:
    """
    State machine for tracking rally states using ball movement.
    
    Features:
    - Dynamic gap tolerance based on ball position and trajectory
    - Lob detection (ball moving upward = expect longer gaps)
    - Close-up handling (ball near bottom of frame = less reliable)
    
    For 1080p @ 30fps videos with close-up camera angles.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        frame_height: int = 1080,
        # Movement thresholds
        min_velocity: float = 3.0,           # Min pixels/frame for "moving"
        serve_velocity: float = 6.0,         # Min velocity to detect serve (lowered for close-up)
        # Base gap tolerances (will be extended dynamically)
        base_gap_during_rally: int = 40,     # Base max frames without ball during rally
        base_rally_end_gap: int = 70,        # Base frames without ball = rally ended
        # Dynamic gap extension
        max_gap_extension: int = 60,         # Maximum additional frames to add
        lob_velocity_threshold: float = -3.0, # Negative = upward movement (Y decreases)
        far_court_threshold: float = 0.4,    # Top 40% of frame = far court
        close_court_threshold: float = 0.7,  # Bottom 30% of frame = close/unreliable
        # Rally validation
        min_rally_frames: int = 45,          # ~1.5 seconds minimum
        min_rally_distance: float = 80,      # Ball must travel at least this far
        # State confirmation
        serve_confirm_frames: int = 8,
        end_confirm_frames: int = 25,        # Longer confirmation for end
        # Velocity smoothing
        velocity_window: int = 5,
    ):
        self.fps = fps
        self.frame_height = frame_height
        self.min_velocity = min_velocity
        self.serve_velocity = serve_velocity
        self.base_gap_during_rally = base_gap_during_rally
        self.base_rally_end_gap = base_rally_end_gap
        self.max_gap_extension = max_gap_extension
        self.lob_velocity_threshold = lob_velocity_threshold
        self.far_court_threshold = far_court_threshold
        self.close_court_threshold = close_court_threshold
        self.min_rally_frames = min_rally_frames
        self.min_rally_distance = min_rally_distance
        self.serve_confirm_frames = serve_confirm_frames
        self.end_confirm_frames = end_confirm_frames
        self.velocity_window = velocity_window
        
        # State
        self.state = RallyState.IDLE
        self.current_rally: Optional[Rally] = None
        self.completed_rallies: List[Rally] = []
        self.rally_counter = 0
        
        # Buffers
        self.position_history = deque(maxlen=60)
        self.velocity_history = deque(maxlen=30)
        self.y_velocity_history = deque(maxlen=10)  # Track vertical movement
        self.frames_without_ball = 0
        self.frames_in_state = 0
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_detection_frame: int = 0
        
        # Lob tracking
        self.is_lob_situation = False
        self.lob_start_frame = 0
    
    def _get_ball_center(self, ball_dict: Dict) -> Optional[Tuple[float, float]]:
        """Extract ball center from detection dict."""
        if not ball_dict:
            return None
        bbox = list(ball_dict.values())[0]
        if any(np.isnan(v) for v in bbox):
            return None
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _compute_velocity(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _get_smoothed_velocity(self) -> float:
        """Get average velocity over recent frames."""
        if not self.velocity_history:
            return 0.0
        return np.mean(list(self.velocity_history))
    
    def _get_vertical_velocity(self) -> float:
        """Get average vertical velocity (negative = moving up)."""
        if not self.y_velocity_history:
            return 0.0
        return np.mean(list(self.y_velocity_history))
    
    def _get_dynamic_gap_tolerance(self) -> Tuple[int, int]:
        """
        Calculate dynamic gap tolerance based on ball position and trajectory.
        
        Returns (gap_during_rally, rally_end_gap)
        """
        extension = 0
        
        # Factor 1: Ball in far court (top of frame) - model works better here
        # but ball takes longer to travel, so extend gap
        if self.last_position is not None:
            y_ratio = self.last_position[1] / self.frame_height
            
            if y_ratio < self.far_court_threshold:
                # Ball in far court - extend tolerance
                extension += int(self.max_gap_extension * 0.5)
            elif y_ratio > self.close_court_threshold:
                # Ball in close court - model unreliable, extend tolerance significantly
                extension += int(self.max_gap_extension * 0.8)
        
        # Factor 2: Lob situation (ball was moving upward)
        if self.is_lob_situation:
            extension += int(self.max_gap_extension * 0.7)
        
        # Factor 3: Recent vertical velocity (ball going up)
        y_vel = self._get_vertical_velocity()
        if y_vel < self.lob_velocity_threshold:
            # Ball is moving upward - likely a lob
            self.is_lob_situation = True
            extension += int(self.max_gap_extension * 0.3)
        
        # Cap extension
        extension = min(extension, self.max_gap_extension)
        
        return (
            self.base_gap_during_rally + extension,
            self.base_rally_end_gap + extension
        )
    
    def _transition(self, new_state: RallyState):
        """Transition to new state."""
        if new_state != self.state:
            self.state = new_state
            self.frames_in_state = 0
            
            # Reset lob tracking on state change (except ACTIVE->ENDING)
            if new_state == RallyState.IDLE:
                self.is_lob_situation = False
    
    def process_frame(self, frame_idx: int, ball_dict: Dict) -> RallyState:
        """Process a single frame and update rally state."""
        position = self._get_ball_center(ball_dict)
        
        # Update position tracking
        if position is not None:
            self.position_history.append((frame_idx, position))
            
            if self.last_position is not None:
                # Total velocity
                velocity = self._compute_velocity(self.last_position, position)
                self.velocity_history.append(velocity)
                
                # Vertical velocity (for lob detection)
                y_velocity = position[1] - self.last_position[1]  # Negative = moving up
                self.y_velocity_history.append(y_velocity)
                
                # Detect lob start
                if y_velocity < self.lob_velocity_threshold and not self.is_lob_situation:
                    self.is_lob_situation = True
                    self.lob_start_frame = frame_idx
            
            self.last_position = position
            self.last_detection_frame = frame_idx
            self.frames_without_ball = 0
        else:
            self.frames_without_ball += 1
        
        self.frames_in_state += 1
        
        # State machine
        if self.state == RallyState.IDLE:
            self._handle_idle(frame_idx, position)
        elif self.state == RallyState.SERVE:
            self._handle_serve(frame_idx, position)
        elif self.state == RallyState.ACTIVE:
            self._handle_active(frame_idx, position)
        elif self.state == RallyState.ENDING:
            self._handle_ending(frame_idx, position)
        
        return self.state
    
    def _handle_idle(self, frame_idx: int, position: Optional[Tuple[float, float]]):
        """IDLE: Waiting for rally to start."""
        if position is None:
            return
        
        velocity = self._get_smoothed_velocity()
        
        if velocity > self.serve_velocity:
            self.rally_counter += 1
            self.current_rally = Rally(
                rally_id=self.rally_counter,
                start_frame=frame_idx
            )
            self.current_rally.ball_positions.append(position)
            self._transition(RallyState.SERVE)
            print(f"  [Frame {frame_idx}] 🎾 Rally #{self.rally_counter} STARTING (velocity: {velocity:.1f})")
    
    def _handle_serve(self, frame_idx: int, position: Optional[Tuple[float, float]]):
        """SERVE: Confirming rally start."""
        if position is not None and self.current_rally:
            self.current_rally.ball_positions.append(position)
        
        gap_during, _ = self._get_dynamic_gap_tolerance()
        
        # Abort if ball disappears too long during serve
        if self.frames_without_ball > gap_during // 2:
            print(f"  [Frame {frame_idx}] ❌ Rally #{self.rally_counter} aborted (ball lost during serve)")
            self._abort_rally()
            return
        
        velocity = self._get_smoothed_velocity()
        
        # Confirm rally after consistent movement
        if self.frames_in_state > self.serve_confirm_frames and velocity > self.min_velocity:
            self._transition(RallyState.ACTIVE)
    
    def _handle_active(self, frame_idx: int, position: Optional[Tuple[float, float]]):
        """ACTIVE: Rally in progress."""
        if position is not None and self.current_rally:
            self.current_rally.ball_positions.append(position)
        
        velocity = self._get_smoothed_velocity()
        gap_during, _ = self._get_dynamic_gap_tolerance()
        
        # Check for rally end conditions
        should_end = (
            self.frames_without_ball > gap_during or
            (velocity < self.min_velocity and self.frames_in_state > 60)
        )
        
        if should_end:
            self._transition(RallyState.ENDING)
    
    def _handle_ending(self, frame_idx: int, position: Optional[Tuple[float, float]]):
        """ENDING: Confirming rally end."""
        if position is not None and self.current_rally:
            self.current_rally.ball_positions.append(position)
        
        velocity = self._get_smoothed_velocity()
        _, rally_end_gap = self._get_dynamic_gap_tolerance()
        
        # Ball resumed - false alarm
        if position is not None and velocity > self.serve_velocity:
            self._transition(RallyState.ACTIVE)
            return
        
        # Confirm end
        if (self.frames_without_ball > rally_end_gap or 
            self.frames_in_state > self.end_confirm_frames):
            self._complete_rally(frame_idx)
    
    def _complete_rally(self, frame_idx: int):
        """Finalize current rally."""
        if self.current_rally is None:
            self._transition(RallyState.IDLE)
            return
        
        self.current_rally.end_frame = frame_idx
        
        # Validate rally
        duration_ok = self.current_rally.duration_frames >= self.min_rally_frames
        
        # Check total distance traveled
        if len(self.current_rally.ball_positions) >= 2:
            positions = np.array(self.current_rally.ball_positions)
            total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
            distance_ok = total_distance >= self.min_rally_distance
        else:
            distance_ok = False
        
        if duration_ok and distance_ok:
            self.completed_rallies.append(self.current_rally)
            duration_sec = self.current_rally.duration_seconds(self.fps)
            print(f"  [Frame {frame_idx}] ✅ Rally #{self.current_rally.rally_id} ENDED - "
                  f"Duration: {duration_sec:.1f}s")
        else:
            reason = "too short" if not duration_ok else "not enough movement"
            print(f"  [Frame {frame_idx}] ❌ Rally #{self.current_rally.rally_id} discarded ({reason})")
        
        self.current_rally = None
        self._transition(RallyState.IDLE)
    
    def _abort_rally(self):
        """Abort current rally (false detection)."""
        self.current_rally = None
        self._transition(RallyState.IDLE)
    
    def process_all_frames(self, ball_detections: List[Dict]) -> List[Rally]:
        """Process all frames and return detected rallies."""
        print(f"\n=== Rally Detection ===")
        print(f"  Processing {len(ball_detections)} frames...")
        print(f"  Base gap tolerance: {self.base_gap_during_rally} frames during rally, "
              f"{self.base_rally_end_gap} frames to end")
        print(f"  Dynamic extension up to +{self.max_gap_extension} frames for lobs/close-ups")
        
        for frame_idx, ball_dict in enumerate(ball_detections):
            self.process_frame(frame_idx, ball_dict)
        
        # Handle video ending during active rally
        if self.current_rally is not None:
            self._complete_rally(len(ball_detections) - 1)
        
        return self.completed_rallies
    
    def get_rally_at_frame(self, frame_idx: int) -> Optional[Rally]:
        """Get rally active at given frame."""
        for rally in self.completed_rallies:
            if rally.start_frame <= frame_idx <= (rally.end_frame or float('inf')):
                return rally
        return None
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.completed_rallies:
            return {"rally_count": 0}
        
        durations = [r.duration_seconds(self.fps) for r in self.completed_rallies]
        
        return {
            "rally_count": len(self.completed_rallies),
            "total_rally_time_sec": sum(durations),
            "avg_duration_sec": np.mean(durations),
            "max_duration_sec": max(durations),
            "min_duration_sec": min(durations)
        }
    
    def draw_overlay(self, video_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Add rally state overlay to video frames."""
        output_frames = []
        
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()
            rally = self.get_rally_at_frame(frame_idx)
            
            if rally is not None:
                # During rally - green indicator
                duration = (frame_idx - rally.start_frame) / self.fps
                
                # Semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 80), (280, 150), (0, 100, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # Text
                cv2.putText(frame, f"RALLY #{rally.rally_id}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Duration: {duration:.1f}s", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Between rallies
                completed = len([r for r in self.completed_rallies 
                               if r.end_frame and r.end_frame <= frame_idx])
                cv2.putText(frame, f"Rallies: {completed}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            output_frames.append(frame)
        
        return output_frames
    
    def export_rallies_csv(self, output_path: str):
        """Export rally data to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rally_id', 'start_frame', 'end_frame', 
                'duration_sec', 'ball_positions_count'
            ])
            
            for rally in self.completed_rallies:
                writer.writerow([
                    rally.rally_id,
                    rally.start_frame,
                    rally.end_frame,
                    round(rally.duration_seconds(self.fps), 2),
                    len(rally.ball_positions)
                ])
        
        print(f"Exported {len(self.completed_rallies)} rallies to {output_path}")