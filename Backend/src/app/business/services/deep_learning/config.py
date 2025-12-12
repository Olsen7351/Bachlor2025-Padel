"""
Configuration dataclasses for the Padel Analysis Pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class ModelPaths:
    """Paths to all ML models used in the pipeline."""
    tracknet: str = "models/TrackNet_best.pt"
    player_detector: str = "models/yolov8s.pt"
    shot_classifier: str = "models/best_model.pth"
    yolo_pose: str = "models/yolov8n-pose.pt"


@dataclass
class VideoConfig:
    """Video input/output configuration."""
    input_path: str = "input_videos/video2_trimmed.mp4"
    output_path: Optional[str] = None
    output_dir: str = "output_videos"
    fps: float = 30.0


@dataclass
class CourtConfig:
    """Court configuration and calibration settings."""
    court_number: int = 9
    court_json: str = "court_info/court_information.json"
    court_width: float = 20.0
    court_height: float = 10.0
    calib_csv: Optional[str] = None
    calib_string: Optional[str] = None  # 4-corner calibration string


@dataclass
class BallDetectionConfig:
    """Ball detection parameters."""
    detection_threshold: float = 0.5
    min_heatmap_confidence: float = 0.5
    trail_length: int = 10
    draw_exclusion_zones: bool = True


@dataclass
class RallyDetectionConfig:
    """Rally detection parameters."""
    min_velocity: float = 3.0
    serve_velocity: float = 6.0
    base_gap_during_rally: int = 40
    base_rally_end_gap: int = 70
    max_gap_extension: int = 60
    min_rally_frames: int = 45
    min_rally_distance: int = 80


@dataclass
class ShotClassificationConfig:
    """Shot classification parameters."""
    enabled: bool = True
    confidence_threshold: Optional[float] = None
    window_size: int = 32
    stride: int = 8
    yolo_resolution: Tuple[int, int] = (640, 360)
    classifier_resolution: int = 128


@dataclass
class HeatmapConfig:
    """Heatmap generation parameters."""
    bins_x: int = 200
    bins_y: int = 100
    gaussian_sigma: int = 9
    inplay_only: bool = False
    min_confidence: Optional[float] = None
    players_filter: str = ""
    alpha: float = 0.7
    show_axes: bool = False


@dataclass
class StubConfig:
    """Stub/cache configuration for development."""
    use_stubs: bool = False
    player_stub: str = "tracker_stubs/player_detections.pkl"
    ball_stub: str = "tracker_stubs/ball_detections.pkl"


@dataclass
class PlayerTrackingConfig:
    """Standalone player tracking configuration."""
    confidence: float = 0.25
    image_size: Optional[int] = None
    device: Optional[str] = None
    tracker_config: str = "bytetrack.yaml"
    auto_heatmap: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration combining all sub-configs."""
    models: ModelPaths = field(default_factory=ModelPaths)
    video: VideoConfig = field(default_factory=VideoConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    ball: BallDetectionConfig = field(default_factory=BallDetectionConfig)
    rally: RallyDetectionConfig = field(default_factory=RallyDetectionConfig)
    shot: ShotClassificationConfig = field(default_factory=ShotClassificationConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    stubs: StubConfig = field(default_factory=StubConfig)
    player_tracking: PlayerTrackingConfig = field(default_factory=PlayerTrackingConfig)
    
    @property
    def device(self) -> torch.device:
        """Get the compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def config_from_args(args) -> PipelineConfig:
    """Create PipelineConfig from argparse namespace."""
    config = PipelineConfig()
    
    # Model paths
    config.models.tracknet = getattr(args, 'tracknet_model', None) or getattr(args, 'tracknet', config.models.tracknet)
    config.models.player_detector = getattr(args, 'player_model', None) or getattr(args, 'model', config.models.player_detector)
    config.models.shot_classifier = getattr(args, 'shot_model', config.models.shot_classifier)
    config.models.yolo_pose = getattr(args, 'yolo_pose', config.models.yolo_pose)
    
    # Video config
    config.video.input_path = getattr(args, 'input_video', None) or getattr(args, 'input', None) or getattr(args, 'video', config.video.input_path)
    config.video.output_path = getattr(args, 'output_video', None) or getattr(args, 'output', None) or getattr(args, 'out_video', None)
    config.video.output_dir = getattr(args, 'output_dir', config.video.output_dir)
    config.video.fps = getattr(args, 'fps', config.video.fps)
    
    # Court config
    config.court.court_number = getattr(args, 'court_number', config.court.court_number)
    config.court.court_json = getattr(args, 'court_json', config.court.court_json)
    config.court.court_width = getattr(args, 'court_w', config.court.court_width)
    config.court.court_height = getattr(args, 'court_h', config.court.court_height)
    config.court.calib_csv = getattr(args, 'calib_csv', None)
    config.court.calib_string = getattr(args, 'calib', None)
    
    # Ball detection config
    config.ball.detection_threshold = getattr(args, 'ball_threshold', None) or getattr(args, 'threshold', config.ball.detection_threshold)
    config.ball.min_heatmap_confidence = getattr(args, 'min_heatmap_conf', config.ball.min_heatmap_confidence)
    config.ball.trail_length = getattr(args, 'trail_length', config.ball.trail_length)
    config.ball.draw_exclusion_zones = getattr(args, 'draw_exclusion_zones', config.ball.draw_exclusion_zones)
    
    # Rally detection config
    config.rally.min_velocity = getattr(args, 'min_velocity', config.rally.min_velocity)
    config.rally.serve_velocity = getattr(args, 'serve_velocity', config.rally.serve_velocity)
    config.rally.base_gap_during_rally = getattr(args, 'base_gap_during_rally', config.rally.base_gap_during_rally)
    config.rally.base_rally_end_gap = getattr(args, 'base_rally_end_gap', config.rally.base_rally_end_gap)
    config.rally.max_gap_extension = getattr(args, 'max_gap_extension', config.rally.max_gap_extension)
    config.rally.min_rally_frames = getattr(args, 'min_rally_frames', config.rally.min_rally_frames)
    config.rally.min_rally_distance = getattr(args, 'min_rally_distance', config.rally.min_rally_distance)
    
    # Shot classification config
    config.shot.enabled = getattr(args, 'enable_shot_classification', config.shot.enabled)
    config.shot.confidence_threshold = getattr(args, 'confidence_threshold', None)
    config.shot.window_size = getattr(args, 'window_size', config.shot.window_size)
    config.shot.stride = getattr(args, 'stride', config.shot.stride)
    yolo_res = getattr(args, 'yolo_resolution', None)
    if yolo_res:
        config.shot.yolo_resolution = tuple(yolo_res)
    config.shot.classifier_resolution = getattr(args, 'classifier_resolution', config.shot.classifier_resolution)
    
    # Heatmap config
    config.heatmap.bins_x = getattr(args, 'heatmap_bins_x', config.heatmap.bins_x)
    config.heatmap.bins_y = getattr(args, 'heatmap_bins_y', config.heatmap.bins_y)
    config.heatmap.gaussian_sigma = getattr(args, 'heatmap_gauss', config.heatmap.gaussian_sigma)
    config.heatmap.inplay_only = getattr(args, 'heatmap_inplay_only', config.heatmap.inplay_only)
    config.heatmap.min_confidence = getattr(args, 'heatmap_min_conf', None)
    config.heatmap.players_filter = getattr(args, 'heatmap_players', config.heatmap.players_filter)
    config.heatmap.alpha = getattr(args, 'heatmap_alpha', config.heatmap.alpha)
    config.heatmap.show_axes = getattr(args, 'heatmap_show_axes', config.heatmap.show_axes)
    
    # Stub config
    config.stubs.use_stubs = getattr(args, 'use_stubs', config.stubs.use_stubs)
    config.stubs.player_stub = getattr(args, 'player_stub', config.stubs.player_stub)
    config.stubs.ball_stub = getattr(args, 'ball_stub', config.stubs.ball_stub)
    
    # Player tracking config
    config.player_tracking.confidence = getattr(args, 'conf', config.player_tracking.confidence)
    config.player_tracking.image_size = getattr(args, 'imgsz', None)
    config.player_tracking.device = getattr(args, 'device', None)
    config.player_tracking.tracker_config = getattr(args, 'tracker', config.player_tracking.tracker_config)
    config.player_tracking.auto_heatmap = getattr(args, 'auto_heatmap', config.player_tracking.auto_heatmap)
    
    return config