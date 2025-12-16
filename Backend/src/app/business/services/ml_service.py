import asyncio
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set BEFORE importing pyplot - required for background threads on macOS
from typing import Dict, List, Optional

import csv

from .deep_learning.config import PipelineConfig
from .deep_learning.pipeline import run_main_pipeline
from .deep_learning.utils import generate_heatmap


# Constants for tracked players (only near-side players)
TRACKED_PLAYERS = ["player_1", "player_2"]


@dataclass
class PlayerStats:
    """
    Stats for a single player from ML analysis.
    Only populated for player_1 and player_2 (near-side players).
    """
    player_identifier: str  # "player_1" or "player_2"
    total_hits: int = 0
    overhead_hits: int = 0
    lob: int = 0
    serve: int = 0
    groundstrokes: int = 0


@dataclass
class RallyData:
    """
    Individual rally from ML analysis.
    Rallies are match-level, not player-specific.
    """
    rally_id: int
    duration_seconds: float


@dataclass 
class HeatmapData:
    """
    Heatmap output from ML analysis.
    Only generated for player_1 and player_2.
    """
    player_identifier: str
    image_path: Path  # Temp file path - read as bytes for storage


@dataclass
class MLAnalysisResult:
    """Complete result from ML pipeline - only includes tracked players"""
    player_stats: Dict[str, PlayerStats]  # keyed by player_identifier (player_1, player_2)
    rallies: List[RallyData]
    heatmaps: Dict[str, HeatmapData]  # keyed by player_identifier
    total_rallies: int
    output_video_path: Optional[Path] = None


class MLService:
    """
    Service for running ML inference on padel videos.
    
    Responsibilities:
    - Run the inference pipeline
    - Parse CSV outputs from the pipeline
    - Generate heatmaps for player_1 and player_2 only
    - Transform results into domain-compatible format
    
    Design Decision:
    - Only tracks player_1 and player_2 (near-side of court)
    - player_3 and player_4 (far-side) don't get meaningful tracking data
    """
    
    def __init__(
        self,
        models_dir: Path = None,
        court_config_path: Path = None,
        output_dir: Path = None
    ):
        """
        Initialize ML Service.
        
        Args:
            models_dir: Directory containing ML model files
            court_config_path: Path to court_information.json
            output_dir: Directory for pipeline outputs
        """
        base_dir = Path(__file__).parent / "deep_learning"
        
        self.models_dir = models_dir or base_dir / "models"
        self.court_config_path = court_config_path or base_dir / "court_info" / "court_information.json"
        self.output_dir = output_dir or base_dir / "output_videos"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_analysis(
        self,
        video_path: Path,
        court_number: int,
        fps: float = 30.0
    ) -> MLAnalysisResult:
        """
        Run full ML analysis pipeline on a video.
        
        Args:
            video_path: Path to the video file
            court_number: Court number for calibration lookup
            fps: Video framerate
            
        Returns:
            MLAnalysisResult with data for player_1 and player_2 only
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_analysis_sync,
            video_path,
            court_number,
            fps
        )
    
    def _run_analysis_sync(
        self,
        video_path: Path,
        court_number: int,
        fps: float
    ) -> MLAnalysisResult:
        """Synchronous analysis - runs in thread pool"""
        
        video_stem = video_path.stem
        output_subdir = self.output_dir / video_stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"[MLService] Starting analysis for: {video_path}", flush=True)
        print(f"[MLService] Court number: {court_number}", flush=True)
        print(f"[MLService] Output dir: {output_subdir}", flush=True)
        print(f"[MLService] Tracking players: {TRACKED_PLAYERS}", flush=True)
        
        config = self._build_pipeline_config(
            video_path=video_path,
            court_number=court_number,
            output_dir=output_subdir,
            fps=fps
        )
        
        run_main_pipeline(config)
        
        print(f"[MLService] Pipeline complete, parsing results...", flush=True)
        
        # Parse outputs - only for tracked players
        player_stats = self._parse_shot_csv(output_subdir / f"{video_stem}_shots.csv")
        rallies = self._parse_rally_csv(output_subdir / f"{video_stem}_rallies.csv", fps)
        
        # Generate per-player heatmaps (only for tracked players)
        heatmaps = {}
        player_csv = output_subdir / f"{video_stem}_player_positions.csv"
        court_frame = output_subdir / f"{video_stem}_court_frame.png"
        
        print(f"[MLService] Generating heatmaps for tracked players only...", flush=True)
        
        # Map ML track_ids to our player identifiers (only near-side)
        track_to_player = {
            1: "player_1",
            2: "player_2",
        }
        
        for track_id, player_id in track_to_player.items():
            heatmap_path = output_subdir / f"{video_stem}_heatmap_{player_id}.png"
            self._generate_player_heatmap(
                player_csv=player_csv,
                court_img=court_frame,
                output_path=heatmap_path,
                player_id=track_id
            )
            if heatmap_path.exists():
                heatmaps[player_id] = HeatmapData(
                    player_identifier=player_id,
                    image_path=heatmap_path
                )
                print(f"[MLService] Generated heatmap for {player_id}: {heatmap_path}", flush=True)
            else:
                print(f"[MLService] Failed to generate heatmap for {player_id}", flush=True)
        
        print(f"[MLService] Analysis complete. Players: {len(player_stats)}, Rallies: {len(rallies)}, Heatmaps: {len(heatmaps)}", flush=True)
        
        return MLAnalysisResult(
            player_stats=player_stats,
            rallies=rallies,
            heatmaps=heatmaps,
            total_rallies=len(rallies),
            output_video_path=output_subdir / f"{video_stem}_output.avi"
        )
    
    def _build_pipeline_config(
        self,
        video_path: Path,
        court_number: int,
        output_dir: Path,
        fps: float
    ) -> PipelineConfig:
        """Build typed PipelineConfig for inference pipeline"""
        config = PipelineConfig()
        
        # Video settings
        config.video.input_path = str(video_path)
        config.video.output_path = None
        config.video.output_dir = str(output_dir)
        config.video.fps = fps
        
        # Court settings
        config.court.court_json = str(self.court_config_path)
        config.court.court_number = court_number
        config.court.court_width = 20.0
        config.court.court_height = 10.0
        
        # Model paths
        config.models.tracknet = str(self.models_dir / "TrackNet_best.pt")
        config.models.player_detector = str(self.models_dir / "yolov8s.pt")
        config.models.shot_classifier = str(self.models_dir / "best_model.pth")
        config.models.yolo_pose = str(self.models_dir / "yolov8n-pose.pt")
        
        config.stubs.use_stubs = False
        
        # Ball detection
        config.ball.detection_threshold = 0.5
        config.ball.min_heatmap_confidence = 0.5
        config.ball.trail_length = 10
        config.ball.draw_exclusion_zones = True
        
        # Rally parameters
        config.rally.min_velocity = 3.0
        config.rally.serve_velocity = 6.0
        config.rally.base_gap_during_rally = 40
        config.rally.base_rally_end_gap = 70
        config.rally.max_gap_extension = 60
        config.rally.min_rally_frames = 45
        config.rally.min_rally_distance = 80
        
        # Shot classification
        config.shot.enabled = True
        config.shot.window_size = 32
        config.shot.stride = 8
        config.shot.yolo_resolution = (640, 360)
        config.shot.classifier_resolution = 128
        
        # Heatmap settings - only for player 1 and 2
        config.heatmap.bins_x = 200
        config.heatmap.bins_y = 100
        config.heatmap.gaussian_sigma = 9
        config.heatmap.inplay_only = False
        config.heatmap.players_filter = "1,2"  # Only track near-side players
        config.heatmap.alpha = 0.7
        config.heatmap.show_axes = False
        
        return config
    
    def _parse_shot_csv(self, csv_path: Path) -> Dict[str, PlayerStats]:
        """
        Parse shot classification CSV output.
        Only returns stats for player_1 and player_2.
        """
        # Initialize only for tracked players
        stats = {
            player_id: PlayerStats(player_identifier=player_id)
            for player_id in TRACKED_PLAYERS
        }
        
        # ML track_id to our player identifier mapping
        # Only map track IDs that correspond to near-side players
        track_id_to_player = {
            "0": "player_1",
            "2": "player_2",
        }
        
        if not csv_path.exists():
            print(f"[MLService] Shot CSV not found: {csv_path}", flush=True)
            return stats
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_player_id = str(row.get('player_id', '')).strip()
                    shot_type = row.get('shot_type', '').lower().strip()
                    
                    # Only process tracked players
                    player_id = track_id_to_player.get(raw_player_id)
                    if not player_id:
                        # Skip far-side players (player_3, player_4)
                        continue
                    
                    player = stats[player_id]
                    player.total_hits += 1
                    
                    if shot_type == 'lob':
                        player.lob += 1
                    elif shot_type == 'serve':
                        player.serve += 1
                    elif shot_type == 'groundstrokes':
                        player.groundstrokes += 1  # Changed from backhand
                    elif shot_type in ['smash', 'overhead']:
                        player.overhead_hits += 1
                    
            print(f"[MLService] Parsed shots - player_1: {stats['player_1'].total_hits}, player_2: {stats['player_2'].total_hits}", flush=True)
                    
        except Exception as e:
            print(f"[MLService] Error parsing shot CSV: {e}", flush=True)
        
        return stats
    
    def _parse_rally_csv(self, csv_path: Path, fps: float) -> List[RallyData]:
        """Parse rally detection CSV output"""
        rallies = []
        
        if not csv_path.exists():
            print(f"[MLService] Rally CSV not found: {csv_path}", flush=True)
            return rallies
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rally = RallyData(
                        rally_id=int(row.get('rally_id', 0)),
                        duration_seconds=float(row.get('duration_sec', 0))
                    )
                    rallies.append(rally)
            
            print(f"[MLService] Parsed {len(rallies)} rallies", flush=True)
            
        except Exception as e:
            print(f"[MLService] Error parsing rally CSV: {e}", flush=True)
        
        return rallies
    
    def _generate_player_heatmap(
        self,
        player_csv: Path,
        court_img: Path,
        output_path: Path,
        player_id: int
    ):
        """Generate heatmap for a specific player"""
        if not player_csv.exists() or not court_img.exists():
            print(f"[MLService] Missing files for heatmap - CSV: {player_csv.exists()}, Court: {court_img.exists()}", flush=True)
            return
        
        try:
            generate_heatmap(
                csv_path=str(player_csv),
                court_img_path=str(court_img),
                out_png=str(output_path),
                bins_x=200,
                bins_y=100,
                gauss=9,
                inplay_only=False,
                min_conf=None,
                players=[player_id],
                heat_alpha=0.7,
                show_axes=False
            )
            print(f"[MLService] Heatmap generated: {output_path}", flush=True)
        except Exception as e:
            print(f"[MLService] Error generating heatmap for player {player_id}: {e}", flush=True)
    
    def read_heatmap_binary(self, heatmap_path: Path) -> Optional[bytes]:
        """Read heatmap PNG as binary data for database storage"""
        if not heatmap_path.exists():
            return None
        
        with open(heatmap_path, 'rb') as f:
            return f.read()