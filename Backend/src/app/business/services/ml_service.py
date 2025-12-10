import asyncio
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set BEFORE importing pyplot - required for background threads on macOS
from typing import Dict, List, Optional, Any
import csv

from .deep_learning.inference import run_main_pipeline
from .deep_learning.utils import generate_heatmap


@dataclass
class PlayerStats:
    """
    Stats for a single player from ML analysis.
    Maps to domain Hits entity for player_1/player_2.
    """
    player_identifier: str  # "player_1", "player_2", etc.
    total_hits: int = 0
    overhead_hits: int = 0
    lob: int = 0
    serve: int = 0
    backhand: int = 0


@dataclass
class RallyData:
    """
    Individual rally from ML analysis.
    Maps to domain Rally entity (only duration is stored).
    """
    rally_id: int
    duration_seconds: float


@dataclass 
class HeatmapData:
    """
    Heatmap output from ML analysis.
    Maps to domain Heatmap entity (stores binary PNG).
    """
    player_identifier: str
    image_path: Path  # Temp file path - read as bytes for storage


@dataclass
class MLAnalysisResult:
    """Complete result from ML pipeline"""
    player_stats: Dict[str, PlayerStats]  # keyed by player_identifier
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
    - Generate heatmaps for specific players
    - Transform results into domain-compatible format
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
        # Default paths relative to deep_learning folder
        base_dir = Path(__file__).parent / "deep_learning"
        
        self.models_dir = models_dir or base_dir / "models"
        self.court_config_path = court_config_path or base_dir / "court_info" / "court_information.json"
        self.output_dir = output_dir or base_dir / "output_videos"
        
        # Ensure output dir exists
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
            MLAnalysisResult with all analysis data
        """
        # Run in thread pool since inference is CPU/GPU bound
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
        
        # Build argument list for the pipeline
        video_stem = video_path.stem
        output_subdir = self.output_dir / video_stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"[MLService] Starting analysis for: {video_path}", flush=True)
        print(f"[MLService] Court number: {court_number}", flush=True)
        print(f"[MLService] Output dir: {output_subdir}", flush=True)
        
        # Create args namespace mimicking argparse output
        args = self._build_pipeline_args(
            video_path=video_path,
            court_number=court_number,
            output_dir=output_subdir,
            fps=fps
        )
        
        # Run the main pipeline
        run_main_pipeline(args)
        
        print(f"[MLService] Pipeline complete, parsing results...", flush=True)
        
        # Parse outputs - note: filenames from inference.py
        player_stats = self._parse_shot_csv(output_subdir / f"{video_stem}_shots.csv")
        rallies = self._parse_rally_csv(output_subdir / f"{video_stem}_rallies.csv", fps)  # Note: rallies plural
        
        # Generate per-player heatmaps
        heatmaps = {}
        player_csv = output_subdir / f"{video_stem}_player_positions.csv"
        court_frame = output_subdir / f"{video_stem}_court_frame.png"
        
        print(f"[MLService] Generating per-player heatmaps...", flush=True)
        print(f"[MLService] Player CSV exists: {player_csv.exists()}", flush=True)
        print(f"[MLService] Court frame exists: {court_frame.exists()}", flush=True)
        
        # Map ML track_ids to our player identifiers
        # Based on player_positions CSV: track_id 1,2 are near-court players
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
                player_id=track_id  # Pass the track_id for filtering
            )
            if heatmap_path.exists():
                heatmaps[player_id] = HeatmapData(
                    player_identifier=player_id,
                    image_path=heatmap_path
                )
                print(f"[MLService] Generated heatmap for {player_id}: {heatmap_path}", flush=True)
            else:
                print(f"[MLService] Failed to generate heatmap for {player_id}", flush=True)
        
        print(f"[MLService] Analysis complete. Rallies: {len(rallies)}, Heatmaps: {len(heatmaps)}", flush=True)
        
        # Build result
        return MLAnalysisResult(
            player_stats=player_stats,
            rallies=rallies,
            heatmaps=heatmaps,
            total_rallies=len(rallies),
            output_video_path=output_subdir / f"{video_stem}_output.avi"
        )
    
    def _build_pipeline_args(
        self,
        video_path: Path,
        court_number: int,
        output_dir: Path,
        fps: float
    ):
        """Build args namespace for inference pipeline"""
        from types import SimpleNamespace
        
        return SimpleNamespace(
            mode="full",
            input_video=str(video_path),
            output_video=None,  # Auto-generated
            output_dir=str(output_dir),
            court_json=str(self.court_config_path),
            court_number=court_number,
            fps=fps,
            
            # Model paths
            tracknet_model=str(self.models_dir / "TrackNet_best.pt"),
            player_model=str(self.models_dir / "yolov8s.pt"),
            shot_model=str(self.models_dir / "best_model.pth"),
            yolo_pose=str(self.models_dir / "yolov8n-pose.pt"),
            
            # Stubs disabled for real inference
            use_stubs=False,
            player_stub=None,
            ball_stub=None,
            
            # Detection thresholds
            ball_threshold=0.5,
            min_heatmap_conf=0.5,
            confidence_threshold=None,
            
            # Rally parameters
            min_velocity=3.0,
            serve_velocity=6.0,
            base_gap_during_rally=40,
            base_rally_end_gap=70,
            max_gap_extension=60,
            min_rally_frames=45,
            min_rally_distance=80,
            
            # Shot classification
            enable_shot_classification=True,
            window_size=32,
            stride=8,
            yolo_resolution=[640, 360],
            classifier_resolution=128,
            
            # Visualization
            trail_length=10,
            draw_exclusion_zones=True,
            generate_heatmap=True,
            
            # Heatmap settings
            heatmap_bins_x=200,
            heatmap_bins_y=100,
            heatmap_gauss=9,
            heatmap_inplay_only=False,
            heatmap_min_conf=None,
            heatmap_players="1,2",  # Only analyze players 1 and 2
            heatmap_alpha=0.7,
            heatmap_show_axes=False,
            
            # Calibration (from JSON)
            calib=None,
            calib_csv=None,
            court_w=20.0,
            court_h=10.0,
        )
    
    def _parse_shot_csv(self, csv_path: Path) -> Dict[str, PlayerStats]:
        """Parse shot classification CSV output"""
        stats = {
            "player_1": PlayerStats(player_identifier="player_1"),
            "player_2": PlayerStats(player_identifier="player_2"),
            "player_3": PlayerStats(player_identifier="player_3"),
            "player_4": PlayerStats(player_identifier="player_4"),
        }
        
        # Map from ML player_id to our player identifiers
        # ML typically uses 0, 2 for near-court players and 1, 3 for far-court
        # Based on actual CSV output: player_id 0 and 2 are near court
        track_id_to_player = {
            "0": "player_1",
            "2": "player_2",
            "1": "player_3",  # Far court (if present)
            "3": "player_4",  # Far court (if present)
        }
        
        if not csv_path.exists():
            print(f"[MLService] Shot CSV not found: {csv_path}", flush=True)
            return stats
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSV columns: frame, shot_type, player_id, confidence
                    raw_player_id = str(row.get('player_id', '')).strip()
                    shot_type = row.get('shot_type', '').lower().strip()
                    
                    # Map track_id to player identifier
                    player_id = track_id_to_player.get(raw_player_id)
                    if not player_id:
                        print(f"[MLService] Unknown player_id in CSV: {raw_player_id}", flush=True)
                        continue
                    
                    player = stats[player_id]
                    player.total_hits += 1
                    
                    # Categorize shot type
                    if shot_type == 'lob':
                        player.lob += 1
                    elif shot_type == 'serve':
                        player.serve += 1
                    elif shot_type == 'backhand':
                        player.backhand += 1
                    elif shot_type in ['smash', 'overhead']:
                        player.overhead_hits += 1
                    # Other shot types just count toward total_hits
                    
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
                    # CSV columns: rally_id, start_frame, end_frame, duration_sec, ball_positions_count
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
                players=[player_id],  # Filter to specific player
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