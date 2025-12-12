"""
Padel Analysis Pipeline Orchestrator.

Coordinates all pipeline stages:
1. Shot Classification
2. Player Detection
3. Ball Detection  
4. Rally Detection
5. Player Position Export & Heatmap
6. Combined Output Video
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Set

from .config import PipelineConfig
from .trackers import BallTrackerTrackNet, PlayerTracker, RallyTracker
from .processors import ShotClassificationProcessor, export_player_positions_csv
from .utils import (
    read_video,
    save_video,
    ensure_dir,
    get_output_paths,
    load_court_config,
    generate_heatmap,
    parse_calib_points,
    build_homography,
    load_calib_csv
)


class PadelAnalysisPipeline:
    """
    Main pipeline orchestrator for padel match analysis.
    
    Coordinates all analysis stages and produces combined outputs.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.output_paths: Dict[str, str] = {}
        self.court_config: Dict = {}
        self.homography: Optional[np.ndarray] = None
        
        # Results storage
        self.video_frames: List[np.ndarray] = []
        self.player_detections: List[Dict] = []
        self.ball_detections: List[Dict] = []
        self.shot_data: Optional[Dict] = None
        self.rallies: List = []
        self.rally_frames: Set[int] = set()
    
    def run(self, output_video: Optional[str] = None) -> Dict[str, str]:
        """
        Run the complete analysis pipeline.
        
        Args:
            output_video: Optional override for output video path
            
        Returns:
            Dict of output file paths
        """
        print("=" * 60)
        print("PADEL MATCH ANALYSIS - FULL PIPELINE")
        print("=" * 60)
        
        # Setup
        self._setup(output_video)
        
        # Load video
        self._load_video()
        
        # Run pipeline stages
        if self.config.shot.enabled:
            self._run_shot_classification()
        
        self._run_player_detection()
        self._run_ball_detection()
        self._run_rally_detection()
        self._export_player_positions()
        self._generate_combined_output()
        
        self._print_summary()
        
        return self.output_paths
    
    def _setup(self, output_video: Optional[str] = None) -> None:
        """Initialize paths and load court configuration."""
        # Get output paths
        self.output_paths = get_output_paths(
            self.config.video.input_path, 
            self.config.video.output_dir
        )
        
        if output_video:
            self.output_paths['video'] = output_video
        elif self.config.video.output_path:
            self.output_paths['video'] = self.config.video.output_path
        
        # Load court configuration
        self.court_config = load_court_config(
            self.config.court.court_number, 
            self.config.court.court_json
        )
        
        self.homography = self.court_config.get('HOMOGRAPHY')
        
        if self.homography is not None:
            print(f"Loaded homography from court {self.config.court.court_number}")
        else:
            print(f"No calibration points found for court {self.config.court.court_number}")
            self._try_load_fallback_homography()
    
    def _try_load_fallback_homography(self) -> None:
        """Try to load homography from fallback sources."""
        if self.config.court.calib_csv:
            try:
                self.homography = load_calib_csv(self.config.court.calib_csv)
                print(f"Loaded homography from {self.config.court.calib_csv}")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
        elif self.config.court.calib_string:
            try:
                img_pts = parse_calib_points(self.config.court.calib_string)
                self.homography = build_homography(
                    img_pts, 
                    court_w=self.config.court.court_width, 
                    court_h=self.config.court.court_height
                )
                print("Loaded homography from 4-corner calibration string")
            except Exception as e:
                print(f"Warning: Could not parse calibration: {e}")
    
    def _load_video(self) -> None:
        """Load video frames."""
        self.video_frames = read_video(self.config.video.input_path)
        frame_height, frame_width = self.video_frames[0].shape[:2]
        
        print(f"\nVideo: {len(self.video_frames)} frames, {frame_width}x{frame_height}, "
              f"{self.config.video.fps} fps")
        
        # Save first frame for heatmap
        court_frame_path = self.output_paths['court_frame']
        ensure_dir(court_frame_path)
        cv2.imwrite(court_frame_path, self.video_frames[0])
        print(f"Saved court frame: {court_frame_path}")
    
    def _run_shot_classification(self) -> None:
        """Step 1: Shot Classification."""
        print("\n" + "=" * 60)
        print("STEP 1: Shot Classification")
        print("=" * 60)
        
        try:
            processor = ShotClassificationProcessor(
                shot_model_path=self.config.models.shot_classifier,
                yolo_pose_path=self.config.models.yolo_pose,
                tracknet_path=self.config.models.tracknet,
                device=self.config.device,
                confidence_threshold=self.config.shot.confidence_threshold,
                window_size=self.config.shot.window_size,
                stride=self.config.shot.stride,
                classifier_resolution=self.config.shot.classifier_resolution,
            )
            
            self.shot_data = processor.process_video(
                self.video_frames, 
                fps=self.config.video.fps, 
                yolo_resolution=self.config.shot.yolo_resolution
            )
            
            # Export shots CSV
            processor.export_shots_csv(self.output_paths['shots_csv'])
            
            # Store processor for later drawing
            self._shot_processor = processor
            
            print(f"\nShot classification complete: {len(self.shot_data['shot_events'])} shots detected")
            print(f"Player stats: {processor.player_stats}")
            
        except Exception as e:
            print(f"Warning: Shot classification failed: {e}")
            print("Continuing without shot classification...")
            self.shot_data = None
            self._shot_processor = None
    
    def _run_player_detection(self) -> None:
        """Step 2: Player Detection."""
        print("\n" + "=" * 60)
        print("STEP 2: Player Detection (YOLO)")
        print("=" * 60)
        
        player_tracker = PlayerTracker(model_path=self.config.models.player_detector)
        self.player_detections = player_tracker.detect_frames(
            self.video_frames,
            read_from_stub=self.config.stubs.use_stubs,
            stub_path=self.config.stubs.player_stub if self.config.stubs.use_stubs else None
        )
        
        # Store for drawing
        self._player_tracker = player_tracker
    
    def _run_ball_detection(self) -> None:
        """Step 3: Ball Detection."""
        print("\n" + "=" * 60)
        print("STEP 3: Ball Detection (TrackNet with Filters)")
        print("=" * 60)
        
        ball_tracker = BallTrackerTrackNet(
            tracknet_path=self.config.models.tracknet,
            detection_threshold=self.config.ball.detection_threshold,
            min_heatmap_confidence=self.config.ball.min_heatmap_confidence,
        )
        
        # Add exclusion zones from court config
        left_zone = self.court_config.get('LEFT_EXCLUSION_ZONE')
        right_zone = self.court_config.get('RIGHT_EXCLUSION_ZONE')
        
        if left_zone:
            ball_tracker.add_exclusion_zone(left_zone, "left_glass")
        if right_zone:
            ball_tracker.add_exclusion_zone(right_zone, "right_glass")
        
        ball_tracker.use_exclusion_filter = True
        ball_tracker.use_player_filter = True
        ball_tracker.use_trajectory_filter = True
        
        self.ball_detections = ball_tracker.detect_frames(
            self.video_frames,
            player_detections=self.player_detections,
            read_from_stub=self.config.stubs.use_stubs,
            stub_path=self.config.stubs.ball_stub if self.config.stubs.use_stubs else None,
        )
        
        detected_count = sum(1 for d in self.ball_detections if d)
        print(f"\nBall detected in {detected_count}/{len(self.video_frames)} frames "
              f"({100 * detected_count / len(self.video_frames):.1f}%)")
        
        # Store for drawing
        self._ball_tracker = ball_tracker
    
    def _run_rally_detection(self) -> None:
        """Step 4: Rally Detection."""
        print("\n" + "=" * 60)
        print("STEP 4: Rally Detection")
        print("=" * 60)
        
        frame_height = self.video_frames[0].shape[0]
        
        rally_tracker = RallyTracker(
            fps=self.config.video.fps,
            frame_height=frame_height,
            min_velocity=self.config.rally.min_velocity,
            serve_velocity=self.config.rally.serve_velocity,
            base_gap_during_rally=self.config.rally.base_gap_during_rally,
            base_rally_end_gap=self.config.rally.base_rally_end_gap,
            max_gap_extension=self.config.rally.max_gap_extension,
            min_rally_frames=self.config.rally.min_rally_frames,
            min_rally_distance=self.config.rally.min_rally_distance,
        )
        
        self.rallies = rally_tracker.process_all_frames(self.ball_detections)
        
        summary = rally_tracker.get_summary()
        print("\n" + "-" * 40)
        print("RALLY SUMMARY")
        print("-" * 40)
        print(f"  Total rallies: {summary['rally_count']}")
        
        self.rally_frames = set()
        if summary['rally_count'] > 0:
            print(f"  Total rally time: {summary['total_rally_time_sec']:.1f}s")
            print(f"  Average duration: {summary['avg_duration_sec']:.1f}s")
            
            for rally in self.rallies:
                print(f"  Rally #{rally.rally_id}: "
                      f"Frames {rally.start_frame}-{rally.end_frame} "
                      f"({rally.duration_seconds(self.config.video.fps):.1f}s)")
                for f in range(rally.start_frame, rally.end_frame + 1):
                    self.rally_frames.add(f)
        
        rally_tracker.export_rallies_csv(self.output_paths['rally_csv'])
        
        # Store for drawing
        self._rally_tracker = rally_tracker
    
    def _export_player_positions(self) -> None:
        """Step 5: Player Position Export & Heatmap."""
        print("\n" + "=" * 60)
        print("STEP 5: Player Position Export & Heatmap")
        print("=" * 60)
        
        export_player_positions_csv(
            self.player_detections, 
            self.output_paths['player_csv'], 
            fps=self.config.video.fps,
            homography=self.homography,
            court_w=self.config.court.court_width,
            court_h=self.config.court.court_height,
            rally_frames=self.rally_frames
        )
        
        # Generate heatmap
        print("\nGenerating player heatmap...")
        
        players_filter = None
        if self.config.heatmap.players_filter:
            players_filter = [int(s) for s in self.config.heatmap.players_filter.split(",") 
                            if s.strip().isdigit()]
        
        try:
            generate_heatmap(
                csv_path=self.output_paths['player_csv'],
                court_img_path=self.output_paths['court_frame'],
                out_png=self.output_paths['heatmap'],
                bins_x=self.config.heatmap.bins_x,
                bins_y=self.config.heatmap.bins_y,
                gauss=self.config.heatmap.gaussian_sigma,
                inplay_only=self.config.heatmap.inplay_only,
                min_conf=self.config.heatmap.min_confidence,
                players=players_filter,
                heat_alpha=self.config.heatmap.alpha,
                show_axes=self.config.heatmap.show_axes,
            )
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")
    
    def _generate_combined_output(self) -> None:
        """Step 6: Generate Combined Output Video."""
        print("\n" + "=" * 60)
        print("STEP 6: Generating Combined Output Video")
        print("=" * 60)
        
        # Start with original frames
        output_frames = [f.copy() for f in self.video_frames]
        
        # Layer 1: Player bboxes
        output_frames = self._player_tracker.draw_bboxes(output_frames, self.player_detections)
        
        # Layer 2: Ball tracking
        output_frames = self._ball_tracker.draw_bboxes(
            output_frames,
            self.ball_detections,
            trail_length=self.config.ball.trail_length,
            show_confidence=True,
            draw_exclusion_zones=self.config.ball.draw_exclusion_zones,
        )
        
        # Layer 3: Shot classification overlay
        if self.shot_data and self._shot_processor:
            print("Adding shot classification overlay...")
            output_frames = self._shot_processor.draw_shot_overlay(output_frames, self.shot_data)
        
        # Layer 4: Rally overlay
        output_frames = self._rally_tracker.draw_overlay(output_frames)
        
        # Add frame counter
        for i, frame in enumerate(output_frames):
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Frame: {i}", (w - 200, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 221), 2)
        
        # Save video
        output_video = self.output_paths['video']
        ensure_dir(output_video)
        save_video(output_frames, output_video, self.config.video.fps)
    
    def _print_summary(self) -> None:
        """Print pipeline completion summary."""
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE - OUTPUT FILES")
        print("=" * 60)
        print(f"  Video:           {self.output_paths['video']}")
        print(f"  Rally CSV:       {self.output_paths['rally_csv']}")
        print(f"  Player CSV:      {self.output_paths['player_csv']}")
        print(f"  Heatmap:         {self.output_paths['heatmap']}")
        print(f"  Court Frame:     {self.output_paths['court_frame']}")
        if self.shot_data:
            print(f"  Shots CSV:       {self.output_paths['shots_csv']}")


def run_main_pipeline(config_or_args, output_video: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to run the full pipeline.
    
    Args:
        config_or_args: Either a PipelineConfig or argparse-like namespace/SimpleNamespace
        output_video: Optional output video path override
        
    Returns:
        Dict of output file paths
    """
    from types import SimpleNamespace
    from .config import config_from_args, PipelineConfig as ConfigClass
    
    # Handle both PipelineConfig and legacy argparse/SimpleNamespace
    if isinstance(config_or_args, ConfigClass):
        config = config_or_args
        print(f"[Pipeline] Using PipelineConfig directly")
    else:
        # It's an args-like object (SimpleNamespace or argparse.Namespace)
        print(f"[Pipeline] Converting {type(config_or_args).__name__} to PipelineConfig...")
        config = config_from_args(config_or_args)
        print(f"[Pipeline] Conversion complete. Input path: {config.video.input_path}")
    
    pipeline = PadelAnalysisPipeline(config)
    return pipeline.run(output_video)