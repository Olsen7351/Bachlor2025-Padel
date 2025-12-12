"""
Padel Match Analysis Pipeline - CLI Entry Point
==================================================

This module provides the command-line interface for the padel analysis pipeline.
All processing logic has been refactored into dedicated modules:

- config.py: Configuration dataclasses
- pipeline.py: Main pipeline orchestrator  
- models/: ML model architectures
- processors/: Shot processing and data export
- trackers/: Ball, player, and rally tracking
- standalone/: Individual mode runners
- utils/: Utility functions

Usage Examples:
---------------
# Full pipeline with shot classification
python -m deep_learning.inference --mode full --input_video video.mp4

# Shot classification only
python -m deep_learning.inference --mode shots --input video.mp4

# Standalone player tracking
python -m deep_learning.inference --mode player_track --video input.mp4

# Rally detection from ball pickle
python -m deep_learning.inference --mode rally --ball_detections_pkl balls.pkl

# Heatmap generation
python -m deep_learning.inference --mode heatmap --csv positions.csv --court_img court.png

# Calibrate a court
python -m deep_learning.inference --mode calibrate --video input.mp4 --court_number 9
"""

import argparse
import sys

from .config import PipelineConfig, config_from_args
from .pipeline import PadelAnalysisPipeline, run_main_pipeline
from .standalone import (
    run_standalone_player_tracker,
    run_rally_detection,
    run_shot_inference,
    run_heatmap_generation
)
from .utils import calibrate_court

# Re-export for backwards compatibility
__all__ = [
    'run_main_pipeline',
    'PipelineConfig',
    'config_from_args',
    'PadelAnalysisPipeline',
]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Padel Match Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        default="full",
        choices=["full", "shots", "player_track", "rally", "heatmap", "calibrate"],
        help="Pipeline mode (default: full)"
    )
    
    # === Common Options ===
    common = parser.add_argument_group("Common Options")
    common.add_argument("--output_dir", default="output_videos",
                        help="Output directory for generated files")
    common.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate")
    
    # === Video Input/Output ===
    video = parser.add_argument_group("Video I/O")
    video.add_argument("--input_video", default="input_videos/video2_trimmed.mp4",
                       help="Input video for full pipeline")
    video.add_argument("--input", type=str, help="Input video (alias)")
    video.add_argument("--video", help="Input video (alias)")
    video.add_argument("--output_video", default=None, help="Output video path")
    video.add_argument("--output", type=str, help="Output video (alias)")
    video.add_argument("--out_video", help="Output video (alias)")
    video.add_argument("--out_csv", help="Output CSV path")
    
    # === Model Paths ===
    models = parser.add_argument_group("Model Paths")
    models.add_argument("--tracknet_model", default="models/TrackNet_best.pt",
                        help="TrackNet model path")
    models.add_argument("--tracknet", default="models/TrackNet_best.pt",
                        help="TrackNet model (alias)")
    models.add_argument("--player_model", default="models/yolov8s",
                        help="Player detection model")
    models.add_argument("--model", default="models/yolov8s.pt",
                        help="Model path (alias)")
    models.add_argument("--shot_model", default="models/best_model.pth",
                        help="Shot classification model")
    models.add_argument("--yolo_pose", default="models/yolov8n-pose.pt",
                        help="YOLO pose model")
    
    # === Court Configuration ===
    court = parser.add_argument_group("Court Configuration")
    court.add_argument("--court_number", type=int, default=9,
                       help="Court number in config JSON")
    court.add_argument("--court_json", default="court_info/court_information.json",
                       help="Court configuration JSON")
    court.add_argument("--court_w", type=float, default=20.0,
                       help="Court width in meters")
    court.add_argument("--court_h", type=float, default=10.0,
                       help="Court height in meters")
    court.add_argument("--calib", default=None,
                       help="4-corner calibration: x1,y1;x2,y2;x3,y3;x4,y4")
    court.add_argument("--calib_csv", default=None,
                       help="CSV with calibration points")
    
    # === Ball Detection ===
    ball = parser.add_argument_group("Ball Detection")
    ball.add_argument("--ball_threshold", type=float, default=0.5,
                      help="Ball detection threshold")
    ball.add_argument("--threshold", type=float, default=0.5,
                      help="Detection threshold (alias)")
    ball.add_argument("--min_heatmap_conf", type=float, default=0.5,
                      help="Minimum heatmap confidence")
    ball.add_argument("--trail_length", type=int, default=10,
                      help="Ball trail length in frames")
    ball.add_argument("--draw_exclusion_zones", action="store_true", default=True,
                      help="Draw exclusion zones on output")
    
    # === Rally Detection ===
    rally = parser.add_argument_group("Rally Detection")
    rally.add_argument("--min_velocity", type=float, default=3.0,
                       help="Minimum ball velocity for movement")
    rally.add_argument("--serve_velocity", type=float, default=6.0,
                       help="Minimum velocity to detect serve")
    rally.add_argument("--base_gap_during_rally", type=int, default=40,
                       help="Base max frames without ball during rally")
    rally.add_argument("--base_rally_end_gap", type=int, default=70,
                       help="Base frames without ball = rally ended")
    rally.add_argument("--max_gap_extension", type=int, default=60,
                       help="Maximum additional gap frames")
    rally.add_argument("--min_rally_frames", type=int, default=45,
                       help="Minimum rally duration in frames")
    rally.add_argument("--min_rally_distance", type=int, default=80,
                       help="Minimum ball travel distance")
    rally.add_argument("--rally_csv", default=None,
                       help="Rally output CSV path")
    rally.add_argument("--ball_detections_pkl", default=None,
                       help="Ball detections pickle for rally mode")
    rally.add_argument("--frame_height", type=int, default=None,
                       help="Frame height for rally detection")
    
    # === Shot Classification ===
    shot = parser.add_argument_group("Shot Classification")
    shot.add_argument("--enable_shot_classification", action="store_true", default=True,
                      help="Enable shot classification in full pipeline")
    shot.add_argument("--no_shot_classification", dest="enable_shot_classification", 
                      action="store_false", help="Disable shot classification")
    shot.add_argument("--confidence_threshold", type=float, default=None,
                      help="Shot classification confidence threshold")
    shot.add_argument("--window_size", type=int, default=32,
                      help="Classification window size")
    shot.add_argument("--stride", type=int, default=8,
                      help="Classification stride")
    shot.add_argument("--yolo_resolution", type=int, nargs=2, default=[640, 360],
                      help="YOLO inference resolution")
    shot.add_argument("--classifier_resolution", type=int, default=128,
                      help="Classifier input resolution")
    shot.add_argument("--processing_resolution", type=int, nargs=2, default=[1280, 720],
                      help="Processing resolution")
    shot.add_argument("--lookahead", type=int, default=16,
                      help="Lookahead frames")
    
    # === Heatmap Options ===
    heatmap = parser.add_argument_group("Heatmap Options")
    heatmap.add_argument("--csv", default=None,
                         help="CSV input for heatmap mode")
    heatmap.add_argument("--court_img", default=None,
                         help="Court image for heatmap")
    heatmap.add_argument("--heatmap_output", default=None,
                         help="Heatmap output path")
    heatmap.add_argument("--heatmap_bins_x", type=int, default=200,
                         help="Heatmap X bins")
    heatmap.add_argument("--heatmap_bins_y", type=int, default=100,
                         help="Heatmap Y bins")
    heatmap.add_argument("--heatmap_gauss", type=int, default=9,
                         help="Gaussian blur sigma")
    heatmap.add_argument("--heatmap_inplay_only", action="store_true",
                         help="Only use in-play positions")
    heatmap.add_argument("--heatmap_min_conf", type=float, default=None,
                         help="Minimum confidence filter")
    heatmap.add_argument("--heatmap_players", default="",
                         help="Player IDs to include (comma-separated)")
    heatmap.add_argument("--heatmap_alpha", type=float, default=0.7,
                         help="Heatmap overlay alpha")
    heatmap.add_argument("--heatmap_show_axes", action="store_true",
                         help="Show axes on heatmap")
    
    # === Player Tracking ===
    player = parser.add_argument_group("Player Tracking")
    player.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    player.add_argument("--imgsz", type=int, default=None,
                        help="Inference image size")
    player.add_argument("--device", default=None,
                        help="Compute device")
    player.add_argument("--tracker", default="bytetrack.yaml",
                        help="Tracker configuration")
    player.add_argument("--auto_heatmap", action="store_true", default=True,
                        help="Auto-generate heatmap after tracking")
    player.add_argument("--no_auto_heatmap", dest="auto_heatmap", action="store_false",
                        help="Disable auto heatmap")
    
    # === Stub/Cache Options ===
    stubs = parser.add_argument_group("Development Options")
    stubs.add_argument("--use_stubs", action="store_true", default=False,
                       help="Use cached detections")
    stubs.add_argument("--no_stubs", dest="use_stubs", action="store_false",
                       help="Don't use cached detections")
    stubs.add_argument("--player_stub", default="tracker_stubs/player_detections.pkl",
                       help="Player detections cache")
    stubs.add_argument("--ball_stub", default="tracker_stubs/ball_detections.pkl",
                       help="Ball detections cache")
    stubs.add_argument("--player_csv", default=None,
                       help="Player CSV output path")
    stubs.add_argument("--generate_heatmap", action="store_true", default=True,
                       help="Generate heatmap (default: True)")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Convert args to config
    config = config_from_args(args)
    
    # Route to appropriate mode
    if args.mode == "calibrate":
        video_path = args.video or args.input_video or args.input
        print(f"Starting calibration for court {config.court.court_number}")
        print(f"Video: {video_path}")
        print(f"JSON output: {config.court.court_json}")
        calibrate_court(video_path, config.court.court_number, config.court.court_json)
    
    elif args.mode == "heatmap":
        run_heatmap_generation(config, args)
    
    elif args.mode == "player_track":
        if not config.video.input_path:
            print("Error: --video required for player_track mode")
            sys.exit(1)
        run_standalone_player_tracker(config, args)
    
    elif args.mode == "rally":
        run_rally_detection(config, args)
    
    elif args.mode == "shots":
        if not config.video.input_path:
            print("Error: --input required for shots mode")
            sys.exit(1)
        run_shot_inference(config, args)
    
    else:  # mode == "full"
        run_main_pipeline(config, args.output_video)


if __name__ == "__main__":
    main()