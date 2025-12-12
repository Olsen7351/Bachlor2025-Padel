# Padel Match Analysis Pipeline

A comprehensive deep learning pipeline for analyzing padel matches, including player tracking, ball detection, rally analysis, shot classification, and heatmap generation.

## Folder Structure

```
deep_learning/
├── court_info/
│   └── court_information.json       # Court calibration configurations
├── input_videos/                    # Input videos for processing
├── output_videos/                   # Generated outputs (not committed)
├── architectures/                   # Model architectures
│   ├── __init__.py
│   └── shot_classifier.py           # Shot classification model architecture
├── models/                          # Weights (not committed)
│   ├── best_model.pth               # Shot classification weights
│   ├── TrackNet_best.pt             # Ball tracking model
│   ├── yolov8n-pose.pt              # YOLO pose estimation
│   └── yolov8s.pt                   # YOLO object detection
├── processors/                      # Processing logic
│   ├── __init__.py
│   ├── shot_processor.py            # Shot classification processor
│   └── export.py                    # Data export utilities
├── standalone/                      # Standalone mode runners
│   ├── __init__.py
│   └── modes.py                     # Individual pipeline modes
├── trackers/                        # Tracking implementations
│   ├── __init__.py
│   ├── ball_tracker.py              # Ball detection and tracking (TrackNet)
│   ├── player_tracker.py            # Player detection and tracking (YOLO)
│   ├── rally_tracker.py             # Rally detection logic
│   └── simple_tracker.py            # Simple IOU-based tracker
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── calibration_utils.py         # Court calibration utilities
│   ├── config_utils.py              # Configuration management
│   ├── heatmap_utils.py             # Heatmap generation
│   ├── inference_utils.py           # Inference helpers
│   └── video_utils.py               # Video processing utilities
├── __init__.py                      # Package exports
├── config.py                        # Configuration dataclasses
├── inference.py                     # CLI entry point
└── pipeline.py                      # Main pipeline orchestrator
```

## Architecture

The pipeline follows SOLID principles:

- **config.py**: Configuration dataclasses (`PipelineConfig`, `VideoConfig`, etc.)
- **pipeline.py**: `PadelAnalysisPipeline` orchestrator class
- **inference.py**: CLI entry point and argument parsing
- **processors/**: Shot classification and data export logic
- **standalone/**: Independent runners for each pipeline mode
- **trackers/**: Ball, player, and rally tracking implementations
- **models/**: ML model architectures

## Setup

### Model Downloads

1. **Ultralytics Models** (automatically downloaded on first run):
   - `yolov8s.pt` - Player detection
   - `yolov8n-pose.pt` - Pose estimation

2. **Custom Models** (download manually):
   - `TrackNet_best.pt` - Ball tracking model 
     - [Download Link](https://drive.google.com/drive/folders/19-P64C1NkzF5lj2qP5MJSk4eR9WaS6qH)
     - Credit: [padel_analytics](https://github.com/Joao-M-Silva/padel_analytics)
   - `best_model.pth` - Shot classification model
     - [Download Link](https://drive.google.com/file/d/1FIb5WcDzeRdmi_qcgA1ZNLLI-xms6MJ7/view?usp=sharing)

Place all model weights in the `models/` directory.

## Usage

**Note**: You may need to provide additional args than showcased examples below. Refer to help command below.

**Important**: Run commands from the parent directory of `deep_learning/` (e.g., `services/`):

```bash
cd services  # or wherever deep_learning/ is located
```

### Help Command
```bash
python -m deep_learning.inference --help
```

### Full Pipeline
```bash
python -m deep_learning.inference --mode full --input_video ./deep_learning/input_videos/video.mp4 --court_number 9
```

### Shot Classification Only
```bash
python -m deep_learning.inference --mode shots --input_video ./deep_learning/input_videos/video2_trimmed.mp4 --shot_model ./deep_learning/models/best_model.pth
```

### Player Tracking Only
```bash
python -m deep_learning.inference --mode player_track --video ./deep_learning/input_videos/video.mp4
```

### Rally Detection
```bash
python -m deep_learning.inference --mode rally --ball_detections_pkl ./path/to/detections.pkl
```

### Court Calibration
```bash
python -m deep_learning.inference --mode calibrate --video ./deep_learning/input_videos/video.mp4 --court_number 9
```

### Heatmap Generation
```bash
python -m deep_learning.inference --mode heatmap --csv player_positions.csv --court_img court_frame.png
```

## Programmatic Usage

You can also use the pipeline programmatically:

```python
from deep_learning.config import PipelineConfig
from deep_learning.pipeline import run_main_pipeline

# Create configuration
config = PipelineConfig()
config.video.input_path = "path/to/video.mp4"
config.video.fps = 30.0
config.court.court_number = 9

# Run pipeline
output_paths = run_main_pipeline(config)
print(f"Output video: {output_paths['video']}")
```

Or use individual components:

```python
from deep_learning.trackers import PlayerTracker, BallTrackerTrackNet
from deep_learning.processors import ShotClassificationProcessor

# Use trackers independently
player_tracker = PlayerTracker(model_path="models/yolov8s.pt")
detections = player_tracker.detect_frames(video_frames)
```

## Key Features

- **Ball Tracking**: TrackNet-based ball detection with trajectory filtering
- **Player Tracking**: Multi-object tracking with ByteTrack integration
- **Rally Detection**: Automatic rally segmentation with velocity analysis
- **Shot Classification**: R(2+1)D + LSTM model for shot type recognition
- **Court Calibration**: Interactive tool for perspective transformation
- **Heatmap Generation**: Player position heatmaps with court overlay

## Output Files

All outputs are saved to `output_videos/{video_name}/`:

| File | Description |
|------|-------------|
| `{name}_output.avi` | Annotated video with all overlays |
| `{name}_player_positions.csv` | Player positions per frame |
| `{name}_rallies.csv` | Rally start/end frames and duration |
| `{name}_shots.csv` | Shot classifications with confidence |
| `{name}_heatmap.png` | Combined player heatmap |
| `{name}_court_frame.png` | First frame for reference |

## Configuration Options

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | Pipeline mode: `full`, `shots`, `player_track`, `rally`, `heatmap`, `calibrate` |
| `--input_video` | - | Input video path |
| `--court_number` | `9` | Court number for calibration lookup |
| `--fps` | `30.0` | Video frame rate |
| `--enable_shot_classification` | `True` | Enable/disable shot classification |
| `--use_stubs` | `False` | Use cached detections for development |

### Court Configuration

Court calibration and exclusion zones are stored in `court_info/court_information.json`. Use calibration mode to set up new courts:

```bash
python -m deep_learning.inference --mode calibrate --video video.mp4 --court_number 9
```

This opens an interactive window to:
1. Define court boundaries (4 corners)
2. Set exclusion zones (glass panels)
3. Save calibration to JSON

## Integration with Backend

The pipeline integrates with the backend via `MLService`:

```python
from ml_service import MLService

service = MLService()
result = await service.run_analysis(
    video_path=Path("video.mp4"),
    court_number=9,
    fps=30.0
)

# Access results
print(result.total_rallies)
print(result.player_stats)
print(result.heatmaps)
```