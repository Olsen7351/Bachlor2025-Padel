# Padel Match Analysis Pipeline

A comprehensive deep learning pipeline for analyzing padel matches, including player tracking, ball detection, rally analysis, shot classification, and heatmap generation.

## Folder Structure

```
deep_learning/
├── court_info/
│   └── court_information.json      # Court calibration configurations
├── models/                          # Model files (not committed - see setup)
│   ├── best_model.pth              # Shot classification model
│   ├── TrackNet_best.pt            # Ball tracking model
│   ├── yolov8n-pose.pt             # YOLO pose estimation
│   └── yolov8s.pt                  # YOLO object detection
├── output_videos/                   # Generated outputs (not committed)
├── trackers/
│   ├── ball_tracker.py             # Ball detection and tracking
│   ├── player_tracker.py           # Player detection and tracking
│   └── rally_tracker.py            # Rally detection logic
├── utils/
│   ├── calibration_utils.py        # Court calibration utilities
│   ├── config_utils.py             # Configuration management
│   └── video_utils.py              # Video processing utilities
└── inference.py                     # Main pipeline script
```

## Setup

### Model Downloads

1. **Ultralytics Models** (automatically downloaded on first run):
   - `yolov8s.pt` - Player detection
   - `yolov8n-pose.pt` - Pose estimation

2. **Custom Models** (download from [https://drive.google.com/drive/folders/19-P64C1NkzF5lj2qP5MJSk4eR9WaS6qH] and [https://drive.google.com/file/d/1FIb5WcDzeRdmi_qcgA1ZNLLI-xms6MJ7/view?usp=sharing]):
   - `TrackNet_best.pt` - Ball tracking model (Credit to [https://github.com/Joao-M-Silva/padel_analytics])
   - `best_model.pth` - Shot classification model

Place all models in the `models/` directory.

## Usage

The pipeline supports multiple modes:

### Full Pipeline
```bash
python inference.py --mode full --input_video path/to/video.mp4 --court_number 9
```

### Shot Classification Only
```bash
python inference.py --mode shots --input path/to/video.mp4
```

### Player Tracking Only
```bash
python inference.py --mode player_track --video path/to/video.mp4
```

### Rally Detection
```bash
python inference.py --mode rally --ball_detections_pkl path/to/detections.pkl
```

### Court Calibration
```bash
python inference.py --mode calibrate --video path/to/video.mp4 --court_number 9
```

### Heatmap Generation
```bash
python inference.py --mode heatmap --csv player_positions.csv --court_img court_frame.png
```

## Key Features

- **Ball Tracking**: TrackNet-based ball detection and tracking
- **Player Tracking**: Multi-object tracking with ByteTrack
- **Rally Detection**: Automatic rally segmentation with velocity analysis
- **Shot Classification**: Deep learning-based shot type recognition
- **Court Calibration**: Interactive tool for perspective transformation
- **Heatmap Generation**: Player position heatmaps on court overlay

## Output Files

All outputs are saved to `output_videos/` directory:
- Annotated videos (`.mp4`, `.avi`)
- Player positions (`.csv`)
- Rally information (`.csv`)
- Shot classifications (`.csv`)
- Heatmaps (`.png`)
- Court frames (`.png`)

## Configuration

Court calibration and settings are stored in `court_info/court_information.json`. Use calibration mode to set up new courts.