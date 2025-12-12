"""
Data Export Functions.

Handles exporting analysis results to various formats (CSV, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set

from ..utils import ensure_dir, to_meters


def export_player_positions_csv(
    player_detections: List[Dict], 
    output_csv: str, 
    fps: float = 30.0, 
    homography: Optional[np.ndarray] = None, 
    court_w: float = 20.0, 
    court_h: float = 10.0, 
    rally_frames: Optional[Set[int]] = None
) -> str:
    """
    Export player detections to CSV with optional homography for meter conversion.
    
    Args:
        player_detections: List of frame detections, each a dict of {track_id: bbox}
        output_csv: Output file path
        fps: Video frame rate
        homography: Optional homography matrix for pixel->meter conversion
        court_w: Court width in meters
        court_h: Court height in meters
        rally_frames: Optional set of frame indices that are during a rally
        
    Returns:
        Path to the created CSV file
    """
    ensure_dir(output_csv)
    
    rows = []
    for frame_idx, detections in enumerate(player_detections):
        if not detections:
            continue
            
        for track_id, bbox in detections.items():
            # Get foot position (bottom center of bbox)
            x_px = (bbox[0] + bbox[2]) / 2.0
            y_px = bbox[3]  # Bottom of bbox
            
            # Convert to meters if homography available
            x_m, y_m = np.nan, np.nan
            if homography is not None:
                pts = np.array([[x_px, y_px]], dtype=np.float32)
                pts_m = to_meters(homography, pts)
                if pts_m.size > 0 and np.all(np.isfinite(pts_m[0])):
                    x_m, y_m = pts_m[0]
                    # Validate within court bounds
                    if not (0 <= x_m <= court_w and 0 <= y_m <= court_h):
                        x_m, y_m = np.nan, np.nan
            
            # Determine if frame is during a rally
            in_play = 1 if rally_frames and frame_idx in rally_frames else 0
            conf = 1.0
            
            rows.append({
                'frame': frame_idx,
                'track_id': track_id,
                'x_px': x_px,
                'y_px': y_px,
                'x_m': x_m,
                'y_m': y_m,
                'confidence': conf,
                'in_play': in_play
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Exported player positions to: {output_csv}")
    return output_csv


def export_analysis_summary(
    rallies: List,
    shot_events: List[Dict],
    player_stats: Dict[int, Dict[str, int]],
    output_path: str,
    fps: float = 30.0
) -> str:
    """
    Export a summary of the analysis to a text/JSON file.
    
    Args:
        rallies: List of Rally objects
        shot_events: List of shot event dicts
        player_stats: Dict of player shot stats
        output_path: Output file path
        fps: Video frame rate
        
    Returns:
        Path to created file
    """
    ensure_dir(output_path)
    
    summary = {
        'rally_count': len(rallies),
        'total_shots': len(shot_events),
        'player_stats': player_stats,
    }
    
    if rallies:
        durations = [r.duration_seconds(fps) for r in rallies]
        summary['rally_stats'] = {
            'total_time_sec': sum(durations),
            'avg_duration_sec': np.mean(durations),
            'max_duration_sec': max(durations),
            'min_duration_sec': min(durations),
        }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Exported analysis summary to: {output_path}")
    return output_path