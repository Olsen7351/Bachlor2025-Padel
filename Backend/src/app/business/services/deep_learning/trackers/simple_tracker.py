from typing import List, Dict
try:
    from ..utils import get_bbox_iou
except ImportError:
    # Fallback for standalone usage
    def get_bbox_iou(bbox1, bbox2):
        """Compute IOU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area


class SimpleTracker:
    """
    Simple IOU-based tracker to maintain player IDs across frames.
    
    Uses greedy matching based on bounding box overlap to associate
    detections across frames.
    """

    def __init__(self, max_missed: int = 30, min_iou: float = 0.1):
        """
        Initialize tracker.
        
        Args:
            max_missed: Maximum frames a track can be missed before deletion
            min_iou: Minimum IOU threshold for matching
        """
        self.tracks: Dict[int, Dict] = {}
        self.next_id: int = 0
        self.max_missed = max_missed
        self.min_iou = min_iou

    def update(self, players: List[Dict]) -> None:
        """
        Update tracks with new detections.
        
        Modifies players in-place to add 'id' field.
        
        Args:
            players: List of player detections with 'bbox_proc' field
        """
        if not self.tracks:
            # Initialize tracks from first detections
            for p in players:
                p['id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': p['bbox_proc'], 'missed': 0}
                self.next_id += 1
            return

        track_ids = list(self.tracks.keys())
        matches = []

        # Compute all IOU scores
        for i, p in enumerate(players):
            for tid in track_ids:
                iou = get_bbox_iou(p['bbox_proc'], self.tracks[tid]['bbox'])
                if iou > self.min_iou:
                    matches.append((iou, tid, i))

        # Greedy matching by IOU score
        matches.sort(key=lambda x: x[0], reverse=True)
        assigned_tracks = set()
        assigned_players = set()

        for iou, tid, idx in matches:
            if tid not in assigned_tracks and idx not in assigned_players:
                players[idx]['id'] = tid
                self.tracks[tid]['bbox'] = players[idx]['bbox_proc']
                self.tracks[tid]['missed'] = 0
                assigned_tracks.add(tid)
                assigned_players.add(idx)

        # Create new tracks for unmatched players
        for i, p in enumerate(players):
            if i not in assigned_players:
                p['id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': p['bbox_proc'], 'missed': 0}
                self.next_id += 1

        # Update missed counts and remove stale tracks
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks."""
        return len(self.tracks)