import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any


def calibrate_court(video_path: str, court_number: int, json_path: str = "court_info/court_information.json"):
    """
    Interactive tool to calibrate exclusion zones and court calibration points.
    
    Instructions:
    1. Press 'L' to start defining LEFT exclusion zone
    2. Click points around the left glass area (clockwise)
    3. Press 'L' again to finish left zone
    4. Press 'R' to start defining RIGHT exclusion zone
    5. Click points around the right glass area (clockwise)
    6. Press 'R' again to finish right zone
    7. Press 'P' to define CALIBRATION POINTS (for homography)
       - Click points on the court where you know real-world coordinates
       - After clicking, you'll enter meter coordinates for each point
    8. Press 'Z' to undo last point
    9. Press 'S' to SAVE to JSON file
    10. Press 'Q' to quit
    
    Args:
        video_path: Path to the video file to calibrate
        court_number: Court number/ID to save in JSON
        json_path: Path to the JSON file for storing court info
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading video")
        return
    
    original_frame = frame.copy()
    
    # State
    left_zone: List[Tuple[int, int]] = []
    right_zone: List[Tuple[int, int]] = []
    calibration_points: List[Dict[str, float]] = []  # {"x_px": x, "y_px": y, "x_m": xm, "y_m": ym}
    current_mode = None  # 'left', 'right', 'points', or None
    current_points: List[Tuple[int, int]] = []
    
    # Load existing data if available
    existing_data = load_court_json(json_path)
    court_key = str(court_number)
    if court_key in existing_data.get("courts", {}):
        court_data = existing_data["courts"][court_key]
        zones = court_data.get("exclusion_zones", {})
        
        # Parse existing exclusion zones
        if "LEFT_EXCLUSION_ZONE" in zones:
            left_zone = parse_zone_points(zones["LEFT_EXCLUSION_ZONE"])
        if "RIGHT_EXCLUSION_ZONE" in zones:
            right_zone = parse_zone_points(zones["RIGHT_EXCLUSION_ZONE"])
        if "calibration_points" in court_data:
            calibration_points = court_data["calibration_points"]
        
        print(f"Loaded existing data for court {court_number}")
    
    def draw_state():
        """Redraw frame with current zones and calibration points."""
        display = original_frame.copy()
        
        # Draw completed exclusion zones
        if left_zone:
            pts = np.array(left_zone, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
            cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            cv2.putText(display, "LEFT", (left_zone[0][0]+10, left_zone[0][1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if right_zone:
            pts = np.array(right_zone, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
            cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            cv2.putText(display, "RIGHT", (right_zone[0][0]+10, right_zone[0][1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw calibration points
        for i, pt in enumerate(calibration_points):
            x, y = int(pt["x_px"]), int(pt["y_px"])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
            label = f"P{i+1}: ({pt['x_m']:.1f}, {pt['y_m']:.1f})m"
            cv2.putText(display, label, (x+12, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw current points being defined
        if current_points:
            color = (0, 255, 255) if current_mode != 'points' else (255, 0, 255)
            for i, pt in enumerate(current_points):
                cv2.circle(display, pt, 5, color, -1)
                cv2.putText(display, str(i+1), (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if len(current_points) > 1 and current_mode != 'points':
                for i in range(len(current_points)-1):
                    cv2.line(display, current_points[i], current_points[i+1], color, 2)
        
        # Instructions
        mode_text = f"Mode: {current_mode.upper() if current_mode else 'NONE'} | Court: {court_number}"
        cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, "L=Left zone, R=Right zone, P=Calib points, Z=Undo", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "S=Save to JSON, Q=Quit", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Stats
        stats = f"Left: {len(left_zone)} pts | Right: {len(right_zone)} pts | Calib: {len(calibration_points)} pts"
        cv2.putText(display, stats, (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return display
    
    def click_event(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN and current_mode:
            current_points.append((x, y))
            print(f"  Point {len(current_points)}: ({x}, {y})")
            cv2.imshow("Court Calibration", draw_state())
    
    cv2.imshow("Court Calibration", draw_state())
    cv2.setMouseCallback("Court Calibration", click_event)
    
    print("\n" + "=" * 60)
    print(f"COURT CALIBRATION - Court {court_number}")
    print("=" * 60)
    print("Press L to define LEFT exclusion zone")
    print("Press R to define RIGHT exclusion zone")
    print("Press P to define CALIBRATION POINTS (for homography)")
    print("Click to add points, press the same key again to finish")
    print("Press Z to undo last point")
    print("Press S to SAVE to JSON file")
    print("Press Q to quit")
    print("=" * 60 + "\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('l'):
            if current_mode == 'left':
                # Finish left zone
                if len(current_points) >= 3:
                    left_zone = current_points.copy()
                    print(f"✓ Left zone set with {len(left_zone)} points")
                current_mode = None
                current_points = []
            else:
                # Start left zone
                current_mode = 'left'
                current_points = []
                print("→ Defining LEFT zone - click points clockwise, press L to finish")
            cv2.imshow("Court Calibration", draw_state())
        
        elif key == ord('r'):
            if current_mode == 'right':
                # Finish right zone
                if len(current_points) >= 3:
                    right_zone = current_points.copy()
                    print(f"✓ Right zone set with {len(right_zone)} points")
                current_mode = None
                current_points = []
            else:
                # Start right zone
                current_mode = 'right'
                current_points = []
                print("→ Defining RIGHT zone - click points clockwise, press R to finish")
            cv2.imshow("Court Calibration", draw_state())
        
        elif key == ord('p'):
            if current_mode == 'points':
                # Finish calibration points - ask for meter coordinates
                if len(current_points) >= 4:
                    print(f"\n→ Now enter real-world coordinates (meters) for each clicked point:")
                    print("  Standard padel court: 20m x 10m (length x width)")
                    print("  Origin (0,0) is typically at one corner")
                    
                    new_calib_points = []
                    for i, (px, py) in enumerate(current_points):
                        while True:
                            try:
                                coords = input(f"  Point {i+1} at pixel ({px}, {py}) - Enter x_m, y_m (e.g., '3.0, 5.0'): ")
                                parts = coords.replace(" ", "").split(",")
                                x_m = float(parts[0])
                                y_m = float(parts[1])
                                new_calib_points.append({
                                    "x_px": float(px),
                                    "y_px": float(py),
                                    "x_m": x_m,
                                    "y_m": y_m
                                })
                                break
                            except (ValueError, IndexError):
                                print("    Invalid format. Use: x_m, y_m (e.g., '3.0, 5.0')")
                    
                    calibration_points = new_calib_points
                    print(f"✓ Calibration points set: {len(calibration_points)} points")
                else:
                    print("⚠ Need at least 4 calibration points for homography")
                
                current_mode = None
                current_points = []
            else:
                # Start calibration points
                current_mode = 'points'
                current_points = []
                print("→ Click calibration points on court lines/intersections (need at least 4)")
                print("  After clicking, press P again to enter meter coordinates")
            cv2.imshow("Court Calibration", draw_state())
        
        elif key == ord('z'):
            # Undo last point
            if current_points:
                current_points.pop()
                print("  ↶ Removed last point")
                cv2.imshow("Court Calibration", draw_state())
            elif current_mode is None:
                # Undo last calibration point
                if calibration_points:
                    removed = calibration_points.pop()
                    print(f"  ↶ Removed calibration point: ({removed['x_px']}, {removed['y_px']})")
                    cv2.imshow("Court Calibration", draw_state())
        
        elif key == ord('s'):
            # Save to JSON
            save_court_to_json(
                json_path=json_path,
                court_number=court_number,
                left_zone=left_zone,
                right_zone=right_zone,
                calibration_points=calibration_points
            )
            print(f"\n✓ Saved to {json_path}")
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Court {court_number}:")
    print(f"  - Left exclusion zone: {len(left_zone)} points")
    print(f"  - Right exclusion zone: {len(right_zone)} points")
    print(f"  - Calibration points: {len(calibration_points)} points")
    
    if calibration_points:
        print("\nCalibration points:")
        for i, pt in enumerate(calibration_points):
            print(f"  {i+1}. Pixel ({pt['x_px']:.0f}, {pt['y_px']:.0f}) → Meters ({pt['x_m']:.1f}, {pt['y_m']:.1f})")


def parse_zone_points(zone_list: List[str]) -> List[Tuple[int, int]]:
    """Parse zone points from string format like '(x, y)' to tuples."""
    points = []
    for pt_str in zone_list:
        # Handle both "(x, y)" string format and [x, y] list format
        if isinstance(pt_str, str):
            pt_str = pt_str.strip().strip("()")
            parts = pt_str.split(",")
            x = int(parts[0].strip())
            y = int(parts[1].strip())
        elif isinstance(pt_str, (list, tuple)):
            x, y = int(pt_str[0]), int(pt_str[1])
        else:
            continue
        points.append((x, y))
    return points


def format_zone_points(points: List[Tuple[int, int]]) -> List[str]:
    """Format zone points to string format like '(x, y)'."""
    return [f"({x}, {y})" for x, y in points]


def load_court_json(json_path: str) -> Dict[str, Any]:
    """Load court JSON file, return empty structure if not exists."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_path}: {e}")
    return {"courts": {}}


def save_court_to_json(json_path: str, court_number: int, 
                       left_zone: List[Tuple[int, int]],
                       right_zone: List[Tuple[int, int]],
                       calibration_points: List[Dict[str, float]]):
    """Save court calibration data to JSON file."""
    # Load existing data
    data = load_court_json(json_path)
    
    if "courts" not in data:
        data["courts"] = {}
    
    court_key = str(court_number)
    
    # Build court data
    court_data = {
        "exclusion_zones": {}
    }
    
    if left_zone:
        court_data["exclusion_zones"]["LEFT_EXCLUSION_ZONE"] = format_zone_points(left_zone)
    
    if right_zone:
        court_data["exclusion_zones"]["RIGHT_EXCLUSION_ZONE"] = format_zone_points(right_zone)
    
    if calibration_points:
        court_data["calibration_points"] = calibration_points
    
    # Update data
    data["courts"][court_key] = court_data
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved court {court_number} to {json_path}")


def load_court_calibration(json_path: str, court_number: int) -> Dict[str, Any]:
    """
    Load calibration data for a specific court.
    
    Returns:
        Dict with keys:
        - 'left_zone': List of (x, y) tuples
        - 'right_zone': List of (x, y) tuples  
        - 'calibration_points': List of dicts with x_px, y_px, x_m, y_m
        - 'homography': numpy array (3x3) or None
    """
    data = load_court_json(json_path)
    court_key = str(court_number)
    
    result = {
        'left_zone': [],
        'right_zone': [],
        'calibration_points': [],
        'homography': None
    }
    
    if court_key not in data.get("courts", {}):
        print(f"Warning: Court {court_number} not found in {json_path}")
        return result
    
    court_data = data["courts"][court_key]
    zones = court_data.get("exclusion_zones", {})
    
    # Parse exclusion zones
    if "LEFT_EXCLUSION_ZONE" in zones:
        result['left_zone'] = parse_zone_points(zones["LEFT_EXCLUSION_ZONE"])
    
    if "RIGHT_EXCLUSION_ZONE" in zones:
        result['right_zone'] = parse_zone_points(zones["RIGHT_EXCLUSION_ZONE"])
    
    # Parse calibration points
    if "calibration_points" in court_data:
        result['calibration_points'] = court_data["calibration_points"]
        
        # Build homography if we have enough points
        if len(result['calibration_points']) >= 4:
            px_pts = np.array([[pt["x_px"], pt["y_px"]] for pt in result['calibration_points']], dtype=np.float32)
            m_pts = np.array([[pt["x_m"], pt["y_m"]] for pt in result['calibration_points']], dtype=np.float32)
            
            H, _ = cv2.findHomography(px_pts, m_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            result['homography'] = H
    
    return result


# Convenience function to match existing interface
def load_court_config(court_number: int, json_path: str = "court_info/court_information.json") -> Dict[str, Any]:
    """
    Load court configuration including exclusion zones and calibration.
    
    Returns dict with:
    - LEFT_EXCLUSION_ZONE: List of [x, y] points
    - RIGHT_EXCLUSION_ZONE: List of [x, y] points
    - HOMOGRAPHY: numpy array (3x3) or None
    - calibration_points: raw calibration point data
    """
    calib_data = load_court_calibration(json_path, court_number)
    
    # Convert to the format expected by existing code
    return {
        'LEFT_EXCLUSION_ZONE': [list(pt) for pt in calib_data['left_zone']],
        'RIGHT_EXCLUSION_ZONE': [list(pt) for pt in calib_data['right_zone']],
        'COURT_POLYGON': [],  # Deprecated - kept for backwards compatibility
        'HOMOGRAPHY': calib_data['homography'],
        'calibration_points': calib_data['calibration_points']
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Court Calibration Tool")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--court", "-c", type=int, required=True, help="Court number")
    parser.add_argument("--json", "-j", default="court_info/court_information.json", 
                        help="Path to JSON file")
    
    args = parser.parse_args()
    
    calibrate_court(args.video, args.court, args.json)