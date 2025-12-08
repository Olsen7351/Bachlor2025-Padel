import cv2
import numpy as np


def calibrate_exclusion_zones(video_path: str):
    """
    Interactive tool to calibrate tilted exclusion zones.
    
    Instructions:
    1. Press 'L' to start defining LEFT exclusion zone
    2. Click points around the left glass area (clockwise)
    3. Press 'L' again to finish left zone
    4. Press 'R' to start defining RIGHT exclusion zone
    5. Click points around the right glass area (clockwise)
    6. Press 'R' again to finish right zone
    7. Press 'C' to define COURT polygon (optional)
    8. Press 'Z' to undo last point
    9. Press 'Q' to quit and print results
    
    Args:
        video_path: Path to the video file to calibrate
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading video")
        return
    
    original_frame = frame.copy()
    
    # State
    left_zone = []
    right_zone = []
    court_zone = []
    current_mode = None  # 'left', 'right', 'court', or None
    current_points = []
    
    def draw_state():
        """Redraw frame with current zones."""
        display = original_frame.copy()
        
        # Draw completed zones
        if left_zone:
            pts = np.array(left_zone, dtype=np.int32)
            cv2.fillPoly(display, [pts], (0, 0, 180))
            cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            cv2.putText(display, "LEFT", (left_zone[0][0]+10, left_zone[0][1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if right_zone:
            pts = np.array(right_zone, dtype=np.int32)
            cv2.fillPoly(display, [pts], (0, 0, 180))
            cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            cv2.putText(display, "RIGHT", (right_zone[0][0]+10, right_zone[0][1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if court_zone:
            pts = np.array(court_zone, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, "COURT", (court_zone[0][0]+10, court_zone[0][1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current points being defined
        if current_points:
            for i, pt in enumerate(current_points):
                cv2.circle(display, pt, 5, (0, 255, 255), -1)
                cv2.putText(display, str(i+1), (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if len(current_points) > 1:
                for i in range(len(current_points)-1):
                    cv2.line(display, current_points[i], current_points[i+1], (0, 255, 255), 2)
        
        # Instructions
        mode_text = f"Mode: {current_mode.upper() if current_mode else 'NONE'}"
        cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, "L=Left zone, R=Right zone, C=Court, Z=Undo, Q=Quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Click to add points, press key again to finish zone", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display
    
    def click_event(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN and current_mode:
            current_points.append((x, y))
            print(f"  Point {len(current_points)}: ({x}, {y})")
            cv2.imshow("Calibration", draw_state())
    
    cv2.imshow("Calibration", draw_state())
    cv2.setMouseCallback("Calibration", click_event)
    
    print("\n" + "=" * 60)
    print("EXCLUSION ZONE CALIBRATION")
    print("=" * 60)
    print("Press L to define LEFT exclusion zone")
    print("Press R to define RIGHT exclusion zone")
    print("Press C to define COURT polygon (optional)")
    print("Click to add points, press the same key again to finish")
    print("Press Z to undo last point")
    print("Press Q to quit and print results")
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
            cv2.imshow("Calibration", draw_state())
        
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
            cv2.imshow("Calibration", draw_state())
        
        elif key == ord('c'):
            if current_mode == 'court':
                # Finish court
                if len(current_points) >= 3:
                    court_zone = current_points.copy()
                    print(f"✓ Court zone set with {len(court_zone)} points")
                current_mode = None
                current_points = []
            else:
                # Start court
                current_mode = 'court'
                current_points = []
                print("→ Defining COURT polygon - click 4 corners clockwise, press C to finish")
            cv2.imshow("Calibration", draw_state())
        
        elif key == ord('z'):
            # Undo last point
            if current_points:
                current_points.pop()
                print("  ↶ Removed last point")
                cv2.imshow("Calibration", draw_state())
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Print results
    print("\n" + "=" * 60)
    print("COPY THESE VALUES TO YOUR court_info.json:")
    print("=" * 60)
    
    if left_zone:
        print(f"\n\"LEFT_EXCLUSION_ZONE\": [")
        for i, pt in enumerate(left_zone):
            comma = "," if i < len(left_zone) - 1 else ""
            print(f"    {list(pt)}{comma}")
        print("],")
    
    if right_zone:
        print(f"\n\"RIGHT_EXCLUSION_ZONE\": [")
        for i, pt in enumerate(right_zone):
            comma = "," if i < len(right_zone) - 1 else ""
            print(f"    {list(pt)}{comma}")
        print("],")
    
    if court_zone:
        print(f"\n\"COURT_POLYGON\": [")
        for i, pt in enumerate(court_zone):
            comma = "," if i < len(court_zone) - 1 else ""
            print(f"    {list(pt)}{comma}")
        print("]")
    
    print("\n" + "=" * 60)