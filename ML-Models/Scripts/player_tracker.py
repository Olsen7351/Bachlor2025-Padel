import argparse
import os
import csv
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------- Utilities ----------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_calib_points(s: str):
    """
    Parse --calib like: "x1,y1;x2,y2;x3,y3;x4,y4"
    Order: TL, TR, BR, BL (of playable rectangle)
    """
    pts = []
    for pair in s.split(";"):
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Calibration must have exactly 4 points as x,y pairs.")
    return pts

def build_homography(corner_px, court_w=20.0, court_h=10.0):
    """
    Four-corner homography: TL,TR,BR,BL (pixels) -> (0,0),(W,0),(W,H),(0,H) meters.
    """
    target = np.array(
        [[0.0, 0.0],
         [court_w, 0.0],
         [court_w, court_h],
         [0.0, court_h]], dtype=np.float32
    )
    H, _ = cv2.findHomography(corner_px, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H

def load_calib_csv(path):
    """
    Read multi-point calibration CSV with columns: x_px,y_px,x_m,y_m
    Return homography (pixels->meters) estimated with RANSAC.
    Need at least 4 rows.
    """
    px_pts, m_pts = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            px_pts.append([float(row["x_px"]), float(row["y_px"])])
            m_pts.append([float(row["x_m"]),  float(row["y_m"])])
    px_pts = np.array(px_pts, dtype=np.float32)
    m_pts  = np.array(m_pts,  dtype=np.float32)
    if len(px_pts) < 4:
        raise ValueError("Need at least 4 rows in calib CSV.")
    H, _ = cv2.findHomography(px_pts, m_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H

def to_meters(H, pts_xy):
    """
    Map Nx2 pixel points to meters with homography H.
    """
    if H is None or pts_xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_xy = pts_xy.astype(np.float32)
    pts_xy = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = (H @ pts_xy.T).T
    mapped /= mapped[:, 2:3]
    return mapped[:, :2]

def draw_annotations(frame, xyxy, ids, pts_px, pts_m=None):
    for i, box in enumerate(xyxy.astype(int)):
        x1, y1, x2, y2 = box
        tid = int(ids[i])
        px, py = int(pts_px[i, 0]), int(pts_px[i, 1])

        label = f"ID {tid}"
        if pts_m is not None and pts_m.shape[0] > i and np.all(np.isfinite(pts_m[i])):
            xm, ym = pts_m[i]
            label += f" | {xm:.2f}m, {ym:.2f}m"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 255), 2, cv2.LINE_AA)

        # draw the "feet" point
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Padel player tracker with YOLOv8 + ByteTrack")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_video", default="outputs/padel_annotated.mp4", help="Path to save annotated video")
    parser.add_argument("--out_csv", default="outputs/padel_tracks.csv", help="Path to save tracks CSV")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 weights (e.g., yolov8n/s/m.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference size (e.g., 960 or 1280). Optional.")
    parser.add_argument("--device", default=None, help='Device: "0" for GPU 0, "cpu" for CPU (defaults to auto)')
    parser.add_argument("--calib", default=None,
                        help='Optional 4 image points for corners "x1,y1;x2,y2;x3,y3;x4,y4" (TL,TR,BR,BL)')
    parser.add_argument("--calib_csv", default=None,
                        help="CSV with columns x_px,y_px,x_m,y_m; 4+ rows to fit homography.")
    parser.add_argument("--court_w", type=float, default=20.0, help="Court width in meters (default: 20.0)")
    parser.add_argument("--court_h", type=float, default=10.0, help="Court height in meters (default: 10.0)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config (ByteTrack).")
    args = parser.parse_args()
    
    # Probe video properties
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    cap.release()

    ensure_dir(args.out_video)
    ensure_dir(args.out_csv)

    # Build homography (if any)
    H = None
    if args.calib_csv:
        try:
            H = load_calib_csv(args.calib_csv)
            print(f"[INFO] Loaded homography from {args.calib_csv} (multi-point).")
        except Exception as e:
            print(f"[WARN] Failed to load {args.calib_csv}: {e}")
    elif args.calib:
        try:
            img_pts = parse_calib_points(args.calib)
            H = build_homography(img_pts, court_w=args.court_w, court_h=args.court_h)
            print("[INFO] Loaded homography from 4-corner string.")
        except Exception as e:
            print(f"[WARN] Failed to parse --calib: {e}")

    # Load YOLO
    model = YOLO(args.model)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    rows = []  # for CSV
    frame_idx = -1

    track_kwargs = dict(
        source=args.video,
        stream=True,
        conf=args.conf,
        classes=[0],             # COCO: class 0 = person
        tracker=args.tracker,
        device=args.device
    )
    if args.imgsz:
        track_kwargs["imgsz"] = args.imgsz

    for result in model.track(**track_kwargs):
        frame_idx += 1
        frame = result.orig_img
    
        if result.boxes is None or len(result.boxes) == 0:
            writer.write(frame)
            continue
    
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()                    # (N,4)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],))
        ids   = boxes.id
        if ids is None:
            ids = np.arange(xyxy.shape[0])
        else:
            ids = ids.cpu().numpy().astype(int)
    
        # Feet positions (bottom-center of the box)
        feet_px = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                                   xyxy[:, 3]))

    
        # map feet positions to meters
        feet_m = to_meters(H, feet_px) if H is not None else None
        
        if feet_m is not None:
            xm = feet_m[:, 0]
            ym = feet_m[:, 1]
        
            valid = np.isfinite(xm) & np.isfinite(ym)
            in_court = (
                (xm >= 0.0) & (xm <= args.court_w) &
                (ym >= 0.0) & (ym <= args.court_h)
            )
            keep = valid & in_court
        
            xyxy    = xyxy[keep]
            confs   = confs[keep]
            ids     = ids[keep]
            feet_px = feet_px[keep]
            feet_m  = feet_m[keep]
        
            if xyxy.shape[0] == 0:
                writer.write(frame)
                continue

    
        # --- log to CSV (only in-court detections) ---
        for i in range(xyxy.shape[0]):
            if feet_m is not None and np.all(np.isfinite(feet_m[i])):
                xm, ym = float(feet_m[i, 0]), float(feet_m[i, 1])
            else:
                xm, ym = np.nan, np.nan
        
            rows.append((
                frame_idx, int(ids[i]),
                float(feet_px[i, 0]), float(feet_px[i, 1]),  # pixel position = feet
                xm, ym,
                float(confs[i])
            ))

    
        # --- draw only the filtered boxes ---
        # draw_annotations(frame, xyxy, ids, centroids, centroids_m)
        draw_annotations(frame, xyxy, ids, feet_px, feet_m)
        writer.write(frame)


    writer.release()

    # Save tracks CSV
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x_px", "y_px", "x_m", "y_m", "confidence"])
    df.to_csv(args.out_csv, index=False)

    print("Done!")
    print(f"- Annotated video: {args.out_video}")
    print(f"- Tracks CSV:     {args.out_csv}")
    if H is None:
        print("Tip: pass --calib_csv (recommended) or --calib (4 corners) to map pixels->meters.")

if __name__ == "__main__":
    main()
