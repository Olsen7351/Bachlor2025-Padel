import cv2, sys
video = sys.argv[1] if len(sys.argv) > 1 else "padel_match.mp4"
frame_index = int(sys.argv[2]) if len(sys.argv) > 2 else 200
cap = cv2.VideoCapture(video); assert cap.isOpened(), f"Cannot open {video}"
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ok, frame = cap.read(); cap.release(); assert ok, "Could not read frame"
out = "frame_for_clicks.jpg"
cv2.imwrite(out, frame); print("Saved", out)
