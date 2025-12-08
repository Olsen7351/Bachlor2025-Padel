import cv2, csv

image_path = "frame_for_clicks.jpg"  # change if needed
N_CLICKS = 8                         # set more (e.g., 6–10) for robustness

img = cv2.imread(image_path)
assert img is not None, f"Cannot read {image_path}"
points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < N_CLICKS:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(img, f"{len(points)}", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Click points (ESC to finish)", img)

cv2.imshow("Click points (ESC to finish)", img)
cv2.setMouseCallback("Click points (ESC to finish)", on_mouse)

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC to finish early
        break
    if len(points) >= N_CLICKS:
        break

cv2.destroyAllWindows()

with open("clicked_pixels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x_px","y_px"])
    for x,y in points: w.writerow([x,y])

print("Saved clicked_pixels.csv with", len(points), "points.")
