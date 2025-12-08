import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_points(csv_path, inplay_only, min_conf, players):
    df = pd.read_csv(csv_path)

    # we ignore x_m/y_m on purpose: use raw pixels
    if "x_px" not in df.columns or "y_px" not in df.columns:
        raise SystemExit("CSV must contain x_px and y_px columns.")

    if "confidence" in df.columns and min_conf is not None:
        df = df[df["confidence"] >= min_conf]
    if inplay_only and "in_play" in df.columns:
        df = df[df["in_play"] == 1]
    if players:
        keep = set(players)
        df = df[df["track_id"].isin(keep)]

    pts = df[["x_px", "y_px", "track_id"]].dropna().to_numpy()
    return pts

def make_hist_on_image(pts_xy, img_w, img_h, bins_x, bins_y):
    """
    pts_xy: N x 2 in pixel coords (x_px, y_px) with origin at top-left.
    """
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]

    # clamp to image bounds just in case
    x = np.clip(x, 0, img_w - 1)
    y = np.clip(y, 0, img_h - 1)

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[bins_x, bins_y],
        range=[[0, img_w], [0, img_h]],
    )
    H = H.T  # (bins_y, bins_x): rows = y, cols = x

    return H, (0, img_w, 0, img_h)

def get_court_blue_cmap():
    # blue → cyan → yellow → red
    return LinearSegmentedColormap.from_list(
        "courtblue",
        [
            (0.0,  "#0050ff"),   # deep blue
            (0.3,  "#00a8ff"),   # light blue
            (0.6,  "#ffff66"),   # soft yellow
            (1.0,  "#ff3300"),   # orange/red
        ]
    )

def gaussian_blur(H, ksize):
    if ksize is None or ksize < 3:
        return H
    try:
        import cv2
    except ImportError:
        return H
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    return cv2.GaussianBlur(H.astype(np.float32), (k, k), 0)

def save_heatmap_on_image(H, extent, img, out_png, title,
                          cmap, heat_alpha=0.7, show_axes=False):
    xmin, xmax, ymin, ymax = extent

    # --- NEW: compute figure size based on image aspect ---
    img_h, img_w = img.shape[0], img.shape[1]
    aspect = img_w / img_h
    base_height = 8
    fig_width = base_height * aspect
    fig, ax = plt.subplots(figsize=(fig_width, base_height))
    # -------------------------------------------------------

    # mask zeros so unused areas don't darken the background
    H_plot = np.where(H > 0.3, H, np.nan)

    # draw background court image
    ax.imshow(
        img,
        origin="upper",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
    )

    # overlay heatmap
    im = ax.imshow(
        H_plot,
        origin="upper",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        cmap=cmap,
        alpha=heat_alpha,
    )

    plt.colorbar(im, ax=ax, label="Counts")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Pixels (x)")
    ax.set_ylabel("Pixels (y)")

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Padel heatmap directly on court image (pixel space only)")
    ap.add_argument("--csv", required=True, help="Tracker CSV (with x_px, y_px)")
    ap.add_argument("--court_img", required=True,
                    help="Court image (ideally a frame from the same camera/video)")
    ap.add_argument("--out_png", default="heatmap_on_image.png", help="Output PNG")
    ap.add_argument("--bins_x", type=int, default=200, help="Number of bins along X")
    ap.add_argument("--bins_y", type=int, default=100, help="Number of bins along Y")
    ap.add_argument("--gauss", type=int, default=9,
                    help="Gaussian blur kernel size (odd int). 0/1 to disable")
    ap.add_argument("--inplay_only", action="store_true",
                    help="Use only frames where in_play==1 (if column exists)")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="Drop detections below this confidence")
    ap.add_argument("--players", default="",
                    help="Comma-separated track IDs to include (blank = all)")
    ap.add_argument("--no_axes", action="store_true",
                    help="Hide axes for a cleaner image")
    ap.add_argument("--heat_alpha", type=float, default=0.7,
                    help="Alpha of heatmap overlay (0–1)")

    args = ap.parse_args()

    players = [int(s) for s in args.players.split(",") if s.strip().isdigit()] if args.players else None

    # load data
    pts = load_points(args.csv, args.inplay_only, args.min_conf, players)
    if pts.size == 0:
        raise SystemExit("No points after filtering. Check CSV/filters.")

    # load court image
    img = plt.imread(args.court_img)
    img_h, img_w = img.shape[0], img.shape[1]

    # build histogram in image coordinates
    H, extent = make_hist_on_image(
        pts[:, :2],
        img_w, img_h,
        args.bins_x, args.bins_y
    )
    Hsmooth = gaussian_blur(H, args.gauss)

    title = "Heatmap on court image"
    if args.inplay_only:
        title += " • in-play only"

    cmap = get_court_blue_cmap()

    save_heatmap_on_image(
        Hsmooth,
        extent,
        img,
        args.out_png,
        title,
        cmap=cmap,
        heat_alpha=args.heat_alpha,
        show_axes=not args.no_axes,
    )

    print(f"Saved {args.out_png}")

if __name__ == "__main__":
    main()
