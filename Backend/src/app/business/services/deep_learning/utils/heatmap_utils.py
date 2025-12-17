from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .inference_utils import ensure_dir
from matplotlib.patches import Rectangle

def load_heatmap_points(csv_path, inplay_only=False, min_conf=None, players=None):
    df = pd.read_csv(csv_path)
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
    x = np.clip(pts_xy[:, 0], 0, img_w - 1)
    y = np.clip(pts_xy[:, 1], 0, img_h - 1)
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[0, img_w], [0, img_h]])
    H = H.T
    return H, (0, img_w, 0, img_h)


def get_court_blue_cmap():
    return LinearSegmentedColormap.from_list(
        "courtblue",
        [(0.0, "#0050ff"), (0.3, "#00a8ff"), (0.6, "#ffff66"), (1.0, "#ff3300")]
    )


def gaussian_blur_heatmap(H, ksize):
    if ksize is None or ksize < 3:
        return H
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    return cv2.GaussianBlur(H.astype(np.float32), (k, k), 0)


def save_heatmap_on_image(H, extent, img, out_png, title, cmap, heat_alpha=0.7, show_axes=False):
    ensure_dir(out_png)
    xmin, xmax, ymin, ymax = extent
    img_h, img_w = img.shape[0], img.shape[1]
    aspect = img_w / img_h
    base_height = 8
    fig_width = base_height * aspect
    fig, ax = plt.subplots(figsize=(fig_width, base_height))
    H_plot = np.where(H > 0.3, H, np.nan)
    ax.imshow(img, origin="upper", extent=[xmin, xmax, ymin, ymax], aspect="auto")
    im = ax.imshow(H_plot, origin="upper", extent=[xmin, xmax, ymin, ymax],
                   aspect="auto", cmap=cmap, alpha=heat_alpha)
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


def generate_heatmap(csv_path, court_img_path, out_png, bins_x=200, bins_y=100,
                     gauss=9, inplay_only=False, min_conf=None, players=None,
                     heat_alpha=0.7, show_axes=False):
    pts = load_heatmap_points(csv_path, inplay_only, min_conf, players)
    if pts.size == 0:
        print("Warning: No points after filtering for heatmap.")
        return

    img = plt.imread(court_img_path)
    img_h, img_w = img.shape[0], img.shape[1]

    H, extent = make_hist_on_image(pts[:, :2], img_w, img_h, bins_x, bins_y)
    Hsmooth = gaussian_blur_heatmap(H, gauss)

    title = "Player Position Heatmap"
    if inplay_only:
        title += " (in-play only)"

    cmap = get_court_blue_cmap()
    save_heatmap_on_image(Hsmooth, extent, img, out_png, title,
                          cmap=cmap, heat_alpha=heat_alpha, show_axes=show_axes)
    print(f"Saved heatmap: {out_png}")
    
def compute_zone_percents(df, court_w=20.0):
    """
    Returns: percentages array of length 3 for zones [defence, transition, offence]
    """
    sub = df[(df["x_m"] >= 0.0) & (df["x_m"] <= 10.0)].copy()
    d = sub["x_m"].to_numpy()

    if len(sub) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # 3 equal zones over 0..10
    edges = np.array([0.0, 10.0/3.0, 6.0, 10.0], dtype=float)
    # bins: [0,3.33), [3.33,6.0), [6.0,10]
    z1 = np.sum((d >= edges[0]) & (d < edges[1]))
    z2 = np.sum((d >= edges[1]) & (d < edges[2]))
    z3 = np.sum((d >= edges[2]) & (d <= edges[3]))
    counts = np.array([z1, z2, z3], dtype=float)
    return counts / counts.sum() * 100.0
    
def generate_zone_overlay(csv_path, court_img_path, out_png,
                          side="near", players=None, min_conf=None,
                          court_w=20.0, court_h=10.0,
                          alpha=0.22, font_size=18, show_axes=False,
                          title="Zone occupancy"):
    """
    Overlays 3 zones + percentages onto a top-down court image.

    Coordinate system for overlay:
      x axis = court width  (0..court_h)
      y axis = court length (0..court_w)
    """
    ensure_dir(out_png)

    df = pd.read_csv(csv_path)
    for col in ["x_m", "y_m"]:
        if col not in df.columns:
            raise ValueError("CSV must contain x_m and y_m for zone overlay.")

    df = df.dropna(subset=["x_m", "y_m"])

    if min_conf is not None and "confidence" in df.columns:
        df = df[df["confidence"] >= min_conf]
    if players:
        df = df[df["track_id"].isin(players)]

    img = plt.imread(court_img_path)

    percs = compute_zone_percents(df, court_w=court_w)

    extent = [0.0, court_h, 0.0, court_w]  # x=width, y=length
    fig_w = 8
    fig_h = fig_w * (court_w / court_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(img, origin="lower", extent=extent, aspect="auto")

    zone_edges = [0.0, 10.0/3.0, 6.0, 10.0]
    labels = ["Defence", "Transition", "Offence"]

    def draw_half(y0, perc):
        for i in range(3):
            y_start = y0 + zone_edges[i]
            y_end   = y0 + zone_edges[i+1]
            rect = Rectangle((0.0, y_start), court_h, (y_end - y_start),
                             linewidth=2, edgecolor="white",
                             facecolor="white", alpha=alpha)
            ax.add_patch(rect)
            ax.text(court_h/2.0, (y_start + y_end)/2.0,
                    f"{labels[i]}\n{perc[i]:.1f}%",
                    ha="center", va="center",
                    fontsize=font_size, weight="bold", color="black")

    draw_half(0.0, percs)

    ax.set_xlim(0, court_h)
    ax.set_ylim(0, court_w)

    if title:
        ax.set_title(title)

    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("Court width (m)")
        ax.set_ylabel("Court length (m)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved zone overlay: {out_png}")
