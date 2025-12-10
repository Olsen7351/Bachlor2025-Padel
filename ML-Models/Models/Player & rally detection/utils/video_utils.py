import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=30):
    """
    Saves frames to a video file.
    Args:
        output_video_frames: List of numpy arrays (frames)
        output_video_path: String path for output file
        fps: Frames Per Second (default 30). MATCH THIS TO YOUR INPUT VIDEO!
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()