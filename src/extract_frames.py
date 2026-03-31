import cv2
import os
import yaml
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, cam_id: str, max_frames: int = None) -> int:
    """Extract frames from a video file into numbered PNGs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[{cam_id}] Video info: {total_frames} frames | {fps:.1f} FPS | {width}x{height}")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    frame_idx = 0
    saved = 0

    with tqdm(total=total_frames, desc=f"Extracting {cam_id}") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out_file = output_path / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_file), frame)
            saved += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"[{cam_id}] Saved {saved} frames to {output_path}")
    return saved


def main():
    # Edit these paths to match your downloaded dataset
    cameras = {
        "cam01": "data/raw/cam1.avi",
        "cam02": "data/raw/cam2.avi",
        "cam03": "data/raw/cam3.avi",
    }

    for cam_id, video_path in cameras.items():
        if not Path(video_path).exists():
            print(f"[SKIP] {cam_id}: file not found at {video_path}")
            continue
        output_dir = f"data/cameras/{cam_id}/frames"
        extract_frames(video_path, output_dir, cam_id, max_frames=500)

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()