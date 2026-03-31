import numpy as np
from pathlib import Path


def parse_epfl_calibration(calib_path: str) -> dict:
    """
    Parse EPFL calibration file and extract ground plane homography per camera.
    Convention: H * x_image = x_topview
    Returns: { "cam0": np.ndarray(3,3), "cam1": ..., ... }
    """
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    homographies = {}
    lines = path.read_text().splitlines()

    i = 0
    current_cam = None

    while i < len(lines):
        line = lines[i].strip()

        # Detect camera block
        if line.startswith("# Camera"):
            current_cam = f"cam{line.split()[-1]}"
            i += 1
            continue

        # Detect ground plane homography block
        if line == "# Ground plane homography" and current_cam is not None:
            rows = []
            for j in range(1, 4):
                if i + j < len(lines):
                    vals = lines[i + j].strip().split()
                    if len(vals) == 3:
                        rows.append([float(v) for v in vals])
            if len(rows) == 3:
                homographies[current_cam] = np.array(rows, dtype=np.float64)
            i += 4
            continue

        i += 1

    print(f"[Calibration] Parsed homographies for: {list(homographies.keys())}")
    return homographies


def save_homographies(homographies: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cam_id, H in homographies.items():
        out_path = out / f"{cam_id}_H.npy"
        np.save(str(out_path), H)
        print(f"[Calibration] Saved {cam_id} homography → {out_path}")


if __name__ == "__main__":
    from src.config_loader import load_yaml
    cfg = load_yaml("configs/global_config.yaml")

    homographies = parse_epfl_calibration("data/floor_plan/calibration-4p.txt")
    save_homographies(homographies, cfg["paths"]["homography_dir"])

    # Print each matrix for inspection
    for cam_id, H in homographies.items():
        print(f"\n[{cam_id}] Ground Plane Homography:")
        print(H)