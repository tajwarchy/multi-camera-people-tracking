import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from src.config_loader import load_camera_config, load_yaml
from src.bev_projector import BEVProjector
from src.bev_animator import BEVAnimator
from src.pipeline_single_cam import annotate_frame, get_color, load_frames


GLOBAL_CFG = "configs/global_config.yaml"
CAM_CFGS   = ["configs/cam01.yaml", "configs/cam02.yaml", "configs/cam03.yaml"]
CAM_IDS    = ["cam01", "cam02", "cam03"]


def load_enriched_stores(global_cfg: dict) -> Dict[str, List[dict]]:
    emb_dir = Path(global_cfg["paths"]["embedding_store_dir"])
    stores = {}
    for cam_id in CAM_IDS:
        p = emb_dir / f"{cam_id}_enriched.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Enriched store not found: {p}. Run run_association first.")
        with open(p, "rb") as f:
            stores[cam_id] = pickle.load(f)
        print(f"[Loader] {cam_id}: {len(stores[cam_id])} enriched records")
    return stores


def build_composite_frame(
    cam_frames: Dict[str, np.ndarray],
    bev_frame: np.ndarray,
    target_w: int = 800
) -> np.ndarray:
    """
    Build composite: [cam01 | cam02] on top, [cam03 | BEV] on bottom.
    All panels resized to uniform dimensions.
    """
    panel_w = target_w // 2
    panel_h = int(panel_w * 0.75)  # 4:3 aspect

    def resize(img):
        return cv2.resize(img, (panel_w, panel_h))

    top_row = np.hstack([
        resize(cam_frames.get("cam01", np.zeros((panel_h, panel_w, 3), dtype=np.uint8))),
        resize(cam_frames.get("cam02", np.zeros((panel_h, panel_w, 3), dtype=np.uint8)))
    ])
    bot_row = np.hstack([
        resize(cam_frames.get("cam03", np.zeros((panel_h, panel_w, 3), dtype=np.uint8))),
        resize(bev_frame)
    ])
    composite = np.vstack([top_row, bot_row])

    # Label each panel
    labels = [
        ("CAM01", (10, 25)),
        ("CAM02", (panel_w + 10, 25)),
        ("CAM03", (10, panel_h + 25)),
        ("BEV MAP", (panel_w + 10, panel_h + 25)),
    ]
    for text, (tx, ty) in labels:
        cv2.putText(composite, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return composite


def main():
    global_cfg = load_yaml(GLOBAL_CFG)
    cfg        = load_camera_config(GLOBAL_CFG, CAM_CFGS[0])

    # Load enriched records
    stores = load_enriched_stores(global_cfg)

    # Init projector and animator
    projector = BEVProjector(cfg)
    animator  = BEVAnimator(cfg, projector)

    # Load frame paths per camera
    frame_paths = {}
    for cam_cfg_path in CAM_CFGS:
        cam_cfg = load_yaml(cam_cfg_path)
        cam_id  = cam_cfg["cam_id"]
        frame_paths[cam_id] = load_frames(
            cam_cfg["source"],
            cam_cfg["start_frame"],
            cam_cfg["end_frame"]
        )

    # Video writers
    fps    = global_cfg["output"]["fps"]
    fourcc = cv2.VideoWriter_fourcc(*global_cfg["output"]["fourcc"])

    bev_dir = Path(global_cfg["output"]["bev_dir"])
    bev_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = Path(global_cfg["output"]["composite_dir"])
    comp_dir.mkdir(parents=True, exist_ok=True)

    bev_writer  = cv2.VideoWriter(
        str(bev_dir / "bev_animation.mp4"), fourcc, fps,
        (global_cfg["bev"]["canvas_width"], global_cfg["bev"]["canvas_height"])
    )

    # Composite size: 2 panels wide × 2 panels tall, each panel 400×300
    comp_writer = cv2.VideoWriter(
        str(comp_dir / "composite.mp4"), fourcc, fps, (800, 600)
    )

    max_frames = min(len(v) for v in stores.values())
    print(f"\n[Main] Rendering {max_frames} frames...")

    for frame_idx in tqdm(range(max_frames), desc="[Render]"):

        # Get enriched record per camera for this frame
        frame_records = {
            cam_id: stores[cam_id][frame_idx]
            for cam_id in CAM_IDS
        }

        # Read raw frames and annotate with global IDs
        cam_frames = {}
        for cam_id in CAM_IDS:
            fpath  = frame_paths[cam_id][frame_idx]
            frame  = cv2.imread(fpath)
            record = frame_records[cam_id]
            if frame is not None:
                annotated = annotate_frame(
                    frame,
                    record["tracks"],
                    cam_id,
                    global_ids=record.get("global_ids")
                )
                cam_frames[cam_id] = annotated
            else:
                cam_frames[cam_id] = np.zeros((288, 360, 3), dtype=np.uint8)

        # Build BEV frame
        bev_frame = animator.update(frame_records)

        # Build composite
        composite = build_composite_frame(cam_frames, bev_frame, target_w=800)

        bev_writer.write(bev_frame)
        comp_writer.write(composite)

    bev_writer.release()
    comp_writer.release()

    print(f"\n[Done] BEV animation  → {bev_dir}/bev_animation.mp4")
    print(f"[Done] Composite video → {comp_dir}/composite.mp4")


if __name__ == "__main__":
    main()