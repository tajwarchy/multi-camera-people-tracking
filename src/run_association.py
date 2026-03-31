import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from src.config_loader import load_camera_config, load_yaml
from src.reid_associator import ReIDAssociator
from src.pipeline_single_cam import annotate_frame, get_color, load_frames


GLOBAL_CFG = "configs/global_config.yaml"
CAM_CFGS   = ["configs/cam01.yaml", "configs/cam02.yaml", "configs/cam03.yaml"]
CAM_IDS    = ["cam01", "cam02", "cam03"]


def load_embedding_stores(cfg: dict, cam_ids: List[str]) -> Dict[str, List[dict]]:
    emb_dir = Path(cfg["paths"]["embedding_store_dir"])
    stores = {}
    for cam_id in cam_ids:
        p = emb_dir / f"{cam_id}_embeddings.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Embedding store not found: {p}. Run pipeline_single_cam first.")
        with open(p, "rb") as f:
            stores[cam_id] = pickle.load(f)
        print(f"[Loader] {cam_id}: {len(stores[cam_id])} frame records loaded")
    return stores


def export_annotated_videos(
    enriched: Dict[str, List[dict]],
    cam_cfgs: List[str],
    global_cfg: dict
) -> None:
    """Re-render per-camera videos with global ID overlays."""
    out_base = Path(global_cfg["output"]["per_camera_dir"])
    fps      = global_cfg["output"]["fps"]
    fourcc   = cv2.VideoWriter_fourcc(*global_cfg["output"]["fourcc"])

    for cam_cfg_path in cam_cfgs:
        cam_cfg  = load_yaml(cam_cfg_path)
        cam_id   = cam_cfg["cam_id"]
        records  = enriched.get(cam_id, [])
        if not records:
            continue

        frame_paths = load_frames(
            cam_cfg["source"],
            cam_cfg["start_frame"],
            cam_cfg["end_frame"]
        )

        first = cv2.imread(frame_paths[0])
        h, w  = first.shape[:2]
        out_dir = out_base / cam_id
        out_dir.mkdir(parents=True, exist_ok=True)
        vid_path = out_dir / f"{cam_id}_globalid.mp4"
        writer   = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))

        for record in tqdm(records, desc=f"[Export {cam_id}]"):
            fid   = record["frame_id"]
            frame = cv2.imread(frame_paths[fid])
            if frame is None:
                continue
            annotated = annotate_frame(
                frame,
                record["tracks"],
                cam_id,
                global_ids=record.get("global_ids")
            )
            writer.write(annotated)

        writer.release()
        print(f"[Export] {cam_id} global ID video → {vid_path}")


def print_association_summary(enriched: Dict[str, List[dict]]) -> None:
    print("\n" + "=" * 55)
    print("  Cross-Camera Association Summary")
    print("=" * 55)
    for cam_id, records in enriched.items():
        all_gids = set()
        for r in records:
            for gid in r.get("global_ids", []):
                if gid != -1:
                    all_gids.add(gid)
        print(f"  {cam_id}: {len(all_gids)} unique global IDs observed")
    print("=" * 55)


def main():
    global_cfg = load_yaml(GLOBAL_CFG)

    # Use cam01 config to build full merged cfg for associator
    cfg = load_camera_config(GLOBAL_CFG, CAM_CFGS[0])

    # Load embedding stores
    stores = load_embedding_stores(global_cfg, CAM_IDS)

    # Run association
    associator = ReIDAssociator(cfg)
    enriched   = associator.run(stores)
    associator.save_global_id_map()

    # Print summary
    print_association_summary(enriched)

    # Save enriched records per camera
    emb_dir = Path(global_cfg["paths"]["embedding_store_dir"])
    for cam_id, records in enriched.items():
        out_path = emb_dir / f"{cam_id}_enriched.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        print(f"[Saved] Enriched records → {out_path}")

    # Re-render videos with global ID overlay
    export_annotated_videos(enriched, CAM_CFGS, global_cfg)


if __name__ == "__main__":
    main()