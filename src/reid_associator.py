import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


class GlobalIDRegistry:
    """
    Maintains a registry of global IDs and their prototype embeddings.
    Each global ID has a running mean embedding (EMA-updated).
    """
    def __init__(self, ema_alpha: float = 0.9):
        self.ema_alpha = ema_alpha
        self.prototypes: Dict[int, np.ndarray] = {}   # global_id → embedding
        self.next_id = 1

    def mint_new(self, embedding: np.ndarray) -> int:
        gid = self.next_id
        self.prototypes[gid] = embedding.copy()
        self.next_id += 1
        return gid

    def find_best_match(self, embedding: np.ndarray, threshold: float) -> Tuple[int, float]:
        """
        Find the best matching global ID for a given embedding.
        Returns (global_id, similarity) or (-1, 0.0) if no match above threshold.
        """
        best_gid = -1
        best_sim = -1.0

        for gid, proto in self.prototypes.items():
            sim = cosine_similarity(embedding, proto)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_sim >= threshold:
            return best_gid, best_sim
        return -1, best_sim

    def update_prototype(self, gid: int, embedding: np.ndarray) -> None:
        """EMA update of prototype embedding."""
        self.prototypes[gid] = (
            self.ema_alpha * self.prototypes[gid] +
            (1 - self.ema_alpha) * embedding
        )
        # Re-normalize after EMA update
        norm = np.linalg.norm(self.prototypes[gid])
        if norm > 1e-6:
            self.prototypes[gid] /= norm


class ReIDAssociator:
    def __init__(self, cfg: dict):
        self.threshold = cfg["reid"]["similarity_threshold"]
        self.ema_alpha = cfg["reid"]["ema_alpha"]
        self.out_path  = Path(cfg["paths"]["global_id_map"])
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.registry = GlobalIDRegistry(ema_alpha=self.ema_alpha)

        # Maps (cam_id, local_track_id) → global_id
        self.global_id_map: Dict[Tuple[str, int], int] = {}

        print(f"[ReIDAssociator] threshold={self.threshold} | ema_alpha={self.ema_alpha}")

    def process_record(self, record: dict) -> dict:
        """
        Process one frame record from a single camera.
        Assigns global IDs to each track in the record.
        Returns enriched record with 'global_ids' list added.
        """
        cam_id    = record["cam_id"]
        tracks    = record["tracks"]       # (M, 5)
        embeddings = record["embeddings"]  # (M, 512)

        global_ids = []

        for i in range(len(tracks)):
            local_tid = int(tracks[i, 4])
            key = (cam_id, local_tid)
            emb = embeddings[i]

            # Skip zero embeddings (invalid crop)
            if np.linalg.norm(emb) < 1e-6:
                global_ids.append(-1)
                continue

            if key in self.global_id_map:
                # Already seen this (cam, local_id) pair — reuse and update
                gid = self.global_id_map[key]
                self.registry.update_prototype(gid, emb)
            else:
                # New (cam, local_id) — find best match or mint new global ID
                gid, sim = self.registry.find_best_match(emb, self.threshold)
                if gid == -1:
                    gid = self.registry.mint_new(emb)
                else:
                    self.registry.update_prototype(gid, emb)
                self.global_id_map[key] = gid

            global_ids.append(gid)

        enriched = record.copy()
        enriched["global_ids"] = global_ids
        return enriched

    def run(self, all_cam_records: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        """
        Process all cameras frame by frame in chronological order.
        Cameras are interleaved per frame to simulate real-time association.
        Args:
            all_cam_records: { cam_id: [records...] }
        Returns:
            enriched_records: { cam_id: [enriched_records...] }
        """
        # Find max frame count across cameras
        max_frames = max(len(v) for v in all_cam_records.values())
        cam_ids = list(all_cam_records.keys())

        enriched: Dict[str, List[dict]] = {c: [] for c in cam_ids}

        print(f"[ReIDAssociator] Processing {max_frames} frames across {len(cam_ids)} cameras...")

        for frame_idx in range(max_frames):
            for cam_id in cam_ids:
                records = all_cam_records[cam_id]
                if frame_idx >= len(records):
                    continue
                record = records[frame_idx]
                enriched_record = self.process_record(record)
                enriched[cam_id].append(enriched_record)

        print(f"[ReIDAssociator] Total global IDs minted: {self.registry.next_id - 1}")
        return enriched

    def save_global_id_map(self) -> None:
        """Save global_id_map as JSON (keys converted to strings for JSON compat)."""
        serializable = {
            f"{cam_id}|{local_tid}": gid
            for (cam_id, local_tid), gid in self.global_id_map.items()
        }
        with open(self.out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[ReIDAssociator] Global ID map saved → {self.out_path}")