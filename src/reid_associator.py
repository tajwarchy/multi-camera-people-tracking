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
        max_frames = max(len(v) for v in all_cam_records.values())
        cam_ids = list(all_cam_records.keys())
        enriched: Dict[str, List[dict]] = {c: [] for c in cam_ids}

        print(f"[ReIDAssociator] Processing {max_frames} frames across {len(cam_ids)} cameras...")

        for frame_idx in range(max_frames):
            for cam_id in cam_ids:
                records = all_cam_records[cam_id]
                if frame_idx >= len(records):
                    continue
                enriched[cam_id].append(self.process_record(records[frame_idx]))

        # ── Second pass: merge global IDs whose prototypes are similar ──
        print("[ReIDAssociator] Running second-pass prototype merging...")
        merge_map = self._build_merge_map()
        if merge_map:
            print(f"[ReIDAssociator] Merging global IDs: {merge_map}")
            enriched = self._apply_merge(enriched, merge_map)

        print(f"[ReIDAssociator] Total global IDs after merge: "
            f"{len(set(merge_map.get(g, g) for g in range(1, self.registry.next_id)))}")
        
        # Suppress ghost IDs with too few observations
        min_obs = 5
        obs_count: Dict[int, int] = {}
        for cam_id, records in enriched.items():
            for record in records:
                for gid in record.get("global_ids", []):
                    if gid != -1:
                        obs_count[gid] = obs_count.get(gid, 0) + 1

        ghost_ids = {gid for gid, cnt in obs_count.items() if cnt < min_obs}
        if ghost_ids:
            print(f"[ReIDAssociator] Suppressing ghost IDs: {ghost_ids}")
            for cam_id, records in enriched.items():
                for record in records:
                    record["global_ids"] = [
                        -1 if gid in ghost_ids else gid
                        for gid in record.get("global_ids", [])
                    ]
        
        return enriched

    def _build_merge_map(self, merge_threshold: float = 0.75) -> Dict[int, int]:
        """
        Compare all prototype pairs. If similarity exceeds merge_threshold,
        map the higher ID to the lower ID.
        """
        gids = sorted(self.registry.prototypes.keys())
        merge_map = {}

        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                ga, gb = gids[i], gids[j]
                # Resolve already-merged IDs
                ga_final = merge_map.get(ga, ga)
                gb_final = merge_map.get(gb, gb)
                if ga_final == gb_final:
                    continue
                pa = self.registry.prototypes[ga_final]
                pb = self.registry.prototypes[gb_final]
                sim = cosine_similarity(pa, pb)
                if sim >= merge_threshold:
                    # Map higher ID → lower ID
                    keep   = min(ga_final, gb_final)
                    remove = max(ga_final, gb_final)
                    merge_map[remove] = keep
                    print(f"  [Merge] G{remove} → G{keep} (sim={sim:.4f})")

        return merge_map

    def _apply_merge(self, enriched: Dict[str, List[dict]],
                    merge_map: Dict[int, int]) -> Dict[str, List[dict]]:
        """Remap global IDs in all enriched records using merge_map."""
        for cam_id, records in enriched.items():
            for record in records:
                record["global_ids"] = [
                    merge_map.get(gid, gid)
                    for gid in record.get("global_ids", [])
                ]
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