import cv2
import numpy as np
import torch
import torchreid
from pathlib import Path


class Embedder:
    def __init__(self, cfg: dict):
        self.device = cfg["device"]
        e_cfg = cfg["embedder"]
        self.input_h, self.input_w = e_cfg["input_size"]   # [256, 128]
        self.dim = e_cfg["embedding_dim"]
        weights = e_cfg["weights"]

        # torchreid FeatureExtractor handles model build + weight load
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=e_cfg["model"],
            model_path=weights,
            device=self.device if self.device != "mps" else "cpu",
            # torchreid doesn't support mps natively — use cpu, still fast
        )
        print(f"[Embedder] OSNet loaded | input={self.input_h}x{self.input_w} | dim={self.dim}")

    def _crop_and_preprocess(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop bbox from frame, resize to OSNet input size."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.input_w, self.input_h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def extract(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized embeddings for all tracks in a frame.
        Args:
            frame : BGR frame
            tracks: (M, 5) — [x1, y1, x2, y2, track_id]
        Returns:
            embeddings: (M, embedding_dim) float32, L2-normalized
                        Row i corresponds to tracks[i]
                        Zero vector if crop is invalid
        """
        M = len(tracks)
        embeddings = np.zeros((M, self.dim), dtype=np.float32)

        if M == 0:
            return embeddings

        crops = []
        valid_idx = []

        for i, t in enumerate(tracks):
            crop = self._crop_and_preprocess(frame, t[:4])
            if crop is not None:
                crops.append(crop)
                valid_idx.append(i)

        if not crops:
            return embeddings

        # FeatureExtractor accepts list of numpy HxWxC RGB images
        feats = self.extractor(crops)   # returns torch.Tensor (len(crops), dim)
        feats = feats.cpu().numpy()     # (len(crops), dim)

        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-6, norms)
        feats = feats / norms

        for out_i, src_i in enumerate(valid_idx):
            embeddings[src_i] = feats[out_i]

        return embeddings