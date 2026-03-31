# Quick sanity check — run in python
import pickle
import numpy as np

with open("outputs/embeddings/cam03_embeddings.pkl", "rb") as f:
    records = pickle.load(f)

# Find a frame with tracks
active = [r for r in records if len(r["tracks"]) > 0]
print(f"Frames with tracks: {len(active)}")

sample = active[len(active) // 2]
embs = sample["embeddings"]
print(f"Embedding shape : {embs.shape}")           # should be (M, 512)
print(f"L2 norms        : {np.linalg.norm(embs, axis=1)}")  # should all be ~1.0
print(f"All unit-normed : {np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)}")