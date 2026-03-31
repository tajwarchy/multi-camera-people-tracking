import cv2
import json
import numpy as np
import torchreid
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
from src.config_loader import load_yaml


def load_crops_and_labels(directory: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load all crops from a directory.
    Filename format: gid_{gid:04d}_cam_{cam_id}_frame_{frame_idx:06d}.png
    Returns: (crops, gid_labels, cam_ids)
    """
    crops, labels, cam_ids = [], [], []
    for p in sorted(Path(directory).glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        parts   = p.stem.split("_")
        gid     = int(parts[1])
        cam_id  = parts[3]
        crops.append(img_rgb)
        labels.append(gid)
        cam_ids.append(cam_id)
    return crops, labels, cam_ids


def extract_embeddings(extractor, crops: List[np.ndarray],
                       batch_size: int = 32) -> np.ndarray:
    """Extract L2-normalized embeddings in batches."""
    all_embs = []
    for i in tqdm(range(0, len(crops), batch_size), desc="  Extracting"):
        batch = crops[i:i + batch_size]
        feats = extractor(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1e-6, norms)
        all_embs.append(feats / norms)
    return np.vstack(all_embs)


def compute_cmc_map(
    query_embs:   np.ndarray,  # (Q, D)
    query_labels: List[int],
    gallery_embs: np.ndarray,  # (G, D)
    gallery_labels: List[int],
    top_k: int = 10
) -> Tuple[Dict[int, float], float]:
    """
    Compute CMC@k and mAP.
    Returns: (cmc_scores dict, mAP float)
    """
    Q = len(query_labels)
    cmc_counts = {k: 0 for k in range(1, top_k + 1)}
    avg_precisions = []

    # Cosine similarity matrix (Q, G)
    sim_matrix = query_embs @ gallery_embs.T  # (Q, G)

    for q_idx in range(Q):
        q_label = query_labels[q_idx]
        sims    = sim_matrix[q_idx]

        # Sort gallery by descending similarity
        sorted_idx = np.argsort(-sims)
        sorted_labels = [gallery_labels[i] for i in sorted_idx]

        # CMC — find first correct match
        for k in range(1, top_k + 1):
            if q_label in sorted_labels[:k]:
                for kk in range(k, top_k + 1):
                    cmc_counts[kk] += 1
                break

        # mAP — precision at each relevant retrieval
        relevant_total = sum(1 for l in gallery_labels if l == q_label)
        if relevant_total == 0:
            continue

        hits, precisions = 0, []
        for rank, label in enumerate(sorted_labels, start=1):
            if label == q_label:
                hits += 1
                precisions.append(hits / rank)

        if precisions:
            avg_precisions.append(np.mean(precisions))

    cmc_scores = {k: cmc_counts[k] / Q for k in range(1, top_k + 1)}
    m_ap = float(np.mean(avg_precisions)) if avg_precisions else 0.0
    return cmc_scores, m_ap


def main():
    cfg = load_yaml("configs/global_config.yaml")

    query_dir   = cfg["eval"]["query_dir"]
    gallery_dir = cfg["eval"]["gallery_dir"]
    out_path    = Path(cfg["eval"]["metrics_output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[Eval] Loading query crops...")
    q_crops, q_labels, q_cams = load_crops_and_labels(query_dir)
    print(f"  Queries : {len(q_crops)} crops | {len(set(q_labels))} unique IDs")

    print("[Eval] Loading gallery crops...")
    g_crops, g_labels, g_cams = load_crops_and_labels(gallery_dir)
    print(f"  Gallery : {len(g_crops)} crops | {len(set(g_labels))} unique IDs")

    if len(q_crops) == 0 or len(g_crops) == 0:
        print("[ERROR] Empty query or gallery — run prepare_reid_eval first.")
        return

    # Load OSNet extractor
    print("\n[Eval] Loading OSNet extractor...")
    extractor = torchreid.utils.FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=cfg["embedder"]["weights"],
        device="cpu"
    )

    print("\n[Eval] Extracting query embeddings...")
    q_embs = extract_embeddings(extractor, q_crops)

    print("\n[Eval] Extracting gallery embeddings...")
    g_embs = extract_embeddings(extractor, g_crops)

    print("\n[Eval] Computing CMC and mAP...")
    cmc_scores, m_ap = compute_cmc_map(q_embs, q_labels, g_embs, g_labels, top_k=5)

    # Results
    print("\n" + "=" * 45)
    print("  ReID Evaluation Results")
    print("=" * 45)
    print(f"  CMC@1  : {cmc_scores[1]*100:.1f}%")
    print(f"  CMC@3  : {cmc_scores[3]*100:.1f}%")
    print(f"  CMC@5  : {cmc_scores[5]*100:.1f}%")
    print(f"  mAP    : {m_ap*100:.1f}%")
    print("=" * 45)

    # Save metrics
    metrics = {
        "CMC@1": round(cmc_scores[1] * 100, 2),
        "CMC@3": round(cmc_scores[3] * 100, 2),
        "CMC@5": round(cmc_scores[5] * 100, 2),
        "mAP"  : round(m_ap * 100, 2),
        "num_queries" : len(q_crops),
        "num_gallery" : len(g_crops),
        "num_identities": len(set(q_labels))
    }
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Metrics saved → {out_path}")

    # Colab fine-tuning trigger check
    if cmc_scores[1] < 0.60:
        print("\n[WARNING] CMC@1 < 60% — consider Phase 8B Colab fine-tuning.")
    else:
        print("\n[OK] CMC@1 >= 60% — pretrained weights are sufficient.")

    # Bar chart
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bars = ["CMC@1", "CMC@3", "CMC@5"]
    vals = [cmc_scores[1]*100, cmc_scores[3]*100, cmc_scores[5]*100]
    axes[0].bar(bars, vals, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[0].set_ylim(0, 100)
    axes[0].set_title("CMC Scores")
    axes[0].set_ylabel("Accuracy (%)")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    axes[1].bar(["mAP"], [m_ap*100], color=["#9C27B0"])
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Mean Average Precision")
    axes[1].set_ylabel("mAP (%)")
    axes[1].text(0, m_ap*100 + 1, f"{m_ap*100:.1f}%", ha="center", fontsize=10)

    plt.suptitle("ReID Evaluation — OSNet x1_0 (Pretrained)", fontsize=12)
    plt.tight_layout()
    chart_path = "outputs/reid_metrics_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Eval] Chart saved → {chart_path}")


if __name__ == "__main__":
    main()