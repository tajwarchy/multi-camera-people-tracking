import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


def merge_configs(global_cfg: Dict, cam_cfg: Dict) -> Dict:
    """Merge camera config into global config. Camera values take precedence."""
    merged = global_cfg.copy()
    merged["cam"] = cam_cfg

    # Apply tracker overrides if specified
    overrides = cam_cfg.get("tracker_overrides", {}) or {}
    for k, v in overrides.items():
        if v is not None:
            merged["tracker"][k] = v

    return merged


def load_camera_config(global_config_path: str, cam_config_path: str) -> Dict:
    global_cfg = load_yaml(global_config_path)
    cam_cfg = load_yaml(cam_config_path)
    merged = merge_configs(global_cfg, cam_cfg)
    return merged


def print_config_summary(cfg: Dict) -> None:
    cam_id = cfg["cam"]["cam_id"]
    print("=" * 50)
    print(f"  Config Summary — {cam_id}")
    print("=" * 50)
    print(f"  Device        : {cfg['device']} (fallback: {cfg['fallback_device']})")
    print(f"  Resolution    : {cfg['resolution']}px")
    print(f"  Detector      : {cfg['detector']['model']} | conf={cfg['detector']['confidence']}")
    print(f"  Tracker       : StrongSORT | max_age={cfg['tracker']['max_age']}")
    print(f"  Embedder      : {cfg['embedder']['model']} | dim={cfg['embedder']['embedding_dim']}")
    print(f"  ReID threshold: {cfg['reid']['similarity_threshold']}")
    print(f"  Source        : {cfg['cam']['source']}")
    print(f"  Frames        : {cfg['cam']['start_frame']} → {cfg['cam']['end_frame']}")
    print("=" * 50)


if __name__ == "__main__":
    # Validation test — run this to confirm all configs load cleanly
    cameras = ["cam01", "cam02", "cam03"]
    for cam in cameras:
        try:
            cfg = load_camera_config(
                global_config_path="configs/global_config.yaml",
                cam_config_path=f"configs/{cam}.yaml"
            )
            print_config_summary(cfg)
            print(f"  [OK] {cam} config loaded successfully\n")
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"  [ERROR] {cam}: {e}\n")