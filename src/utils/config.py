from pathlib import Path
from copy import deepcopy
import yaml

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst

def _find_configs_dir() -> Path:
    for cand in (Path("../configs"), Path("configs")):
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not find 'configs' folder.")

def load_config(override_name: str | None = None, base_name: str = "base") -> dict:
    cfg_dir = _find_configs_dir()
    base_path = cfg_dir / f"{base_name}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_path}")
    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if override_name is None:
        return base_cfg

    override_path = cfg_dir / f"{override_name}.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"Missing override config: {override_path}")
    with override_path.open("r", encoding="utf-8") as f:
        override_cfg = yaml.safe_load(f) or {}

    merged = deepcopy(base_cfg)
    return _deep_update(merged, override_cfg)
