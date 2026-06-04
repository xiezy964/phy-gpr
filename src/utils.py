import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_output_dirs() -> None:
    cache_root = Path("/tmp") / "codex-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
