import argparse
import json
import logging
import os
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str, filename: str = "train.log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(output_dir, filename),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("Install pyyaml or use a JSON config file.") from exc
            data = yaml.safe_load(f)
    return _to_namespace(data)


def get_device(config: SimpleNamespace) -> str:
    requested = getattr(getattr(config, "runtime", SimpleNamespace()), "device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_folds(config: SimpleNamespace):
    folds = getattr(config.training, "folds", [1, 2, 3, 4, 5])
    return [int(f) for f in folds]


def namespace_to_args(config: SimpleNamespace, **extra):
    args = argparse.Namespace()
    for group in vars(config).values():
        if isinstance(group, SimpleNamespace):
            for key, value in vars(group).items():
                setattr(args, key, value)
    for key, value in extra.items():
        setattr(args, key, value)
    return args
