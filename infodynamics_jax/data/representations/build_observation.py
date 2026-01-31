from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from .phi_convbank import phi_convbank
from .phi_swt import phi_swt


@dataclass
class ObservationCFG:
    kind: str = "pixel"  # "pixel" | "convbank" | "swt"
    use_luma: bool = True
    # Add representation-specific knobs below (kept minimal for now)



def build_observation(cfg: ObservationCFG, images: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Build observation representation Y=Upphi(images).

    Returns:
        Y: observation array
        aux: dict of representation metadata
    """
    if cfg.kind == "pixel":
        return images, {"kind": "pixel"}
    if cfg.kind == "convbank":
        return phi_convbank(images, use_luma=cfg.use_luma)
    if cfg.kind == "swt":
        return phi_swt(images, use_luma=cfg.use_luma)
    raise ValueError(f"Unknown observation kind: {cfg.kind}")
