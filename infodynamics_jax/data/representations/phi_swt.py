from __future__ import annotations
from typing import Dict, Tuple

import jax.numpy as jnp


def phi_swt(images: jnp.ndarray, use_luma: bool = True) -> Tuple[jnp.ndarray, Dict]:
    """Placeholder SWT representation. Implement in project-specific code."""
    # Keep as identity for now to avoid dependencies.
    if use_luma:
        # simple luma projection
        img = images[..., :3]
        lum = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
        return lum.reshape(images.shape[0], -1), {"kind": "swt", "note": "placeholder (luma)"}
    return images.reshape(images.shape[0], -1), {"kind": "swt", "note": "placeholder (identity)"}
