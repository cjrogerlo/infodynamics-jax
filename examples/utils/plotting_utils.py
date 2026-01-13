"""
Plotting utilities for evaluation metrics.
"""
import jax.numpy as jnp
from typing import Dict, Optional


def compute_metrics(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    y_std: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics.
    
    Args:
        y_true: True values (N,)
        y_pred: Predicted means (N,)
        y_std: Predicted standard deviations (N,) [optional]
    
    Returns:
        Dict with keys: rmse, mae, r2, [nlpd, coverage] (if y_std provided)
    """
    y_true = jnp.asarray(y_true).flatten()
    y_pred = jnp.asarray(y_pred).flatten()
    
    # RMSE
    rmse = float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)))
    
    # MAE
    mae = float(jnp.mean(jnp.abs(y_true - y_pred)))
    
    # R²
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }
    
    # NLPD and coverage if std provided
    if y_std is not None:
        y_std = jnp.asarray(y_std).flatten()
        y_var = jnp.maximum(y_std ** 2, 1e-12)
        
        # NLPD: negative log predictive density
        nlpd = float(
            0.5 * jnp.mean(
                jnp.log(2 * jnp.pi * y_var) + (y_true - y_pred) ** 2 / y_var
            )
        )
        metrics['nlpd'] = nlpd
        
        # Coverage: fraction of points within 2 std
        within_2std = jnp.abs(y_true - y_pred) <= 2 * y_std
        coverage = float(100.0 * jnp.mean(within_2std))
        metrics['coverage'] = coverage
    
    return metrics
