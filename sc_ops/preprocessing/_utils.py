"""Utility functions for preprocessing operations."""

import numpy as np
import pandas as pd


def _sum_is_integer(x, atol=1e-6):
    """Check whether a sum can be safely interpreted as integer."""
    return np.isclose(x, np.round(x), atol=atol)


def check_raw_counts(adata, layer=None):
    """
    Ensure that adata.X or adata.layers[layer] contains raw integer counts.
    """
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        mat = adata.layers[layer]
        location = f"adata.layers['{layer}']"
    else:
        mat = adata.X
        location = "adata.X"

    # Handle sparse and dense uniformly
    data_sum = mat.sum()
    if hasattr(data_sum, "item"):
        data_sum = data_sum.item()

    if not _sum_is_integer(data_sum):
        raise ValueError(
            f"Data in {location} must be raw counts (integers) for delnx. "
            f"Detected non-integer sum: {data_sum}"
        )


def group_by_max(expr: pd.DataFrame) -> pd.DataFrame:
    """Order genes by group of max value.

    Parameters:
    ----------
    expr : pd.DataFrame
        A DataFrame where rows are genes and columns are samples.

    Returns:
    -------
    expr: pd.DataFrame
        A DataFrame which is sorted by the group of max value.
    """
    max_group = expr.idxmax(axis=1)
    ordered = []
    for group in expr.columns:
        ordered.extend(expr[max_group == group].index.tolist())
    return expr.loc[ordered]


def minmax_df(df: pd.DataFrame, axis: int) -> pd.DataFrame:
    """Min-max normalize along axis.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame to be normalized.
    axis : int
        Axis along which to normalize (0 for columns, 1 for rows).

    Returns:
    -------
    pd.DataFrame
        Min-max normalized DataFrame.
    """
    mn = df.min(axis=axis)
    mx = df.max(axis=axis)
    denom = (mx - mn).replace(0, np.nan)
    out = (df - mn) / denom
    return out.fillna(0.0)