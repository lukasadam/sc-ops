from __future__ import annotations

from typing import Optional

import anndata as ad
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

from ._adata_ops import aggregate


def cosine_similarity(
    adata_source: ad.AnnData,
    adata_target: ad.AnnData,
    *,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute cosine similarity between two AnnData objects after aggregation.

    Parameters
    ----------
    adata_source, adata_target
        Input AnnData objects.
    layer
        Layer to aggregate from. If None, uses .X.

    Returns
    -------
    pandas.DataFrame
        Cosine similarity matrix with rows = aggregated groups of adata_source
        and columns = aggregated groups of adata_target.
    """

    mat_source = aggregate(adata_source, layer=layer)
    mat_target = aggregate(adata_target, layer=layer)

    if mat_source.shape[1] != mat_target.shape[1]:
        raise ValueError(
            f"Feature mismatch: mat_source has {mat_source.shape[1]} columns, mat_target has {mat_target.shape[1]}. "
            "Make sure both were aggregated to the same feature space (same genes/order)."
        )

    sim = _cosine_similarity(mat_source.to_numpy(), mat_target.to_numpy())

    return pd.DataFrame(sim, index=mat_source.index, columns=mat_target.index)
