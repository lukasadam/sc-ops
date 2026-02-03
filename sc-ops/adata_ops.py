# Aggregates single cell data by specified grouping

import anndata as ad
import scanpy as sc
import pandas as pd
from _utils import group_by_max

def aggregate(adata: ad.AnnData, groupby: str, method: str = 'mean') -> pd.DataFrame:
    """
    Aggregate single cell data by specified grouping.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        The key in adata.obs to group by.
    method : str
        The aggregation method to use ('mean', 'sum', etc.).

    Returns:
    -------
    AnnData
        The aggregated annotated data matrix.
    """
    adata_agg = sc.get.aggregated(adata, groupby=groupby, method=method)
    mat_agg = pd.DataFrame(adata_agg.X, index=adata_agg.obs_names, columns=adata_agg.var_names)
    mat_agg = group_by_max(mat_agg)
    return mat_agg