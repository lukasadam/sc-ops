# Aggregates single cell data by specified grouping

import anndata as ad
import scanpy as sc
import pandas as pd
from _utils import group_by_max, minmax

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
    # Perform aggregation using scanpy's built-in function
    adata_agg = sc.get.aggregated(adata, groupby=groupby, method=method)
    # Convert the aggregated AnnData to a DataFrame
    mat_agg = pd.DataFrame(adata_agg.X, index=adata_agg.obs_names, columns=adata_agg.var_names)
    # Apply min-max scaling per gene
    mat_agg = minmax(mat_agg, axis=0)
    # Order DataFrame by group of max value
    mat_agg = group_by_max(mat_agg)
    return mat_agg