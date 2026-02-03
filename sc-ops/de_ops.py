# Differential expression operations for single-cell data

import pandas as pd
import scanpy as sc
import anndata as ad

def run_within_de(adata: ad.AnnData, groupby: str, method: str = 't-test_overestim_var') -> pd.DataFrame:
    """Run differential expression within groups in the AnnData object.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        The key in adata.obs to group by.
    method : str
        The differential expression method to use.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the differential expression results.
    """
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)
    # Collect results
    de_results = pd.concat([sc.get.rank_genes_groups_df(adata, group).assign(group=group) for group in adata.obs[groupby].unique()], ignore_index=True)
    # Drop scores from de_results
    de_results = de_results.drop(columns=['scores'], errors='ignore')
    # Unify column names
    de_results = de_results.rename(columns={'names': 'genes', 'logfoldchanges': 'logFC', 'pvals_adj': 'padj', 'pvals': 'pval'})
    return de_results
