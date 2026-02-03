# Differential expression operations for single-cell data

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import delnx as dx
from typing import Literal,  TypeAlias

Method: TypeAlias = Literal[
    "t-test",
    "t-test_overestim_var",
    "wilcoxon",
    "logreg"
]

def run_de_scanpy(
    adata: ad.AnnData, groupby: str, method: Method | None = "t-test_overestim_var"
) -> pd.DataFrame:
    """Run differential expression using scanpy's rank_genes_groups function.

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
    de_results = pd.concat(
        [
            sc.get.rank_genes_groups_df(adata, group).assign(group=group)
            for group in adata.obs[groupby].unique()
        ],
        ignore_index=True,
    )
    # Drop scores from de_results
    de_results = de_results.drop(columns=["scores"], errors="ignore")
    # Unify column names
    de_results = de_results.rename(
        columns={
            "names": "feature",
            "logfoldchanges": "logFC",
            "pvals_adj": "padj",
            "pvals": "pval",
        }
    )
    return de_results

def run_de_delnx(
    adata: ad.AnnData, group_key: str = "group", layer: str | None = None, method: str = "negbinom"
) -> pd.DataFrame:
    """Run differential expression using delnx's function

    Parameters:
    ----------
    adata : AnnData
        Input AnnData object.
    group_key : str, default="group"
        Key in adata.obs that identifies the groups or conditions to compare.
    layer : str or None, default=None
        Layer in adata to use for pseudobulk aggregation. If None, uses adata.X.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the differential expression results.
    """
    # Run DE analysis 
    de_results = []
    for group in adata.obs[group_key].unique():
        # Create comparison column for the current group vs all others
        adata.obs["comparison"] = (adata.obs[group_key] == group).astype(int)
        # Run DE analysis using delnx
        print("Running DE for group:", group)
        result = dx.tl.de(
                adata,
                condition_key="comparison",
                mode="1_vs_1",
                reference=("rest", group),
                method=method,
                layer=layer
        )
        # Add group information to the result
        result["group"] = group
        # Collect results
        de_results.append(result)
    # Concatenate results from all groups into a single DataFrame
    de_results_df = pd.concat(de_results, ignore_index=True)
    # Only keep relevant columns i.e
    de_results_df = de_results_df[["feature", "log2fc", "coef", "pval", "padj", "group"]]
    # Rename columns for consistency
    de_results_df = de_results_df.rename(
        columns={
            "log2fc": "logFC",
            "coef": "coef",
            "pval": "pval",
            "padj": "padj",
        }
    )
    return de_results_df


def get_de_genes(
    df: pd.DataFrame,
    group_key: str = "group",
    effect_key: str = "logFC",
    pval_key: str = "pval",
    feature_key: str = "feature",
    effect_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_n: int | None = None,
    return_labeled_df: bool = False,
) -> dict[str, dict[str, list]] | tuple[dict[str, dict[str, list]], pd.DataFrame]:
    """Analyze differential expression data: label significant genes and extract lists per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing differential expression results.
    group_key : str, default="group"
        Column indicating the group or condition.
    effect_key : str, default="logFC"
        Column with effect size values (e.g., log fold change, coefficient).
    pval_key : str, default="pval"
        Column with p-values.
    feature_key : str, default="feature"
        Column containing gene names.
    effect_thresh : float, default=1.0
        Threshold for absolute effect size.
    pval_thresh : float, default=0.01
        Threshold for significance.
    top_n : int or None, default=None
        Number of top genes to select per direction. If None, return all.
    return_labeled_df : bool, default=False
        If True, also return the labeled DataFrame.

    Returns
    -------
    dict or tuple
        If return_labeled_df is False:
            Dictionary of the form:
            {
                "group1": {"up": [...], "down": [...]},
                "group2": {"up": [...], "down": [...]},
                ...
            }
        If return_labeled_df is True:
            Tuple of (gene_lists_dict, labeled_dataframe)
    """
    # Create a copy to avoid modifying the original dataframe
    df_labeled = df.copy()

    # Add significance labels
    df_labeled["-log10(pval)"] = -np.log10(df_labeled[pval_key])

    # Determine significance based on p-value and effect size thresholds
    sig_pval_mask = df_labeled[pval_key] < pval_thresh
    up_mask = sig_pval_mask & (df_labeled[effect_key] > effect_thresh)
    down_mask = sig_pval_mask & (df_labeled[effect_key] < -effect_thresh)

    df_labeled["significant"] = "NS"
    df_labeled.loc[up_mask, "significant"] = "Up"
    df_labeled.loc[down_mask, "significant"] = "Down"

    # Extract DE genes per group
    result = {}
    grouped = df_labeled.groupby(group_key)

    for group, sub_df in grouped:
        up_df = sub_df[sub_df["significant"] == "Up"]
        down_df = sub_df[sub_df["significant"] == "Down"]

        if top_n is not None:
            up_df = up_df.nlargest(top_n, effect_key) # pyright: ignore[reportArgumentType]
            down_df = down_df.nsmallest(top_n, effect_key) # pyright: ignore[reportArgumentType]

        result[group] = {
            "up": up_df[feature_key].tolist(),
            "down": down_df[feature_key].tolist(),
        }

    if return_labeled_df:
        return result, df_labeled
    else:
        return result
