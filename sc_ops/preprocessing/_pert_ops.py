from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pertpy as pt
from sklearn.metrics.pairwise import cosine_distances
from ._adata_ops import aggregate


def _composite_obs_key(
    obs: pd.DataFrame,
    cols: tuple[str, ...],
    *,
    sep: str,
) -> pd.Series:
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    return obs.loc[:, list(cols)].astype(str).agg(sep.join, axis=1)


def aggregate_expression_long_by_obs(
    adata: ad.AnnData,
    *,
    group_cols: tuple[str, ...] = ("sample", "cell_type", "perturbation"),
    group_col: str = "_group",
    sep: str = "|",
    min_cells_per_group: int = 10,
    normalize: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Aggregate expression per group and return a long dataframe.

    This creates `group_col` on the fly (and restores/drops it afterwards) based on `group_cols`.

    Returns a dataframe with columns:
      - gene
      - expr
      - group_col
      - one column per entry in `group_cols`
    """
    if min_cells_per_group < 1:
        raise ValueError("min_cells_per_group must be >= 1")

    adata_work = adata.copy() if copy else adata

    existed = group_col in adata_work.obs.columns
    old_group = adata_work.obs[group_col].copy() if existed else None

    group_key = _composite_obs_key(adata_work.obs, group_cols, sep=sep)
    adata_work.obs[group_col] = group_key

    counts = adata_work.obs[group_col].value_counts(dropna=False)
    valid_groups = counts.index[counts >= min_cells_per_group]

    adata_filt = adata_work[adata_work.obs[group_col].isin(valid_groups)].copy()

    mat_avg = aggregate(adata_filt, groupby=group_col, normalize=normalize)

    df_avg = (
        mat_avg.T.reset_index(names="index")
        .melt(id_vars="index", var_name=group_col, value_name="expr")
        .rename(columns={"index": "gene"})
    )

    group_meta = (
        adata_filt.obs.loc[:, [group_col, *group_cols]]
        .drop_duplicates(subset=group_col)
        .set_index(group_col)
    )

    return df_avg.join(group_meta, on=group_col)

def compute_perturbation_effects_from_df(
    df: pd.DataFrame,
    gene_col: str = "gene",
    celltype_col: str = "cell_type",
    perturbation_col: str = "perturbation",
    expr_col: str = "expr",
    sample_col: str = "sample",
    control_label: str = "Control",
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute perturbation effect magnitudes using distance metrics.
    
    Parameters
    ----------
    metric : str
        Distance metric to use. Options: "euclidean", "cosine", "edistance"
        - euclidean, cosine: Compare perturbed samples to control mean
        - edistance: Compare perturbed cell distributions to control cell distribution
    """
    
    # Validate inputs
    required = {gene_col, celltype_col, perturbation_col, expr_col, sample_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    valid_metrics = {"euclidean", "cosine", "edistance"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}")
    
    # Metrics that operate on cell-level distributions vs. mean-level
    cell_level_metrics = {"edistance"}
    use_cell_level = metric in cell_level_metrics
    
    # Initialize distance calculator once
    distance_calc = pt.tl.Distance(metric=metric)
    
    results = []
    perturbations = df.loc[df[perturbation_col] != control_label, perturbation_col].dropna().unique()
    
    for cell_type, subdf in df.groupby(celltype_col):
        # Prepare metadata and expression data
        meta = (
            subdf[[sample_col, perturbation_col]]
            .drop_duplicates()
            .set_index(sample_col)
        )
        
        expr = subdf.pivot_table(
            index=sample_col,
            columns=gene_col,
            values=expr_col,
            aggfunc="mean",
        )
        
        # Align indices
        common_samples = expr.index.intersection(meta.index)
        expr = expr.loc[common_samples]
        meta = meta.loc[common_samples]
        
        # Get control data
        ctrl_samples = meta.index[meta[perturbation_col] == control_label]
        if len(ctrl_samples) == 0:
            continue
        
        ctrl_expr = expr.loc[ctrl_samples].to_numpy()
        ctrl_ref = ctrl_expr.mean(axis=0).reshape(1, -1) if not use_cell_level else ctrl_expr
        
        # Compute effects for each perturbation
        for pert in perturbations:
            pert_samples = meta.index[meta[perturbation_col] == pert]
            if len(pert_samples) == 0:
                continue
            
            pert_expr = expr.loc[pert_samples].to_numpy()
            dist = distance_calc(pert_expr, ctrl_ref)
            
            results.append(
                {
                    celltype_col: cell_type,
                    perturbation_col: pert,
                    "n_samples": len(pert_samples),
                    "perturbation_effect_magnitude": dist,
                }
            )
    
    if not results:
        raise ValueError("No perturbation effect magnitudes could be computed.")
    
    return pd.DataFrame(results)

def compute_celltype_specific_perturbation_correlations(
    df: pd.DataFrame,
    gene_col: str = "gene",
    celltype_col: str = "cell_type",
    perturbation_col: str = "perturbation",
    expr_col: str = "expr",
    sample_col: str = "sample",
    control_label: str = "Control",
    min_celltypes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each cell type and perturbation, compute the difference vector:
        mean(perturbation samples) - mean(control samples)

    Then, within each perturbation, correlate the difference vectors across
    cell types and compute, for each cell type, its average correlation to
    all other cell types (excluding self-correlation).

    Returns
    -------
    summary_df : pd.DataFrame
        One row per perturbation x cell_type with:
        - perturbation
        - cell_type
        - n_cell_types
        - perturbation_effect_sharedness

    diff_df : pd.DataFrame
        Wide dataframe with one row per (cell_type, perturbation)
        and genes as columns containing the difference vector.
    """
    required = {gene_col, celltype_col, perturbation_col, expr_col, sample_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    diff_rows = []

    perturbations = (
        df.loc[df[perturbation_col] != control_label, perturbation_col]
        .dropna()
        .unique()
    )

    for cell_type, subdf in df.groupby(celltype_col):
        meta = (
            subdf[[sample_col, perturbation_col]]
            .drop_duplicates()
            .set_index(sample_col)
        )

        expr = subdf.pivot_table(
            index=sample_col,
            columns=gene_col,
            values=expr_col,
            aggfunc="mean",
        )

        common_samples = expr.index.intersection(meta.index)
        expr = expr.loc[common_samples]
        meta = meta.loc[common_samples]

        ctrl_samples = meta.index[meta[perturbation_col] == control_label]
        if len(ctrl_samples) == 0:
            continue

        ctrl_mean = expr.loc[ctrl_samples].mean(axis=0)

        for pert in perturbations:
            pert_samples = meta.index[meta[perturbation_col] == pert]
            if len(pert_samples) == 0:
                continue

            pert_mean = expr.loc[pert_samples].mean(axis=0)
            diff = pert_mean - ctrl_mean

            row = {
                celltype_col: cell_type,
                perturbation_col: pert,
                "n_control_samples": len(ctrl_samples),
                "n_perturbation_samples": len(pert_samples),
            }
            row.update(diff.to_dict())
            diff_rows.append(row)

    if not diff_rows:
        raise ValueError("No difference vectors could be computed.")

    diff_df = pd.DataFrame(diff_rows)

    metadata_cols = {
        celltype_col,
        perturbation_col,
        "n_control_samples",
        "n_perturbation_samples",
    }
    gene_cols = [c for c in diff_df.columns if c not in metadata_cols]

    summary_rows = []

    for pert, subdf in diff_df.groupby(perturbation_col):
        if subdf.shape[0] < min_celltypes:
            continue

        subdf = subdf.reset_index(drop=True)
        mat = subdf[gene_cols].to_numpy()

        # correlation between cell-type difference vectors
        corr = np.corrcoef(mat)

        for i, cell_type in enumerate(subdf[celltype_col]):
            row_corrs = np.delete(corr[i], i)  # remove self-correlation
            row_corrs = row_corrs[~np.isnan(row_corrs)]

            mean_corr = np.nan if len(row_corrs) == 0 else float(row_corrs.mean())

            summary_rows.append({
                perturbation_col: pert,
                celltype_col: cell_type,
                "n_cell_types": subdf.shape[0],
                "perturbation_effect_sharedness": mean_corr,
            })

    if not summary_rows:
        raise ValueError("No cell type-specific perturbation correlations could be computed.")

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, diff_df


def compute_perturbation_screen_results(
    adata: ad.AnnData,
    *,
    sep: str = "|",
    group_col: str = "_group",
    min_cells_per_group: int = 10,
    normalize: str | None = None,
    gene_col: str = "gene",
    celltype_col: str = "cell_type",
    perturbation_col: str = "perturbation",
    expr_col: str = "expr",
    sample_col: str = "sample",
    control_label: str = "Control",
    metric: str = "euclidean",
    min_celltypes: int = 2,
    copy: bool = True,
) -> dict[str, pd.DataFrame]:
    """End-to-end wrapper: AnnData -> response amplitude + cell-type specificity."""
    group_cols = (sample_col, celltype_col, perturbation_col)

    df_long = aggregate_expression_long_by_obs(
        adata,
        group_cols=group_cols,
        group_col=group_col,
        sep=sep,
        min_cells_per_group=min_cells_per_group,
        normalize=normalize,
        copy=copy,
    )

    amp_df = compute_perturbation_effects_from_df(
        df_long,
        gene_col=gene_col,
        celltype_col=celltype_col,
        perturbation_col=perturbation_col,
        expr_col=expr_col,
        sample_col=sample_col,
        control_label=control_label,
        metric=metric,
    )

    summary_df, diff_df = compute_celltype_specific_perturbation_correlations(
        df_long,
        gene_col=gene_col,
        celltype_col=celltype_col,
        perturbation_col=perturbation_col,
        expr_col=expr_col,
        sample_col=sample_col,
        control_label=control_label,
        min_celltypes=min_celltypes,
    )

    perturbation_effects_df = amp_df.merge(summary_df, on=[perturbation_col, celltype_col], how="outer")
    out: dict[str, pd.DataFrame] = {
        "perturbation_effects": perturbation_effects_df,
        "difference_vectors": diff_df,
        "agg_expr": df_long,
    }

    return out