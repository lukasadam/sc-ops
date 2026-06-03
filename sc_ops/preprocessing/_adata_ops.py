# Common Anndata operations for single-cell data
from __future__ import annotations
from typing import Literal, Optional, TypeAlias, Union

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from ._utils import group_by_max, minmax_df

DimReduceMethod: TypeAlias = Literal["pca", "umap"]
Metric: TypeAlias = Literal["mean", "sum", "detection"]
Norm: TypeAlias = Optional[Literal["minmax", "none"]]


def merge_adatas(
    adatas: list[ad.AnnData],
    datasets_names: list[str],
    join: str = "inner",
    batch_key: str = "batch",
) -> ad.AnnData:
    """Merge multiple AnnData objects into a single AnnData with batch annotation.

    Parameters
    ----------
    adatas
        List of AnnData objects to merge.
    datasets_names
        List of names corresponding to each AnnData in `adatas`. These will be used as batch labels.
    join
        How to handle genes (var_names) across adatas. Options:
        - "inner": Keep only genes present in all adatas (intersection).
        - "outer": Keep all genes present in any adata (union).
        - "left": Keep genes present in the first adata only.
    batch_key
        The name of the column in `adata.obs` to store batch labels.

    Returns
    -------
    AnnData
        A single AnnData object containing all merged datasets with batch annotation.

    """
    # Now we can concatenate them into a single AnnData object
    adata = adatas[0].concatenate(
        adatas[1:], join=join, batch_key=batch_key, batch_categories=datasets_names
    )
    return adata


def clean_genes(
    adata: ad.AnnData,
    min_cells: int = 3,
    filter_mito: bool = True,
    filter_ribo: bool = True,
    filter_protein_coding: bool = True,
) -> ad.AnnData:
    """Clean gene features from an AnnData object.

    Performs a sequence of filters to remove lowly expressed and unwanted genes:
    1. Remove genes detected in fewer than ``min_cells`` cells (uses ``sc.pp.filter_genes``).
    2. Optionally remove mitochondrial genes (``var_names`` starting with ``'MT-'``).
    3. Optionally remove ribosomal genes (``var_names`` starting with ``'RPS'`` or ``'RPL'``).
    4. Optionally keep only genes annotated as protein-coding in ``adata.var['gene_biotype']``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix to be filtered.
    min_cells : int, optional
        Minimum number of cells a gene must be expressed in to be retained. Default is 3.
    filter_mito : bool, optional
        If True, remove mitochondrial genes whose names start with ``'MT-'``. Default is True.
    filter_ribo : bool, optional
        If True, remove ribosomal genes whose names start with ``'RPS'`` or ``'RPL'``. Default is True.
    filter_protein_coding : bool, optional
        If True, keep only genes with ``adata.var['gene_biotype'] == 'protein_coding'``.
        Requires that ``'gene_biotype'`` exists in ``adata.var``. Default is True.

    Returns
    -------
    AnnData
        An AnnData object containing only the filtered genes. The result may be a view
        or a copy depending on AnnData slicing semantics.

    Raises
    ------
    ValueError
        If ``filter_protein_coding`` is True but ``'gene_biotype'`` is not present in ``adata.var``.

    Notes
    -----
    - The function applies filters in the order described above.
    - Users who require an explicit deep copy should call ``.copy()`` on the returned AnnData.

    Examples
    --------
    >>> cleaned = clean_genes(adata, min_cells=5, filter_mito=True, filter_ribo=False)
    """
    # Make sure that gene_biotype information is available if filtering for protein-coding genes
    if filter_protein_coding and "gene_biotype" not in adata.var.columns:
        raise ValueError(
            "Gene biotype information not found in adata.var['gene_biotype']. Cannot filter for protein-coding genes."
        )

    # Filter genes by minimum cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if filter_mito:
        # Exclude mitochondrial genes
        mito_genes = adata.var_names.str.startswith("MT-")
        adata = adata[:, ~mito_genes]
    if filter_ribo:
        # Exclude ribosomal genes
        ribo_genes = adata.var_names.str.startswith(("RPS", "RPL"))
        adata = adata[:, ~ribo_genes]
    if filter_protein_coding:
        # Keep only protein-coding genes
        protein_coding_mask = adata.var["gene_biotype"] == "protein_coding"
        adata = adata[:, protein_coding_mask]
    return adata


def aggregate_matrix(
    adata: ad.AnnData,
    groupby: str,
    *,
    metric: Metric = "mean",
    layer: Optional[str] = None,
    use_raw: bool = False,
    detection_threshold: float = 0.0,
    normalize: Norm = "none",
    normalize_axis: Literal["gene", "group"] = "gene",
    sort_groups: bool = False,
    sort_genes: bool = False,
) -> pd.DataFrame:
    """Aggregate AnnData by group into a wide matrix (groups x genes) for dotplots/heatmaps.
    
    Parameters    
    ----------
    adata : AnnData
        The annotated data matrix to aggregate.
    groupby : str
        The key in adata.obs to group by for aggregation.
    metric : str, optional
        The metric to compute for each group. Options:
        - "mean": Average expression per group (default).
        - "sum": Total expression per group.
        - "detection": Fraction of cells in the group with expression above `detection_threshold`.
    layer : str, optional
        The layer in adata to use for aggregation. If None, uses adata.X. Default is None.
    use_raw : bool, optional
        If True, use adata.raw for aggregation instead of adata.X or the specified layer. Default is False.
    detection_threshold : float, optional
        Threshold for considering a gene as "detected" in the "detection" metric. Default is 0.0.
    normalize : str, optional
        Normalization method to apply to the aggregated matrix. Options:
        - "minmax": Apply min-max scaling to the aggregated values (0 to 1).
        - "none" or None: No normalization (default).
    normalize_axis : str, optional
        Axis to apply normalization on if `normalize` is "minmax". Options:
        - "gene": Normalize each gene across groups (default).
        - "group": Normalize each group across genes.
    sort_groups : bool, optional
        If True, sort the resulting DataFrame by group names (index). Default is False.
    sort_genes : bool, optional
        If True, sort the resulting DataFrame by gene names (columns). Default is False.    
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"{groupby!r} not found in adata.obs")

    var_names = adata.raw.var_names if use_raw else adata.var_names

    if metric in ("mean", "sum"):
        adata_agg = sc.get.aggregate(adata, by=groupby, func=metric)  # type: ignore[arg-type]

        if metric not in adata_agg.layers:
            raise RuntimeError(
                f"Expected aggregated layer {metric!r} in adata_agg.layers, got {list(adata_agg.layers.keys())}"
            )

        mat = adata_agg.layers[metric]
        df = pd.DataFrame(mat, index=adata_agg.obs_names, columns=adata_agg.var_names)

    elif metric == "detection":
        if use_raw:
            if adata.raw is None:
                raise ValueError("use_raw=True but adata.raw is None.")
            X = adata.raw.X
        else:
            X = adata.layers[layer] if layer is not None else adata.X

        if sparse.issparse(X):
            det = X.copy()
            det.data = (det.data > detection_threshold).astype(np.float32)
        else:
            det = (X > detection_threshold).astype(np.float32)

        tmp = ad.AnnData(
            det,
            obs=adata.obs[[groupby]].copy(),
            var=pd.DataFrame(index=var_names),
        )
        det_agg = sc.get.aggregate(tmp, by=groupby, func="mean")  # type: ignore[arg-type]

        mat = det_agg.layers["mean"]
        df = pd.DataFrame(mat, index=det_agg.obs_names, columns=det_agg.var_names)

    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    if sort_groups:
        df = df.sort_index()
    if sort_genes:
        df = df.loc[:, sorted(df.columns)]

    if normalize == "minmax":
        axis = 0 if normalize_axis == "gene" else 1
        df = minmax_df(df, axis=axis)
    elif normalize in (None, "none"):
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize!r}")

    return df

def aggregate(
    adata: ad.AnnData, groupby: str, method: Metric | None = "mean", normalize: Norm = "minmax", normalize_axis: Literal["gene", "group"] = "gene"
) -> pd.DataFrame:
    """
    Aggregate single cell data by specified grouping.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        The key in adata.obs to group by.
    method : Metric | None
        The aggregation method to use ('mean', 'sum', 'detection').

    Returns:
    -------
    AnnData
        The aggregated annotated data matrix.
    """
    # Compute matrix via new engine
    mat_agg = aggregate_matrix(
        adata,
        groupby,
        metric=method,
        normalize=normalize,      
        normalize_axis=normalize_axis,    
    )

    # Preserve old ordering step if you rely on it downstream
    # group_by_max expected genes x groups, hence transpose
    # (Assumes you already have group_by_max imported in your module.)
    mat_agg = group_by_max(mat_agg.T)
    return mat_agg.T


def lognorm(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """
    Log-normalize the data in an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix to be log-normalized.
    target_sum : float, optional
        The target sum for normalization. Default is 1e4.

    Returns:
    -------
    AnnData
        The log-normalized annotated data matrix.
    """
    # First make copy of adata to avoid modifying the original object
    adata = adata.copy()
    # Check whether adata.raw is not None
    if adata.raw is None:
        adata.raw = adata.copy()
        adata.layers["counts"] = adata.X.copy()
    # Normalize total counts per cell and log-transform the data
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    return adata


def dimreduce(
    adata: ad.AnnData,
    method: DimReduceMethod = "umap",
    *,
    copy: bool = True,
    n_comps: int = 50,  # PCA components
    n_neighbors: int = 15,  # neighbors graph
    n_pcs: Optional[int] = None,  # PCs to use for neighbors; default = n_comps
) -> ad.AnnData:
    """
    Perform dimensionality reduction (PCA or UMAP) on an AnnData object.

    Results are stored in:
      - PCA:   adata.obsm["X_pca"] (and adata.uns["pca"])
      - UMAP:  adata.obsm["X_umap"] (and adata.uns["umap"])

    Parameters
    ----------
    adata
        Input AnnData.
    method
        "pca" or "umap".
    copy
        If True, return a copy and do not modify the input object.
    n_comps
        Number of PCA components to compute (if needed).
    n_neighbors
        Number of neighbors for the kNN graph (UMAP).
    n_pcs
        Number of PCs to use for neighbors (UMAP). If None, uses n_comps.

    Returns
    -------
    AnnData
        AnnData with computed embeddings in `.obsm`.
    """
    adata_out = adata.copy() if copy else adata
    n_pcs_eff = n_comps if n_pcs is None else n_pcs

    def _ensure_pca() -> None:
        if "X_pca" not in adata_out.obsm:
            sc.tl.pca(adata_out, n_comps=n_comps)
            return

        # If PCA exists but has fewer comps than required, recompute
        X_pca = adata_out.obsm["X_pca"]
        if isinstance(X_pca, np.ndarray) and X_pca.shape[1] < n_comps:
            sc.tl.pca(adata_out, n_comps=n_comps)

    if method == "pca":
        _ensure_pca()
        return adata_out

    # method == "umap"
    _ensure_pca()

    # If neighbors already exist, Scanpy will overwrite if we call sc.pp.neighbors again.
    # That is usually fine, but you could add a flag to reuse.
    sc.pp.neighbors(adata_out, n_neighbors=n_neighbors, n_pcs=n_pcs_eff)
    sc.tl.umap(adata_out)

    return adata_out


def preprocess(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: Optional[str] = None,
    covariates: Optional[list[str]] = None,
    layer: Optional[str] = None,
) -> ad.AnnData:
    """Preprocess an AnnData object for downstream analysis.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix to preprocess.
    n_top_genes : int, optional
        Number of highly variable genes to select. Default is 2000.
    flavor : str, optional
        The method to use for selecting highly variable genes. Default is "seurat_v3".
    batch_key : str, optional
        The key in adata.obs to use for batch-aware HVG selection. Default is None (no batch-aware selection).
    covariates : list of str, optional
        List of covariate keys in adata.obs to regress out. Default is None (no regression).
    layer : str, optional
        The layer in adata to use for HVG selection. Default is None (use adata.X).
    """
    # Assume data is already filtered for quality control and cleaned of unwanted genes
    # Log-normalize the data
    adata = lognorm(adata)
    # Select highly variable genes
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor=flavor, batch_key=batch_key, layer=layer
    )
    # Check if covariates are provided
    if covariates is not None:
        # Regress out effects of total counts per cell and percentage of mitochondrial genes
        sc.pp.regress_out(adata, covariates=covariates)
    # Scale data to unit variance and zero mean
    sc.pp.scale(adata, max_value=10)
    # Perform dimensionality reduction
    adata = dimreduce(adata)
    # Put log-normalized back
    adata.X = adata.layers["lognorm"]
    return adata

def weighted_proportions(
    adata: ad.AnnData,
    batch_key: str,
    group_key: str,
):
    """
    Compute sample-size-weighted proportions of `group_key` across batches
    
    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix containing the batch and group annotations.
    batch_key : str
        The key in adata.obs that identifies the batch (e.g., sample or dataset).
    group_key : str
        The key in adata.obs that identifies the group for which to compute proportions (e.g., cell type).  
        
    Returns:
    -------
    pd.Series
        A Series indexed by group_key categories containing the weighted proportions across batches.
    """

    obs = adata.obs

    # Count cells per batch and group
    counts_df = (
        obs.groupby([batch_key, group_key])
        .size()
        .reset_index(name="count")
    )

    # Total cells per batch
    total_cells = (
        obs.groupby(batch_key)
        .size()
        .rename("sample_size")
        .reset_index()
    )

    # Merge sample sizes
    counts_df = counts_df.merge(total_cells, on=batch_key, how="left")

    # Per-batch percentage
    counts_df["percentage"] = counts_df["count"] / counts_df["sample_size"]

    # Your original weighted pooled formula
    weighted_prop = (
        counts_df
        .groupby(group_key)
        .apply(lambda x: (x["percentage"] * x["sample_size"]).sum() / x["sample_size"].sum())
    )

    return weighted_prop