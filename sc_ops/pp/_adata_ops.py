# Common Anndata operations for single-cell data

from __future__ import annotations

from typing import Literal, Optional, TypeAlias

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sc_ops.pp._utils import group_by_max, minmax

DimReduceMethod: TypeAlias = Literal["pca", "umap"]
AggFunc: TypeAlias = Literal["mean", "sum", "median", "count_nonzero"]


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


def aggregate(
    adata: ad.AnnData, groupby: str, method: AggFunc | None = "mean"
) -> pd.DataFrame:
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
    adata_agg = sc.get.aggregate(adata, by=groupby, func=method)  # pyright: ignore[reportArgumentType]
    # Convert the aggregated AnnData to a DataFrame
    mat_agg = pd.DataFrame(
        adata_agg.layers[method], index=adata_agg.obs_names, columns=adata_agg.var_names
    )
    # Apply min-max scaling per gene
    mat_agg = minmax(mat_agg, axis=0)
    # Order DataFrame by group of max value
    mat_agg = group_by_max(mat_agg.T)
    return mat_agg


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
