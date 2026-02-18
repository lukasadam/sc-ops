# Common Anndata operations for single-cell data

import anndata as ad
import pandas as pd
import scanpy as sc
from sc_ops.pp._utils import group_by_max, minmax
from typing import Literal, TypeAlias

AggFunc: TypeAlias = Literal["mean", "sum", "median", "count_nonzero"]


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
