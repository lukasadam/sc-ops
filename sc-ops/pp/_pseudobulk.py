# Pseudobulking of single-cell RNA-seq data. Adopted from delnx

import decoupler as dc
import numpy as np
from anndata import AnnData

def pseudobulk(
    adata: AnnData,
    sample_key: str = "batch",
    group_key: str | None = None,
    n_pseudoreps: int | None = None,
    layer: str | None = None,
    min_cells: int | None = 5,
    min_counts: int | None = 5000,
    mode: str = "sum",
    **kwargs,
) -> AnnData:
    """Create pseudobulk samples from single-cell RNA-seq data.

    This function aggregates single-cell RNA-seq data into pseudobulk samples based on specified sample and group identifiers. It can optionally create random pseudoreplicates.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        AnnData object containing single-cell expression data.
    sample_key : str, default="batch"
        Column name in `adata.obs` that contains the sample identifiers.
        Each unique value will become a separate pseudobulk sample
        (or multiple samples if `n_pseudoreps` is specified).
    group_key : str | None, default=None
        Column name in `adata.obs` that contains group identifiers like cell types.
        If provided, pseudobulk samples will be generated separately for each combination
        of sample and group, enabling cell type-specific analysis.
    n_pseudoreps : int | None, default=None
        Number of pseudoreplicates to create per sample. If :obj:`None`, uses the original
        sample structure without resampling. If specified, creates `n_pseudoreps`
        pseudoreplicates per sample by randomly sampling cells with replacement.
    layer : str | None, default=None
        Layer in `adata.layers` to use for aggregation. If :obj:`None`, uses `adata.X`.
        Should contain raw or normalized counts, not log-transformed values.
    mode : str, default="sum"
        Method to aggregate cell-level data into pseudobulk samples:
            - "sum": Sum of counts (recommended for RNA-seq data)
            - "mean": Average of counts across cells
    min_cells : int | None, default=5
        Minimum number of cells required in a pseudobulk sample to retain it.
        Samples with fewer cells will be discarded.
    min_counts : int | None, default=5000
        Minimum total counts required in a pseudobulk sample to retain it.
        Samples with fewer total counts will be discarded.
    **kwargs
        Additional arguments to pass to :func:`decoupler.pp.pseudobulk`

    Returns
    -------
        AnnData object containing the pseudobulk data. The structure changes from
        cell-level to sample-level, with each row representing a pseudobulk sample.
        Original sample and group identifiers are preserved in the observations.
    
    """
    # Create pseudoreplicates if requested
    if n_pseudoreps is not None:
        # Randomly assign each cell to one of the n_pseudoreps replicates for its sample
        pseudoreps = np.random.choice(
            np.arange(n_pseudoreps),
            size=adata.n_obs,
            replace=True,
        )
        # Create unique identifiers for each pseudoreplicate by combining sample ID with replicate number
        adata.obs["psbulk_replicate"] = adata.obs[sample_key].astype(str) + "_" + pseudoreps.astype(str)
    else:
        # Use original sample IDs if no pseudoreplicates requested
        adata.obs["psbulk_replicate"] = adata.obs[sample_key].astype(str)

    # Call decoupler's pseudobulk function to perform the actual aggregation
    adata_pb = dc.pp.pseudobulk(
        adata,
        sample_col="psbulk_replicate",  # Column containing our sample/pseudoreplicate IDs
        groups_col=group_key,  # Optional column for separate aggregation (e.g., cell types)
        layer=layer,  # Which data layer to use
        mode=mode,  # How to aggregate (sum, mean, median)
        empty=True,  # Discard empty pseudobulk samples
        **kwargs,  # Pass additional parameters to decoupler
    )

    # Filter out pseudobulk samples with too few cells or counts
    if min_cells is not None:
        adata_pb = adata_pb[adata_pb.obs["psbulk_cells"] >= min_cells, :].copy()

    if min_counts is not None:
        adata_pb = adata_pb[adata_pb.obs["psbulk_counts"] >= min_counts, :].copy()

    return adata_pb