from typing import Tuple

import anndata
import cellmapper


def cellmapper(
    adata_query: anndata.AnnData,
    adata_reference: anndata.AnnData,
    obs_keys: list[str],
    use_rep: str,
    batch_key: str = None,
    obsm_key="presence_score",
    knn_method: str = "pynndescent",
    t: int = 100,
) -> Tuple[anndata.AnnData, anndata.AnnData, cellmapper.CellMapper]:
    """Run CellMapper to map query cells to reference cell types.

    Parameters
    ----------
    adata_query : anndata.AnnData
        The query AnnData object containing the cells to be mapped.
    adata_reference : anndata.AnnData
        The reference AnnData object containing the annotated cell types.
    obs_keys : list[str]
        List of observation keys in the reference AnnData to use for mapping (e.g., ["Class", "Subclass
    use_rep : str
        The key in adata_reference.obsm to use for the representation (e.g., "X_scVI").
    knn_method : str, optional
        The method to use for nearest neighbor search (default is "pynndescent").
    batch_key : str, optional
        If provided, the key in adata_reference.obs to use for computing presence scores (e.g., "dataset").
    t : int, optional
        Iterations for smoothing presence scores (default is 100).

    Returns
    -------
    adata_reference : anndata.AnnData
        The reference AnnData object with mapping results
    adata_query : anndata.AnnData
        The query AnnData object with mapping results
    cmap : cellmapper.CellMapper
        The fitted CellMapper object containing the mapping results.
    """
    # Check that obs_keys exist in adata_reference.obs
    for key in obs_keys:
        if key not in adata_reference.obs.columns:
            raise ValueError(
                f"Observation key '{key}' not found in adata_reference.obs."
            )
    # Check that use_rep exists in adata_reference.obsm
    if use_rep not in adata_reference.obsm:
        raise ValueError(
            f"Representation key '{use_rep}' not found in adata_reference.obsm."
        )
    cmap = cellmapper.CellMapper(adata_query, adata_reference).map(
        obs_keys=obs_keys,
        use_rep=use_rep,
        knn_method=knn_method,
    )
    # Compute presence scores
    if batch_key is not None:
        # Compute presence scores for each batch in the reference data
        cmap.compute_presence_score(groupby=batch_key)
        # Smooth presence scores using the specified number of iterations
        cellmapper.CellMapper(adata_reference).map(
            obs_keys="presence_score", t=t, use_rep=use_rep
        )
        # Add presence scores to adata_reference.obs for each batch
        query_batches = adata_query.obs[batch_key].unique().tolist()
        obs_score_names = [f"{obsm_key}_{key}" for key in query_batches]
        for obsm_name, obs_name in zip(query_batches, obs_score_names, strict=False):
            adata_reference.obs[obs_name] = adata_reference.obsm[obsm_key][obsm_name]
    return adata_reference, adata_query, cmap
