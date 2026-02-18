import pandas as pd
from anndata import AnnData
from pybiomart import Dataset


def get_orthologs(
    source_species: str,
    target_species: str,
    *,
    only_1to1: bool = True,
    host: str = "https://www.ensembl.org",
) -> pd.DataFrame:
    """Fetch a gene ortholog mapping from Ensembl BioMart using pybiomart.

    Parameters
    ----------
    source_species
        Source species identifier. Accepts Ensembl short names (e.g. 'mmusculus',
        'hsapiens') or common aliases ('mouse', 'human').
    target_species
        Target species identifier (same accepted formats as source_species).
    only_1to1
        If True, restricts to 1:1 orthologs only.
    host
        Ensembl host URL.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['source_symbol', 'target_symbol', 'orthology_type'].
    """
    if source_species == target_species:
        raise ValueError(
            f"source_species and target_species must differ (got {source_species!r})"
        )

    source_dataset_name = f"{source_species}_gene_ensembl"
    target_symbol_attr = f"{target_species}_homolog_associated_gene_name"
    target_orthology_attr = f"{target_species}_homolog_orthology_type"

    dataset = Dataset(name=source_dataset_name, host=host)
    try:
        df = dataset.query(
            attributes=[
                "external_gene_name",
                target_symbol_attr,
                target_orthology_attr,
            ]
        )
    except Exception as e:  # pragma: no cover
        msg = (
            "Failed to query Ensembl BioMart for orthologs. "
            "This usually means the species identifiers are not valid Ensembl prefixes "
            "or the requested homolog attributes are not available for this dataset. "
            f"source_dataset={source_dataset_name!r}, target_prefix={target_species!r}."
        )
        raise RuntimeError(msg) from e

    df.columns = ["source_symbol", "target_symbol", "orthology_type"]
    df = df.dropna(subset=["source_symbol", "target_symbol"])

    if only_1to1:
        df = df[df["orthology_type"] == "ortholog_one2one"]

    return df


def map_genes_to_orthologs(
    adata: AnnData,
    source_species: str,
    target_species: str,
    *,
    only_1to1: bool = True,
    drop_ambiguous: bool = True,
    make_var_names_unique: bool = True,
    host: str = "https://www.ensembl.org",
) -> AnnData:
    """Map gene symbols in an AnnData object from one species to another.

    Notes
    -----
    - Assumes `adata.var_names` contain gene symbols matching Ensembl's
      `external_gene_name` field for the source species.
    - If `only_1to1=False`, some source genes can map to multiple target genes.
      When `drop_ambiguous=True`, those source genes are removed.
    """
    print(f"Fetching {source_species}->{target_species} gene orthologs...")
    biomart_df = get_orthologs(
        source_species=source_species,
        target_species=target_species,
        only_1to1=only_1to1,
        host=host,
    )

    source_genes = pd.Index(adata.var_names)
    mapped = biomart_df[biomart_df["source_symbol"].isin(source_genes)].copy()

    if mapped.empty:
        print("No orthologs found for genes present in adata.var_names.")
        return adata.copy()

    if drop_ambiguous:
        # Remove source genes mapping to multiple target symbols.
        mapped = mapped.drop_duplicates(subset=["source_symbol", "target_symbol"])
        ambiguous = (
            mapped.groupby("source_symbol")["target_symbol"]
            .nunique()
            .pipe(lambda s: s[s > 1])
            .index
        )
        if len(ambiguous) > 0:
            mapped = mapped[~mapped["source_symbol"].isin(ambiguous)]

    mapped = mapped.drop_duplicates(subset=["source_symbol"], keep="first")
    gene_map = dict(zip(mapped["source_symbol"], mapped["target_symbol"]))
    genes_with_orthologs = list(gene_map.keys())

    adata_mapped = adata[:, adata.var_names.isin(genes_with_orthologs)].copy()
    adata_mapped.var_names = adata_mapped.var_names.map(gene_map)

    if make_var_names_unique and adata_mapped.var_names.has_duplicates:
        adata_mapped.var_names_make_unique()

    print(
        f"Mapped {len(genes_with_orthologs)} genes from {source_species} to {target_species}."
    )
    return adata_mapped
