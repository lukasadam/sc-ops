import snapatac2 as snap
from pathlib import Path
from typing import Dict

def import_fragments(files: Dict[str, Path], output_dir: Path, genome: snap.genome.Genome, min_num_fragments: int = 1000, sorted_by_barcode: bool = False, n_jobs: int = 50) -> None:
    """Import fragment files into AnnData objects and save them to disk.
    
    Parameters
    ----------
    files : Dict[str, Path]
        A dictionary mapping sample names to their corresponding fragment file paths.
    output_dir : Path
        The directory where the output AnnData files will be saved.
    genome : snap.genome.Genome
        The genome object containing chromosome sizes and annotations.
    min_num_fragments : int, optional
        Minimum number of fragments required for a cell to be included, by default 1000.
    sorted_by_barcode : bool, optional
        Whether the input fragment files are sorted by barcode, by default False.
    n_jobs : int, optional
        The number of parallel jobs to use for importing fragments, by default 50.
    """
    return snap.pp.import_fragments(
        [files[fl] for fl in files.keys()],
        file=[output_dir / (name + '.atac.raw.h5ad') for name in files.keys()],
        chrom_sizes=genome,
        min_num_fragments=min_num_fragments,
        sorted_by_barcode=sorted_by_barcode,
        n_jobs=n_jobs
    )