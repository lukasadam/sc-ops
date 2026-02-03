# Load genome file for snapatac2

from pathlib import Path
from snapatac2.genome import Genome

def load_genome(genome_dir: Path, genome_name: str) -> Genome:
    """Load genome data from the specified directory."""
    genome = Genome(fasta=genome_dir / f"{genome_name}.fa",
                    annotation=genome_dir / f"{genome_name}.gtf")
    return genome

def remove_nonstandard_chromosomes(genome: Genome) -> Genome:
    """Remove non-standard chromosomes from the genome object."""
    for chrom in list(genome.chrom_sizes.keys()):
        if "." in chrom:
            del genome.chrom_sizes[chrom]