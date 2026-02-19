import snapatac2 as snap
from pathlib import Path

def load_hg38_genome() -> snap.genome.Genome:
    """Load the human genome (hg38) and its annotation."""
    genomes_dir = Path("/projects/site/pred/ihb-intestine-evo/lukas_area/genomes")
    genome = snap.genome.Genome(fasta=genomes_dir / "hg38.fa", annotation=genomes_dir / "hg38.sorted.gtf")
    # Remove non-standard chromosomes (e.g., haplotypes, unplaced contigs)
    for key in list(genome.chrom_sizes.keys()):
        if "." in key:
            del genome.chrom_sizes[key]
    return genome