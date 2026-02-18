import scanpy as sc

sc.settings.verbosity = 2               # Show progress
sc.settings.set_figure_params(
    dpi=150,                            # High-res output
    dpi_save=300,                       # High-res when saving
    format='pdf',                       # or 'pdf', 'svg'
    facecolor='white',                  # or 'none' for transparent
    frameon=False,                      # No outer frame
    vector_friendly=True,               # No rasterization warnings
    fontsize=10,                        # Base font size
    figsize=(4, 4),                     # Default figure size
    transparent=True                    # Transparent background if saving
)