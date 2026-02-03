# Computes the tissue-specificity score (Tau) for gene expression data.
# The tissue specificity index tau is a metric, ranging from 0 to 1, that
# quantifies how specifically a gene is expressed across different tissues.
# A value near 0 indicates ubiquitous expression, while a value near 1 indicates
# high tissue specificity.

# Formula for Tau:
# tau = sum(1 - (x_i / max(x))) / (n - 1)
# where x_i is the expression level in tissue i, max(x) is the maximum expression level
# across all tissues for that gene, and n is the number of tissues.

import numpy as np
import pandas as pd


def tau_score(expression_data: pd.DataFrame) -> pd.Series:
    """
    Calculate the Tau tissue-specificity score for each gene.


    Parameters:
    ----------
    expression_data : pd.DataFrame
        A DataFrame where rows are genes and columns are tissues.
        Each cell contains the expression level of a gene in a tissue.

    Returns:
    -------
    pd.Series
        A Series containing the Tau score for each gene.
    """
    # Initialize a Series to hold Tau scores
    tau_scores = pd.Series(index=expression_data.index, dtype=float)

    # Iterate over each gene to calculate its Tau score
    for gene in expression_data.index:
        expr_values = expression_data.loc[gene].values
        max_expr = np.max(expr_values)
        n_tissues = len(expr_values)

        if max_expr == 0:
            tau_scores[gene] = 0.0
        else:
            tau = np.sum(1 - (expr_values / max_expr)) / (n_tissues - 1)
            tau_scores[gene] = tau

    return tau_scores
