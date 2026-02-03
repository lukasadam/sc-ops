import pandas as pd

def group_by_max(expr: pd.DataFrame) -> pd.DataFrame:
    """Order genes by group of max value.

    Parameters:
    ----------
    expr : pd.DataFrame
        A DataFrame where rows are genes and columns are samples.

    Returns:
    -------
    expr: pd.DataFrame
        A DataFrame which is sorted by the group of max value.
    """
    max_group = expr.idxmax(axis=1)
    ordered = []
    for group in expr.columns:
        ordered.extend(expr[max_group == group].index.tolist())
    return expr.loc[ordered]