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

def minmax(df: pd.DataFrame, axis=1, eps=1e-8) -> pd.DataFrame:
    """Min-max normalize a DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame to be normalized.
    axis : int
        Axis along which to normalize (0 for columns, 1 for rows).
    eps : float
        Small value to avoid division by zero.

    Returns:
    -------
    pd.DataFrame
        Min-max normalized DataFrame.
    """
    return (df - df.min(axis=axis, keepdims=True)) / (df.max(axis=axis, keepdims=True) - df.min(axis=axis, keepdims=True) + eps)