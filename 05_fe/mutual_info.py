from sklearn.metrics import mutual_info_score

def calculate_mutual_information(series1, series2):
    """
    Calculate the mutual information between two pandas Series.

    Args:
    series1 (pd.Series): First series.
    series2 (pd.Series): Second series.

    Returns:
    float: The mutual information score between the two series.
    """
    return mutual_info_score(series1, series2)