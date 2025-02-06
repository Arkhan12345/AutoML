
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()
    feature_list = []

    for column in list(df.columns):
        unique_values = set(df[column])
        if df[column].dtype == str or len(unique_values) < 10:
            type = "categorical"
        elif df[column].dtype in [int, 'int64', float, 'float64']:
            type = "numerical"
        elif df[column].dtype == bool or df[column].dtype == 'bool':
            type = "categorical"
        else:
            raise ValueError("This is not a valid type for a feature")
        feature_list.append(Feature(name=column, type=type))
    return feature_list
