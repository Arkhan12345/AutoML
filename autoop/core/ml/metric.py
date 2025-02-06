from abc import ABC, abstractmethod
from typing import Any
import numpy as np


METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "accuracy",
    "weighted_precision",
    "weighted_recall"

]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> Any:
    """
    Factory function to get a metric by name.

    Args:
        name (str): str name of the metric

    Returns:
        Any: a metric instance given its str name. If name is not found,
        returns None.
    """
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "root_mean_squared_error": RootMeanSquaredError(),
        "accuracy": Accuracy(),
        "weighted_precision": WeightedPrecision(),
        "weighted_recall": WeightedRecall()
    }
    return metrics.get(name, None)


class Metric(ABC):
    """Base class for all metrics.
    """
    def __init__(self, name: str, description: str = None):
        """
        Initialize a metric.

        Args:
            name (str): str name of the metric.
            description (str, optional): str description of the metric.
        """
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Expose the name of the metric."""
        return self._name

    def __str__(self) -> str:
        """Return a string representation of the object.

        Returns:
            str: A string representation of the object in the format
            {name}: {description}
        """
        return f"{self._name}: {self._description}"

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric given ground truth and predictions.
        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values
        Returns:
            float: Calculated metric as real number
        """
        pass


class MeanAbsoluteError(Metric):
    def __init__(self):
        """
        Initialize the MeanAbsoluteError metric.

        This metric measures the average absolute difference
        between the true values and the predicted values.

        Args:
            None
        """
        super().__init__(
            name="mean_absolute_error",
            description="Measures the\
            average absolute difference"
            )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean absolute difference between the ground truth
        and the predictions.

        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values

        Returns:
            float: The mean absolute difference between the ground truth
            and the predictions
        """
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Metric):
    def __init__(self):
        """
        Initialize the MeanSquaredError metric.

        This metric measures the average squared difference
        between the true values and the predicted values.

        Args:
            None
        """
        super().__init__(
            name="mean_squared_error",
            description="Measures the average squared difference"
            )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean squared difference between the ground truth
        and the predictions.

        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values

        Returns:
            float: The mean squared difference between the ground truth
            and the predictions
        """
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    def __init__(self):
        """
        Initialize the RootMeanSquaredError metric.

        This metric measures the square root of the average
        squared difference between the true values and the predicted values.

        Args:
            None
        """
        super().__init__(
            name="root_mean_squared_error",
            description="Measures the square root\
            of the average squared difference"
            )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the root mean squared difference between the ground truth
        and the predictions.

        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values

        Returns:
            float: The root mean squared difference between the ground truth
            and the predictions
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse


class Accuracy(Metric):
    def __init__(self):
        """
        Initialize the Accuracy metric.

        This metric measures the proportion of correctly determined features.
        Accuracy is defined as the number of correctly classified samples
        divided by the total number of samples.

        Args:
            None
        """
        super().__init__(
            name="accuracy",
            description="measures the proportion of\
            correctly determined features"
            )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the accuracy between the ground truth and the predictions.

        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values

        Returns:
            float: The accuracy between the ground truth and the predictions
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The lengths of y_true and y_pred must match.")
        assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape}\
            vs {y_pred.shape}"

        return np.sum(y_true == y_pred) / len(y_true)


def precision_recall(
        type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
        ) -> float:
    """
    Compute the weighted precision or recall between the ground truth
    and the predictions.

    Parameters
    ----------
    type : str
        The type of metric to compute. It can be either 'precision' or 'recall'
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        The weighted precision or recall between the ground truth
        and the predictions
    """

    classes = np.unique(y_true)
    total_weight = y_true.size
    result = 0
    weighted_result = 0

    for cls in classes:
        true_positives = np.sum((y_pred == cls) & (y_true == cls))
        if type == "recall":
            false_pos_neg = np.sum((y_pred != cls) & (y_true == cls))
        elif type == "precision":
            false_pos_neg = np.sum((y_pred == cls) & (y_true != cls))
        try:
            result = true_positives/(true_positives+false_pos_neg)
        except ZeroDivisionError:
            result = 0
        weight = np.sum(y_true == cls)

        weighted_result += (result * weight) / total_weight
    return weighted_result


class WeightedPrecision(Metric):
    def __init__(self):
        """
        Initializes a WeightedPrecision metric.

        Parameters
        ----------
        None

        Returns
        -------
        WeightedPrecision
            An instance of the WeightedPrecision metric
        """
        super().__init__(
            name="weighted_precision",
            description="calculates the weighted\
            true positive proportion"
            )

    def evaluate(self, y_true, y_pred) -> float:
        """
        Evaluate the weighted precision metric using the
        precision_recall function.

        Parameters
        ----------
        y_true : np.ndarray
        The true labels of the data
        y_pred : np.ndarray
        The predicted labels of the data

        Returns
        -------
        float
            The weighted precision score
        """
        return precision_recall("precision", y_true, y_pred)


class WeightedRecall(Metric):
    def __init__(self):
        """
        Initializes a WeightedRecall metric.

        Parameters
        ----------
        None

        Returns
        -------
        WeightedRecall
            An instance of the WeightedRecall metric
        """
        super().__init__(
            name="weighted_recall",
            description="calculates the weighted\
            true positive rate"
            )

    def evaluate(self, y_true, y_pred) -> float:
        """
        Evaluates the weighted recall metric using the
        precision_recall function.

        Parameters
        ----------
        y_true : np.ndarray
        The true labels of the data
        y_pred : np.ndarray
        The predicted labels of the data

        Returns
        -------
        float
            The weighted recall metric
        """
        return precision_recall("recall", y_true, y_pred)
