from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import LassoRegression
from autoop.core.ml.model.classification import (
    KNearestClassifier,
    SupportVectorClassifier,
    DTreeClassifier
)


REGRESSION_MODELS = [
    "multiple_linear_regression",
    "lasso_regression",
    "ridge_regression"
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "k_neighbours_classification",
    "support_vector_classification",
    "decision_tree_classification"
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    models = {
        "multiple_linear_regression": MultipleLinearRegression(),
        "lasso_regression": LassoRegression(),
        "ridge_regression": RidgeRegression(),
        "k_neighbours_classification": KNearestClassifier(),
        "support_vector_classification": SupportVectorClassifier(),
        "decision_tree_classification": DTreeClassifier()
    }
    return models.get(model_name, None)
