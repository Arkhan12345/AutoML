from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import SklearnWrapperModel


class RidgeRegression(SklearnWrapperModel):
    """Wrapper for Ridge Regression using scikit-learn"""
    def __init__(self, *args, **kwargs):
        """
        Initialize Ridge Regression model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for Ridge model.
        **kwargs : dict
            Keyword arguments for Ridge model.
        """
        super().__init__(Ridge(*args, **kwargs), type="regression")
        
    def __str__(self) -> str:
        return "Ridge Regression Model"
