from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import SklearnWrapperModel


class MultipleLinearRegression(SklearnWrapperModel):
    """Wrapper for a scikit-learn Multiple Linear Regression model"""
    def __init__(self, *args, **kwargs):
        """
        Initialize Linear Regression model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for Linear Regression model.
        **kwargs : dict
            Keyword arguments for Linear Regression model.
        """
        super().__init__(LinearRegression(*args, **kwargs), type="regression")
        
    def __str__(self) -> str:
        return "Multiple Linear Regression Model"
