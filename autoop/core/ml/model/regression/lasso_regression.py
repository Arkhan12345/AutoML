from sklearn.linear_model import Lasso
from autoop.core.ml.model.model import SklearnWrapperModel


class LassoRegression(SklearnWrapperModel):
    """Wrapper for Lasso Regression using scikit-learn"""
    def __init__(self, *args, **kwargs):
        """
        Initialize Lasso Regression model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for Lasso model.
        **kwargs : dict
            Keyword arguments for Lasso model.
        """
        super().__init__(Lasso(*args, **kwargs), type="regression")
        
    def __str__(self) -> str:
        return "Lasso Regression Model"
