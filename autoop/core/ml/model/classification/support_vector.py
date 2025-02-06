from sklearn.svm import SVC
from autoop.core.ml.model.model import SklearnWrapperModel


class SupportVectorClassifier(SklearnWrapperModel):
    """
    Wrapper of SVC model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize Support Vector Classifier model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for Support Vector Classifier model.
        **kwargs : dict
            Keyword arguments for Support Vector Classifier model.
        """
        super().__init__(SVC(*args, **kwargs), type="classification")
        
    def __str__(self) -> str:
        return "Support Vector Classifier Model"
