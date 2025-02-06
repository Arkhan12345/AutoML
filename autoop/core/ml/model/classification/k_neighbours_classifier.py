from autoop.core.ml.model.model import SklearnWrapperModel
from sklearn.neighbors import KNeighborsClassifier


class KNearestClassifier(SklearnWrapperModel):
    """
    Wrapper of KNeighboursClassifier model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize KNearest Classifier model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for KNearest Classifier model.
        **kwargs : dict
            Keyword arguments for KNearest Classifier model.
        """
        super().__init__(KNeighborsClassifier(*args, **kwargs), type="classification")

    def __str__(self) -> str:
        return "KNearest Classifier Model"
