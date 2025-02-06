from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model.model import SklearnWrapperModel


class DTreeClassifier(SklearnWrapperModel):
    """
    Wrapper of the DecisionTreeClassifier model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize DecisionTree Classifier model.

        Parameters
        ----------
        *args : tuple
            Positional arguments for DecisionTree Classifier model.
        **kwargs : dict
            Keyword arguments for DecisionTree Classifier model.
        """
        super().__init__(model=DecisionTreeClassifier(*args, **kwargs), type="classification")

    def __str__(self) -> str:
        return "DecisionTree Classifier Model"
