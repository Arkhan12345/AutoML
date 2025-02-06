from abc import abstractmethod
import numpy as np
from typing import Literal


from autoop.core.ml.artifact import Artifact


class Model:
    def __init__(
            self,
            type: Literal["classification", "regression"]
                 ):
        """
        Initializes a Model instance.

        Parameters
        ----------
        type : Literal["classification", "regression"]
            The type of model, specifying whether it is a classification
            or regression model.
        model : Model
            The model class to be instantiated.
        *args :
            Positional arguments to be passed to the model instantiation.
        **kwargs :
            Keyword arguments to be passed to the model instantiation.
        """
        self._parameters: dict = {}
        self._type = type

    @property
    def type(self) -> str:
        """
        Returns the type of the model.

        Returns
        -------
        str
            Either "classification" or "regression".
        """
        return self._type

    @abstractmethod
    def parameters(self):
        """
        Returns a deep copy of the model's parameters.

        Returns
        -------
        dict
            A dictionary containing the model's parameters, including both
            strict parameters and hyperparameters.
        """
        pass

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model to an artifact.

        The artifact is created by serializing the model's parameters
        into a bytes object and saving it in the artifact's data
        attribute.

        Parameters
        ----------
        name : str
            The name of the artifact.

        Returns
        -------
        Artifact
            The created artifact.
        """
        return Artifact(name=name, data=self.parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, target: np.ndarray) -> None:
        """
        Fits the model to the given observations and target values.

        Parameters
        ----------
        observations : np.ndarray
            The observations to fit the model to. It should be an array, where
            each row is an observation and each column is a feature.
        target : np.ndarray
            The target values to fit the model to. It should be an array.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the input data.

        Parameters
        ----------
        input : np.ndarray
            The input data to make predictions on. It should be an array, where
            each row is an observation and each column is a feature.

        Returns
        -------
        np.ndarray
            The predictions made by the model. It should be an array, where
            each row is the prediction of the corresponding observation in the
            input array.
        """
        pass


class SklearnWrapperModel(Model):
    def __init__(
            self, 
            model,
            type: Literal["classification", "regression"]
            ):
        super().__init__(type)
        self._model = model
    
    @property
    def model(self):
        return self._model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the provided data.

        Parameters
        ----------
        X : np.ndarray
            2D array of input features.
        y : np.ndarray
            1D array of target variable.
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions based on the input data.

        Parameters
        ----------
        X : np.ndarray
            2D array of input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return self._model.predict(X)

    def __str__(self) -> str:
        return f"SklearnWrapperModel(model={self._model})"
