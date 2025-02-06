from pydantic import BaseModel, PrivateAttr
from typing import Literal


class Feature(BaseModel):
    _name: str = PrivateAttr
    _type: str = PrivateAttr

    def __init__(self, name: str,
                 type: Literal["numerical", "categorical"], **kwargs) -> None:
        super().__init__(**kwargs)
        self.type = type
        self.name = name

    @property
    def type(self) -> str:
        """
        The type of the feature.
        It can be either numerical or categorical.
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Setter for the type of the feature.
        Validates that the provided name is either
        numerical or categorical.
        """
        if value not in ["numerical", "categorical"]:
            raise ValueError(f"{value} is not a valid type!")
        else:
            self._type = value

    @property
    def name(self) -> str:
        """
        The name of the feature.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Setter for the name of the feature.
        Validates that the provided value is a non-empty string.
        """
        if isinstance(value.strip(), str):
            self._name = value
        else:
            raise ValueError(f"{value} is not a valid name!")
