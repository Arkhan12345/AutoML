from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
         ) -> 'Dataset':
        """
        Creates a new Dataset artifact from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be saved as
            a dataset artifact.
            name (str): The name of the dataset artifact.
            asset_path (str): The path to the dataset artifact.
            version (str, optional): The version of the dataset artifact.

        Returns:
            Dataset: The created dataset artifact.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the data from this dataset artifact and returns it as
        a pandas DataFrame.

        Returns:
            pd.DataFrame: The data from this dataset artifact.
        """
        csv = self.data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the given DataFrame as bytes.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The data encoded as bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
