from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_validator
import base64


class Artifact(BaseModel):
    name: str = Field(..., description="Name of the artifact")
    asset_path: str = Field(..., description="The storage path of the asset")
    version: str = Field(..., description="Version of the artifact")
    data: Any = Field(..., description="Binary or serialized data")
    metadata: Dict[str, Any] = Field({}, description="Experiment and run ID")
    type: str = Field(..., description="Model of the artifact")
    tags: List[str] = Field([], description="Tags describing the artifact")
    id: str = Field(None, description="ID made from asset_path and version")

    @model_validator(mode="before")
    def _generate_id(cls, values):
        """
        Generates an ID for the artifact based on the asset_path and version.

        The ID is created by encoding the asset_path using URL-safe base64
        and combining it with the version using a colon separator.

        Args:
            values (dict): A dictionary containing 'asset_path' and
            'version' keys.

        Returns:
            str: A unique ID for the artifact.

        Raises:
            ValueError: If 'asset_path' or 'version' is not provided in
            the values.
        """
        if values.get("id") is None:
            asset_path = values.get("asset_path")
            version = values.get("version")
            if asset_path and version:
                path = base64.urlsafe_b64encode(asset_path.encode()).decode()
                values["id"] = f"{path}:{version}"
            else:
                raise ValueError("Asset_path and version are not provided")
        return values

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns the metadata of the artifact.

        The metadata includes the asset_path, version, type, tags, and
        id of the artifact.
        Additionally, the metadata includes any additional fields that
        were provided in
        the `metadata` dictionary when initializing the artifact.

        Returns:
            Dict[str, Any]: A dictionary containing the artifact's metadata
        """
        return {
            "name": self.name,
            "asset_path": self.asset_path,
            "version": self.version,
            "data": self.data,
            **self.metadata,
            "type": self.type,
            "tags": self.tags,
            "id": self.id
        }
