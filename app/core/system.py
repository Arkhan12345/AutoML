from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    def __init__(self,
                 database: Database,
                 storage: Storage):
        """
        Initialize the ArtifactRegistry with a database and storage.

        Args:
            database (Database): The database instance to store
            artifact metadata.
            storage (Storage): The storage instance to manage artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        # save the artifact in the storage
        """
        Registers an artifact in the database and storage.

        Args:
            artifact (Artifact): The artifact to register.

        """
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts in the registry.

        Args:
            type (str, optional): Filter artifacts by type. Defaults to None.

        Returns:
            List[Artifact]: A list of artifacts in the registry.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact from the registry.

        Args:
            artifact_id (str): The id of the artifact.

        Returns:
            Artifact: The artifact with the given id.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        """
        Deletes an artifact from the registry.

        This method removes the artifact's data from the storage and
        deletes its metadata from the database.

        Args:
            artifact_id (str): The id of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes an instance of the AutoMLSystem class.

        The AutoMLSystem is a singleton class that manages the storage and
        database of the AutoML system.

        Args:
            storage (LocalStorage): The local storage instance.
            database (Database): The database instance.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """
        Gets the instance of the AutoMLSystem singleton class.

        The first time this method is called, it creates an instance of the
        AutoMLSystem class and initializes its storage and database.
        Subsequent calls will return the same instance.

        Returns:
            AutoMLSystem: The instance of the AutoMLSystem class.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        return self._registry
