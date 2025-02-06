from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    def __init__(self, path):
        """
        Constructor for NotFoundError
        :param path: The path that was not found
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):

    @abstractmethod
    def save(self, data: bytes, path: str):
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str):
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):

    def __init__(self, base_path: str = "./assets"):
        """
        Initialize a LocalStorage object.

        Args:
            base_path (str): Base directory where assets are stored.
            Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str):
        """
        Save data to a given path

        Args:
            data (bytes): Data to save
            key (str): Key to save data
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given key

        Args:
            key (str): Key to load data

        Returns:
            bytes: Loaded data
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/"):
        """
        Delete the file at the specified key.

        Args:
            key (str): Key representing the path to delete.
            Defaults to root ("/").
        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all keys under a given prefix.

        Args:
            prefix (str): Key to list under. Defaults to root ("/").

        Returns:
            List[str]: List of keys under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for
                p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str):
        """
        Assert that a given path exists.

        Args:
            path (str): Path to assert

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        # Ensure paths are OS-agnostic
        """
        Join the given path with the base path

        Args:
            path (str): Path to be joined with the base path.

        Returns:
            str: Normalized path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
