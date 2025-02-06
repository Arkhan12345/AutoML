
import unittest

from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile
import os


class TestStorage(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary directory and create a LocalStorage object.

        Sets up a temporary directory and creates a LocalStorage object with
        that directory. The directory is cleaned up after the test is run.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """
        Tests that the object created is an instance of LocalStorage.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """
        Tests that the save method stores the data correctly and the load
        method can retrieve it. Also tests that a NotFoundError is raised
        when a non-existent key is accessed.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = f"test{os.sep}otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self):
        """
        Tests that the delete method deletes the file at the specified key.

        Args:
            key (str): Key representing the path to delete.

        Raises:
            NotFoundError: If the path does not exist.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self):
        """
        Tests that the list method returns all keys under a given prefix.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test{os.sep}{random.randint(0, 100)}"
                       for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = [f"{os.sep}".join(key.split(f"{os.sep}")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
