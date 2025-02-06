import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile


class TestDatabase(unittest.TestCase):

    def setUp(self):
        """
        Set up test environment with a temporary LocalStorage and Database.

        Creates a temporary directory for LocalStorage and initializes a
        Database object with this storage. This setup is used for
        testing purposes to ensure isolation and cleanup after tests.
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        """
        Tests that the object created is an instance of Database.
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self):
        """
        Tests that the set method stores data correctly.

        Verifies that the set method stores an entry in the database and
        that the entry can be retrieved by the get method.
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self):
        """
        Tests that the delete method deletes the specified entry
        from the database.

        Tests that the delete method removes the specified entry from the
        database and that the entry can no longer be retrieved by the get
        method. Also tests that the database is refreshed after deletion.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self):
        """
        Tests that the data is persisted to storage and can be retrieved by
        a new instance of the Database class.

        Verifies that the data is stored to the storage and that it can be
        retrieved by a new instance of the Database class.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self):
        """
        Tests that the refresh method loads the data from storage.

        Verifies that the refresh method loads the data from storage and
        that the data can be retrieved by the get method.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        """
        Tests that the list method returns a list of all entries in
        the specified collection.

        Verifies that the list method returns a list of tuples,
        where each tuple
        contains the id and data of an entry in the specified collection.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))
