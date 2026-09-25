"""Named MongoDB snapshots: explicit writes, no collection drops or implicit replacement."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone

from pymongo import ASCENDING, GEOSPHERE, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from .data import Dataset
from .errors import DatabaseError, DataError

COLLECTIONS = {"node": "Nodes", "way": "Ways", "relation": "Relations"}


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name):
        raise DataError("Names must contain only letters, digits, underscores and hyphens (1-100 characters)")
    return name


@contextmanager
def connect(database: str = "OSMResearch"):
    validate_name(database)
    if database.lower() in {"admin", "local", "config"}:
        raise DataError("Choose a dedicated research database, not a MongoDB system database")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise DatabaseError("Set MONGODB_URI in your environment; it is never read from committed source")
    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000, socketTimeoutMS=60000)
        client.admin.command("ping")
        yield client[database]
    except PyMongoError:
        raise DatabaseError("MongoDB operation failed. Check credentials, TLS, network access, indexes and permissions.") from None
    finally:
        if client is not None:
            client.close()


def create_indexes(database) -> None:
    for collection in COLLECTIONS.values():
        database[collection].create_index([("dataset_id", ASCENDING), ("id", ASCENDING)], unique=True)
    database.Nodes.create_index([("location", GEOSPHERE)])
    database.Ways.create_index([("dataset_id", ASCENDING), ("nodes", ASCENDING)])
    database.Relations.create_index([("dataset_id", ASCENDING), ("members.ref", ASCENDING)])


def import_dataset(dataset: Dataset, database, dataset_id: str, *, write: bool = False, batch_size: int = 1000) -> dict:
    validate_name(dataset_id)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise DataError("batch_size must be a positive integer")
    dataset.validate()
    graph = dataset.graph()  # Reject unusable topology before any MongoDB write.
    if not graph.edges:
        raise DataError("The dataset has no routing edges")
    report = {"dataset_id": dataset_id, "counts": dataset.counts(), "dataset_sha256": dataset.fingerprint(),
              "directed_edges": len(graph.edges), "write": write, "metadata": dataset.metadata}
    if not write:
        return report  # Works with database=None and no database credentials.
    try:
        if database.Datasets.find_one({"_id": dataset_id}) or any(
            database[name].find_one({"dataset_id": dataset_id}, {"_id": 1}) for name in COLLECTIONS.values()
        ):
            raise DataError("Dataset ID already exists. Choose a new snapshot ID; no data was overwritten")
        create_indexes(database)
        # Claim a unique snapshot name before bulk inserts. Readers accept ready only.
        database.Datasets.insert_one({"_id": dataset_id, **report, "status": "loading",
                                      "created_utc": datetime.now(timezone.utc).isoformat()})
    except DuplicateKeyError:
        raise DataError("Snapshot name or unique index already exists; choose a new dedicated snapshot") from None
    except PyMongoError:
        raise DatabaseError("Cannot initialize the snapshot. Check write privileges and research-database indexes") from None
    try:
        for documents, name in ((dataset.nodes, "Nodes"), (dataset.ways, "Ways"), (dataset.relations, "Relations")):
            for offset in range(0, len(documents), batch_size):
                batch = [{**document, "dataset_id": dataset_id} for document in documents[offset:offset + batch_size]]
                if batch:
                    database[name].insert_many(batch, ordered=True)
        for name, count in report["counts"].items():
            if database[name.capitalize()].count_documents({"dataset_id": dataset_id}) != count:
                raise DataError("Snapshot count verification failed")
        database.Datasets.update_one({"_id": dataset_id, "status": "loading"}, {"$set": {"status": "ready"}})
        return report | {"status": "ready"}
    except (PyMongoError, DataError):
        try:
            database.Datasets.update_one({"_id": dataset_id}, {"$set": {"status": "failed"}})
        except PyMongoError:
            pass
        raise DatabaseError("Snapshot import did not complete. Partial data was retained for inspection; use a new ID after resolving the cause") from None


def load_dataset(database, dataset_id: str) -> Dataset:
    validate_name(dataset_id)
    try:
        manifest = database.Datasets.find_one({"_id": dataset_id, "status": "ready"})
        if manifest is None:
            raise DataError("Dataset is absent or not ready. Import a complete research snapshot first")

        def elements():
            for kind, collection in COLLECTIONS.items():
                for document in database[collection].find({"dataset_id": dataset_id}, {"_id": 0, "dataset_id": 0}):
                    yield {**document, "type": kind}

        dataset = Dataset.from_elements(elements(), manifest.get("metadata", {}) | {"dataset_id": dataset_id})
        if dataset.counts() != manifest["counts"] or dataset.fingerprint() != manifest["dataset_sha256"]:
            raise DataError("Snapshot fingerprint/count mismatch: stored data changed after import")
        return dataset
    except PyMongoError:
        raise DatabaseError("Cannot read research snapshot. Check connection and read permissions") from None