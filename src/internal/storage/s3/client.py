import os
from typing import override

import s3fs

from src.config.types import StorageConfig
from src.internal.storage.types import Storage


# TODO: error handling
class QClient(Storage):
    def __init__(
        self,
        config: StorageConfig | None = None,
        client: s3fs.S3FileSystem | None = None,
    ) -> None:
        self.config = config.config
        if not client:
            self._client = s3fs.S3FileSystem(**config.config)
        self.stores = {k: v for k, v in config.stores.items()}

    @property
    def client(self) -> s3fs.S3FileSystem:
        return self._client

    # TODO: handle interpolation better
    @override
    def get_record(self, store: str, id_: str):
        path = os.path.join(store.removesuffix(id_), id_)
        with self.client.open(path, "rb") as f:
            data = f.read()
        return data

    @override
    def get_records(self, store: str):
        pass

    @override
    def store_exists(self, store: str):
        return self.client.exists(store)

    @override
    def add_record(self, store: str, id: str, body: dict[str, any]):
        raise PermissionError("out of application scope")

    @override
    def delete_record(self, store: str, id_: str):
        raise PermissionError("out of application scope")

    @override
    def upsert_record(self, store: str, id_: str, body: dict[str, any]):
        raise PermissionError("out of application scope")
