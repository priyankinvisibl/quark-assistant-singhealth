from dataclasses import dataclass

from src.config.types import CLI, Config, User
from src.internal.storage.opensearch import client as opensearch
from src.internal.storage.s3 import client as s3
from src.internal.storage.memory import InMemoryClient
from src.internal.storage.types import Storage


@dataclass(frozen=True, slots=True)
class State:
    cli: CLI
    config: Config
    storages: dict[str, Storage]
    # NOTE: tmp
    users: list[User]

    @classmethod
    def from_config(cls, cli: CLI) -> None:
        cli = cli
        config = Config.from_yaml_file(cli.app_config)
        storages = {}
        if cli.storages and config.storages:
            for _type, storage_conf in config.storages.items():
                if _type == "opensearch" and storage_conf.enabled:
                    # Use in-memory storage instead of OpenSearch
                    storages[_type] = InMemoryClient(storage_conf)
                if _type == "s3" and storage_conf.enabled:
                    storages[_type] = s3.QClient(storage_conf)
        
        # Create mock users for in-memory storage
        users = [
            User("suryaa@invisibl.io", "Suryaa"),
            User("admin@invisibl.io", "Admin"),
        ]
        return cls(cli, config, storages, users)

    def get_storage(self, storage: str) -> Storage:
        try:
            return self.storages[storage]
        except KeyError as e:
            raise KeyError(f"missing storage: {storage}") from e
