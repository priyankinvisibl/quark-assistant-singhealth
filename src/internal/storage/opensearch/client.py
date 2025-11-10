from dataclasses import dataclass, field
from typing import Any, override

import boto3
import opensearchpy.exceptions as os_exceptions
from dataclass_wizard import JSONWizard, json_field
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection

from src.config.types import StorageConfig
from src.internal.storage.types import Storage


class MyConnection(RequestsHttpConnection):
    def __init__(self, *args, **kwargs):
        proxies = kwargs.pop("proxies", {})
        super(MyConnection, self).__init__(*args, **kwargs)
        self.session.proxies = proxies

@dataclass
class ResponseHitRecord(JSONWizard):
    index: str = json_field("_index")
    source: dict[str, Any] = json_field("_source")
    id: str = json_field("_id")
    score: float | None = json_field("_score", default=None)
    sort: list = json_field("_sort", default_factory=list)
    inner_hits: dict[str, Any] = json_field("inner_hits", default_factory=dict)


@dataclass
class ResponseHitTotal:
    value: int
    relation: str


@dataclass
class Aggregations:
    metrics: Any | None = json_field(["value", "values"], default=None)
    buckets: list[dict[str, Any]] = json_field(["buckets"], default_factory=list)
    after_key: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseHit:
    total: ResponseHitTotal
    max_score: float | None
    hits: list[ResponseHitRecord] = json_field("hits")


@dataclass
class Response(JSONWizard):
    took: int
    hits: ResponseHit
    scroll_id: str | None = json_field("_scroll_id", default=None)
    aggregations: dict[str, Aggregations] = field(default_factory=dict)


# TODO: validate if index exists
# integrate implicit support for multiple indicies
class QClient(Storage):
    def __init__(
        self,
        config: StorageConfig | None = None,
        client: OpenSearch | None = None,
    ) -> None:
        self.config = config.config
        self.indices = config.stores
        self._client = client

    @property
    def client(self) -> OpenSearch:
        if not self._client:
            self._client = OpenSearch(
                "https://vpc-gravity-dev-opensearch-twyyo3waeyvxtpvdenhkt4rrda.us-east-1.es.amazonaws.com",
                proxies={
                    "http": "socks5://localhost:3128",
                    "https": "socks5://localhost:3128",
                },
                timeout=60,
                connection_class=MyConnection,
            )
        return self._client
    # @property
    # def client(self) -> OpenSearch:
    #     if not self._client:
    #         connection_class = RequestsHttpConnection
    #         auth = self.config.pop("http_auth", None)
    #         if not auth:
    #             session = boto3.Session()
    #             region = session.region_name
    #             credentials = session.get_credentials()
    #             auth = RequestsAWSV4SignerAuth(credentials, region)
    #         self._client = OpenSearch(
    #             **self.config, http_auth=auth, connection_class=connection_class
    #         )
    #     return self._client

    @override
    def store_exists(self, store: str):
        self.client.indicies.exists(self.indices[store])

    @override
    def scroll(
        self,
        *stores: str,
        query: dict[str, Any] = {"size": 10000},
        scroll: str = "1m",
    ) -> Response | None:
        i = []
        for store in stores:
            try:
                i.append(self.indices[store])
            except KeyError as e:
                raise KeyError(f"missing store: {store}") from e
        indices = ",".join(i)
        try:
            result = self.client.search(
                body=query, index=indices, params={"scroll": scroll}
            )
            output = Response.from_dict(result)
            res = output
            while True:
                output = self.client.scroll(
                    scroll_id=output.scroll_id, params={"scroll": scroll}
                )
                output = Response.from_dict(output)
                if len(output.hits.hits) == 0:
                    break
                res.hits.hits += output.hits.hits
        except Exception as e:
            print(e)
            return None
        finally:
            try:
                self.client.clear_scroll(scroll_id=output.scroll_id)
            except Exception as e:
                print(e)
        return res

    @override
    def delete_record(self, store: str, id_: str):
        try:
            res = self.client.delete(self.indices[store], id_)
        except os_exceptions.NotFoundError:
            return None
        return res

    @override
    def delete_by_query(self, store: str, query: dict[str, Any]):
        res = self.client.delete_by_query(self.indices[store], query)
        return res

    @override
    def get_record(self, store: str, id_: str):
        try:
            res = self.client.get(index=self.indices[store], id=id_)
            return ResponseHitRecord.from_dict(res)
        except os_exceptions.NotFoundError:
            return None

    @override
    def upsert_record(self, store: str, id_: str, body: dict[str, any]):
        try:
            response = self.client.index(index=self.indices[store], id=id_, body=body)
        except os_exceptions.ConflictError:
            raise ValueError(
                "record already exists, embed doc_as_upsert with the body to update",
                body,
            )
        return None

    @override
    def get_records(
        self,
        store: str,
        body: dict[str, Any] = {"size": 10000},
    ) -> Response:
        import json

        print(json.dumps(body))
        resp = self.client.search(body=body, index=self.indices[store])
        return Response.from_dict(resp)
