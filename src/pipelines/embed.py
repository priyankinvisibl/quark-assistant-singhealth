import mimetypes
import os

from haystack import Pipeline
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ByteStream

from src.config.types import Embedding
from src.internal.storage.types import KnowledgeStoreClient, Storage


class Embed:
    def __init__(self, ks: KnowledgeStoreClient, storage: Storage):
        self.storage = storage
        self.ks_client = ks

    # TODO: Add support for EFS, FTP, etc through datalocations and context
    def embed_file(
        self,
        filename: str,
        user: str,
        project: str,
        knowledgebase: str,
        et: str,
    ):
        store = self.storage.stores["myfiles"].format(user=user, project=project)
        data = self.storage.get_record(store, filename)
        fn, ft = os.path.splitext(filename)
        index_ = f"embeddings-{fn.lower()}-{user}"
        ds = self.ks_client.get_document_store(store=index_)
        embedder = self.ks_client.get_embedder(et)
        # TODO: move to end after opensearch plugin is fixed
        embedding = Embedding(index_, filename, f"{store}/{filename}", et)
        kb = self.ks_client.get_knowledgebase(knowledgebase)
        kb.embeddings.append(embedding)
        kb = self.ks_client.update_knowledgebase(kb.id, kb)
        ##TILL HERE
        pipe = Pipeline()
        pipe.add_component("filetype-router", FileTypeRouter(["application/pdf"]))
        pipe.add_component("pdf-converter", PDFMinerToDocument())
        pipe.add_component("cleaner", DocumentCleaner())
        pipe.add_component(
            "splitter", DocumentSplitter(split_length=100, split_overlap=20)
        )
        pipe.add_component("embedder", embedder)
        pipe.add_component("writer", DocumentWriter(document_store=ds))
        pipe.connect("filetype-router.application/pdf", "pdf-converter")
        pipe.connect("pdf-converter", "cleaner")
        pipe.connect("cleaner", "splitter")
        pipe.connect("splitter", "embedder")
        pipe.connect("embedder", "writer")
        pipe.run(
            {"sources": [ByteStream(data, mime_type=mimetypes.types_map[ft])]},
        )
        return kb
