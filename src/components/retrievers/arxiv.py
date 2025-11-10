import arxiv
from arxiv import Result
from haystack import Document, component


@component
class ArxivRetriever:
    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        max_results=3,
        client: arxiv.Client | None = None,
    ) -> list[Document]:
        if not client:
            client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = client.results(search)
        documents = [self._to_document(result) for result in results]
        return documents

    def _to_document(self, result: Result) -> Document:
        doc = result.__dict__
        return Document(content=doc.pop("summary"), meta=doc)
