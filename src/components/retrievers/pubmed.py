import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

import xmltodict
from haystack import Document, component


# TODO: integrate with tools.eutils
@component
class PubMedRetriever:
    parse: Any
    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    max_retry: int = 5
    sleep_time: float = 0.2
    top_k_results: int = 3

    @component.output_types(documents=list[Document])
    def run(self, input: str) -> list[Document]:
        return {"documents": list(self.lazy_load_docs(query=input))}

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        for d in self.lazy_load(query=query):
            yield self._dict2document(d)

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id="
            + uid
            + "&webenv="
            + webenv
        )

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = xmltodict.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _dict2document(self, doc: dict) -> Document:
        summary = doc.pop("Summary")
        return Document(content=summary, meta=doc)

    def lazy_load(self, query: str) -> Iterator[dict]:
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str({urllib.parse.quote(query)})
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
        )
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)

        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = [
            f"{txt['@Label']}: {txt['#text']}"
            for txt in abstract_text
            if "#text" in txt and "@Label" in txt
        ]
        summary = (
            "\n".join(summaries)
            if summaries
            else (
                abstract_text
                if isinstance(abstract_text, str)
                else (
                    "\n".join(str(value) for value in abstract_text.values())
                    if isinstance(abstract_text, dict)
                    else "No abstract available"
                )
            )
        )
        a_d = ar.get("ArticleDate", {})
        pub_date = "-".join(
            [a_d.get("Year", ""), a_d.get("Month", ""), a_d.get("Day", "")]
        )

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Copyright Information": ar.get("Abstract", {}).get(
                "CopyrightInformation", ""
            ),
            "Summary": summary,
        }
