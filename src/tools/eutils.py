import xmltodict
import yaml
from Bio import Entrez

DB = ["gene", "snp", "omim"]


def esearch(db: str, query: str):
    if db not in DB:
        return "db is not valid"
    handle = Entrez.esearch(
        db=db, term=query, retmax=5, retmode="xml", sort="relevance"
    )
    try:
        records = Entrez.read(handle)
        handle.close()
        return ",".join(records["IdList"])
    except Exception() as e:
        return e


def efetch(db: str, id: str):
    if db not in DB:
        return "db is not valid"
    handle = Entrez.efetch(db=db, id=id, retmax=5, retmode="text")
    try:
        result = handle.read()
        handle.close()
        return result
    except Exception() as e:
        return e


def esummary(db: str, id: list[int]):
    if db not in DB:
        return "db is not valid"
    handle = Entrez.esummary(db=db, id=id, retmax=5, retmode="xml")
    try:
        result = Entrez.read(handle)
        records = xmltodict.parse()
        handle.close()
        return yaml.dump(records)
    except Exception() as e:
        return e
