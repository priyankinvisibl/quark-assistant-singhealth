import logging
from textwrap import dedent
from typing import Any

import boto3
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

from src.components.agents.helpers import get_boto3session_haystack

from ...config.types import Config


class Gremlin:
    """Gremlin-related pipeline components."""

    def __init__(
        self, schema_context: str, document_store: InMemoryDocumentStore, config: Config
    ):
        self.schema_context = schema_context
        self.document_store = document_store

        self.rag_pipeline_tool = Tool(
            name="rag_pipeline_tool",
            description="Get gremlin query results",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The query to use in the search. Infer this from the "
                            "user's message. It should be a question or a statement"
                        ),
                    }
                },
                "required": ["query"],
            },
            function=self._rag_pipeline_func,
        )

        # Use the host's creds for the Gremlin queries.
        kg_settings = config.knowledge_graph
        raise_exc = True
        endpoint = ""
        if kg_settings is not None:
            gtex_settings = kg_settings.gtex
            if gtex_settings is not None:
                graph_db_settings = gtex_settings.graph_db
                if (
                    graph_db_settings is not None
                    and graph_db_settings.endpoint is not None
                    and graph_db_settings.endpoint
                ):
                    endpoint = graph_db_settings.endpoint
                    raise_exc = False
        if raise_exc:
            raise Exception("GTEx graph DB endpoint not set")

        neptune_kwargs: dict[str, str | bool] = {
            "service_name": "neptunedata",
            "endpoint_url": endpoint,
        }
        if "localhost" in endpoint or "0.0.0.0" in endpoint or "127.0.0.1" in endpoint:
            neptune_kwargs["verify"] = False
        self.neptune_client = boto3.client(**neptune_kwargs)

        # Use cross-account permissions if needed.
        self.creds_kwargs = {}
        if config is not None:
            self.creds_kwargs.update(
                get_boto3session_haystack(config.models.get("aws", {}))
            )

    def _rag_pipeline_func(self, query: str):
        logging.info("rag_pipeline function query: %s", query)
        template = [
            ChatMessage.from_user(
                dedent(
                    """
                    You are an expert gremlin AI assistant . Answer the user's question based **only** on the provided context. If the context does not contain relevant information, **do not make up an answer**. Instead, say, "I don't have enough information to answer this."
                    
                    ### **Context:**
                    {% for document in documents %}
                        - {{ document.content }}
                    {% endfor %}

                    ### **Question:**
                    {{ question }}

                    ### **Guidelines:**
                    - Provide a clear, concise, and fact-based answer.
                    - Be aware of the fact that the neptune database has millions of edges and nodes and we want to keep the data scanned by the query to a minimum.
                    - If the context provides conflicting information, mention it explicitly.
                    - Do not make up answers. Only use the provided context.
                    - Do not add any external knowledge.
                    - Do not substitute node labels unless explicitly stated.
                    - Keep in mind to use id and not preferred_id when finding the nodes and edges
                    - Only respond with the Gremlin query, with no additional information, introduction, preamble, or post-amble.
                    - If comparing properties like read-counts, treat numeric values as strings, e.g., use gt('100') instead of gt(100).
                    - Avoid using `where(without(...))` or `where(within(...))` for filtering; instead, use `has('property', within(...))` or `has('property', without(...))` for compatibility with AWS Neptune's supported operations.
                    - **Do not use `.where(neq(...).and(...))` or chained `.where()` filters with logical conditions.** Instead, use `.has('property', without([...]))` to exclude values.
                    - Use `has('property', within([...]))` to include a list of allowed values.
                    - **IMPORTANT: Whenever pathways are mentioned in any query, ALWAYS include Pmid if available else mention no pmid for that particular pathway

                    # Disease Enrichment
                    g.V().hasLabel('Gene').has('id', within('IL10', 'IFNG', 'FOXP3', 'CD8A', 'CD8B')).as('gene')
                    .in('StudyToDiseaseAssociation').hasLabel('DiseaseOrPhenotypicFeature').as('disease')
                    .select('gene', 'disease').by(valueMap('id', 'name')).by(valueMap('id', 'keywords'))

                    # Drug Enrichment  
                    g.V().hasLabel('Gene').has('id', within('IL10', 'IFNG', 'FOXP3', 'CD8A', 'CD8B')).as('gene')
                    .in('DrugToGeneAssociation').hasLabel('Drug').as('drug')
                    .select('gene', 'drug').by(valueMap('id', 'name')).by(valueMap('id', 'name', 'concept_id'))

                    # Pathway Enrichment (current - but enhance)
                    g.V().hasLabel('Gene').has('id', within('IL10', 'IFNG', 'FOXP3', 'CD8A', 'CD8B')).as('gene')
                    .outE('Gene_pathway_association').inV().hasLabel('Pathway').as('pathway')
                    .select('gene', 'pathway').by(valueMap('id', 'name')).by(valueMap('id', 'name', 'reactome_id'))


                    ### **sample queries and their solutions:**
                    -query = "Perform enrichment analysis for IL10, IFNG, FOXP3, CD8A and CD8B genes"
                              "enrichment analysis for genes IL10, IFNG, FOXP3, CD8A, CD8B"
                              "analyze genes IL10, IFNG, FOXP3, CD8A, CD8B"
                    -solution = g.V().hasLabel('Gene').has('id', within('IL10', 'IFNG', 'FOXP3', 'CD8A', 'CD8B')).as('gene')
                        .union(
                            __.outE('Gene_pathway_association').inV().hasLabel('Pathway').as('pathway')
                            .select('gene', 'pathway').by(valueMap('id', 'name')).by(valueMap('id', 'name', 'reactome_id')),
                            __.inE('DrugToGeneAssociation').outV().hasLabel('Drug').as('drug')
                            .select('gene', 'drug').by(valueMap('id', 'name')).by(valueMap('id', 'name', 'concept_id')),
                            __.inE('MacromolecularMachineToCellularComponentAssociation').outV().hasLabel('CellularComponent').as('location')
                            .select('gene', 'location').by(valueMap('id', 'name')).by(valueMap('id', 'name')),
                            __.inE('MacromolecularMachineToMolecularActivityAssociation').outV().hasLabel('BiologicalProcess').as('function')
                            .select('gene', 'function').by(valueMap('id', 'name')).by(valueMap('id', 'name')),
                            __.outE('Gene_pathway_association').inV().hasLabel('Pathway').as('pathway')
                            .outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                            .outE('Reaction_pmid_association').inV().hasLabel('Pmid').as('pmid')
                            .select('gene', 'pathway', 'pmid').by(valueMap('id', 'name')).by(valueMap('id', 'name', 'reactome_id')).by(valueMap('id', 'pmid'))
                        ))

                    -query = "a gremlin query to get list of genes associated or involved with tissue sample with id GTEX-1117F-0526-SM-5EGHJ and no other details?"
                            "list of genes expressed in the tissue sample with id GTEX-1117F-0526-SM-5EGHJ and no other details?"
                    -solution = "g.V('GTEX-1117F-0526-SM-5EGHJ').in('MaterialSampleToEntityAssociation').hasLabel('Gene').values('id')"
                    -query = "a gremlin query to get tissue sample ids where the tissue type is Breast ,donors are female  ,tissue sample are associated with Gene id BRCA1 ?"
                            "list of tissue samples from female donores with tissue type Breast and have expression of gene BRCA1 ?"
                    -solution = g.V().hasLabel('MaterialSample').has('SMTS', 'Breast').as('sample').out('CaseToEntityAssociation').has('sex', 'female').as('donor').select('sample').in('MaterialSampleToEntityAssociation').hasLabel('Gene').has('id', 'BRCA1').select('sample').values('id')
                    -query = "can you give me a gremlin query to get list of female tissue samples donors , death reason slow death and AGE group is 60-69?"
                            "list of female sample donors with death reason slow death and age group 60-69?"
                    -solution = "g.V().hasLabel('Case').has('sex', 'female').has('DTHHRDY', 'Slow death').has('AGE', '60-69').values('id')"
                    -query = "What are the sample IDs where the tissue type is breast and the read-count expression of the genes CD274 and PDCD1 is greater than '100', where the patients are female and in the age group 50-59 years?"
                    -solution = g.V().hasLabel('MaterialSample').has('SMTS', 'Breast').as('sample').out('CaseToEntityAssociation').has('sex', 'female').has('AGE', '50-59').as('donor').select('sample').in('MaterialSampleToEntityAssociation').hasLabel('Gene').or(__.has('id', 'CD274'), __.has('id', 'PDCD1')).as('gene').outE('MaterialSampleToEntityAssociation').has('read-counts', gt('100')).inV().where(eq('sample')).dedup().values('id')
                    -query= "Summarize the molecular functions associated with the genes CD79A, CD79B, CD19, CD22, and CD37."
                    -solution = g.V().hasLabel('Gene').has('id', within('CD79A', 'CD79B', 'CD19', 'CD22', 'CD37')).in('MacromolecularMachineToMolecularActivityAssociation').hasLabel('MolecularActivity').valueMap('id', 'name')
                    -query = "for a list of genes GENE_IDS, get the mapping to their annotations via edges like ANNOTATION_EDGES and return both gene and annotation information?"
                    -solution = g.V().hasLabel('Gene').has('id', within(GENE_IDS)).as('gene') \
                    .union( \
                        __.inE('MacromolecularMachineToMolecularActivityAssociation').outV().hasLabel('MolecularActivity'), \
                        __.inE('MacromolecularMachineToCellularComponentAssociation').outV().hasLabel('CellularComponent'), \
                        __.inE('PathwayGeneInteraction').outV().hasLabel('Pathway') \
                    ).as('annotation') \
                    .select('gene', 'annotation').by(valueMap('id', 'name')).limit(10)

                    -query: "Given the following context from a biological knowledge graph, summarize the roles of the genes FOXP3, IL2RA, and CTLA4 in immune tolerance and regulatory T cell development."
                    -solution: g.V().hasLabel('Gene').has('id', within('FOXP3', 'IL2RA', 'CTLA4')).as('gene')
                                .union(
                                    __.inE('MacromolecularMachineToMolecularActivityAssociation').outV().hasLabel('MolecularActivity'),
                                    __.inE('MacromolecularMachineToCellularComponentAssociation').outV().hasLabel('CellularComponent'),
                                    __.inE('PathwayGeneInteraction').outV().hasLabel('Pathway')
                                ).as('annotation')
                                .select('gene', 'annotation')
                                .by(valueMap('id', 'name'))
                                .limit(10)

                    -query = "give me any one pathway and pmids associated with it"
                            "get one pathway with its pubmed ids"
                            "show me a pathway and its associated pmids"
                    -solution = g.V().hasLabel('Pathway')
                        .where(__.outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                        .outE('Reaction_pmid_association').inV().hasLabel('Pmid'))
                        .limit(1).as('pathway')
                        .outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                        .outE('Reaction_pmid_association').inV().hasLabel('Pmid').as('pmid')
                        .select('pathway', 'pmid').by(valueMap('id', 'name')).by(valueMap('id', 'pmid'))

                    -query = "give me pmids associated with R-HSA-425410"
                            "what are the pubmed ids for pathway R-HSA-425410"
                            "show pmids for reactome pathway R-HSA-425410"
                    -solution = g.V().hasLabel('Pathway').has('id', 'R-HSA-425410')
                        .outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                        .outE('Reaction_pmid_association').inV().hasLabel('Pmid').valueMap('id', 'pmid')

                    -query = "show me all pathways"
                            "list pathways in the database"
                            "get pathway information"
                            "how many pathways are there"
                            "what pathways are involved with CD8A"
                            "pathways for gene BRCA1"
                    -solution = g.V().hasLabel('Pathway')
                        .where(__.outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                        .outE('Reaction_pmid_association').inV().hasLabel('Pmid')).as('pathway')
                        .outE('Pathway_reaction_association').inV().hasLabel('Reaction')
                        .outE('Reaction_pmid_association').inV().hasLabel('Pmid').as('pmid')
                        .select('pathway', 'pmid').by(valueMap('id', 'name', 'reactome_id')).by(valueMap('id', 'pmid')).limit(10)

                    -query = "check if pathways exist"
                            "count pathways"
                            "do we have pathway data"
                    -solution = g.V().hasLabel('Pathway').count()

                    -query = "check if genes IL10, IFNG, FOXP3, CD8A, CD8B exist"
                            "do these genes exist in database"
                            "verify gene presence"
                    -solution = g.V().hasLabel('Gene').has('id', within('IL10', 'IFNG', 'FOXP3', 'CD8A', 'CD8B')).valueMap('id', 'name')
                    ### **Answer:**
                    """  # noqa: E501
                ).strip()
            )
        ]
        # Use cross-account permissions if needed.
        generator_kwargs = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "generation_kwargs": {"temperature": 0.0},
        }
        generator_kwargs.update(self.creds_kwargs)
        generator = AmazonBedrockChatGenerator(**generator_kwargs)

        rag_pipe = Pipeline()
        rag_pipe.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
        rag_pipe.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=self.document_store)
        )
        rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
        rag_pipe.add_component("llm", generator)

        rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
        rag_pipe.connect("retriever", "prompt_builder.documents")
        rag_pipe.connect("prompt_builder.prompt", "llm.messages")
        result = rag_pipe.run(
            {"embedder": {"text": query}, "prompt_builder": {"question": query}}
        )
        rag_response = result["llm"]["replies"][0].text
        print(f"🔍 RAG PIPELINE RESPONSE: {rag_response}")
        logging.info("RAG PIPELINE RESPONSE: %s", rag_response)
        return {"reply": rag_response}

    # Judge prompt builder
    def _build_judge_prompt(self, user_query, gremlin_query):
        return [
            ChatMessage.from_system(
                "You are a Gremlin query validator. Respond with 'Yes' or 'No' only."
            ),
            ChatMessage.from_user(
                dedent(
                    f"""
                    ### User Query:
                    {user_query}

                    ### Generated Gremlin Query:
                    {gremlin_query}

                    ### Schema Context:
                    {self.schema_context}

                    Does the Gremlin query correctly fulfill the user's request?  if the answer is "No" also mention why the it wont fullfill the user request. only mention the reason with no.
                    """  # noqa: E501
                ).strip()
            ),
        ]

    # Judge function
    def _judge_gremlin_query(self, judge_llm, user_query, gremlin_query):
        messages = self._build_judge_prompt(user_query, gremlin_query)
        result = judge_llm.run(messages)

        logging.debug("Judge Result: %s", result)

        try:
            response_text = result["replies"][0]._content[0].text.strip()
            lowered = response_text.lower()

            if lowered.startswith("yes"):
                return True
            elif lowered.startswith("no"):
                # Extract reason after "No." or "No:"
                reason = response_text[response_text.find(".") + 1 :].strip()
                return False, reason
        except (KeyError, IndexError, AttributeError):
            pass

        return False, "Could not extract a valid judgment."

    def _generate_and_validate_query(self, user_query, rag_pipeline_tool, judge_llm):
        # First try
        response = rag_pipeline_tool.function(user_query)
        gremlin_query = response["reply"].strip()
        logging.info("Gremlin Query: %s", gremlin_query)

        # Judge first attempt
        first_judgment = self._judge_gremlin_query(judge_llm, user_query, gremlin_query)
        if first_judgment is True:
            return gremlin_query
        elif isinstance(first_judgment, tuple):
            _, reason = first_judgment
            # Add reason to user_query for retry
            user_query += (
                f" A previous try failed because of the following reason: {reason}"
            )

        # Retry once with updated user query
        response_retry = rag_pipeline_tool.function(user_query)
        gremlin_query_retry = response_retry["reply"].strip()
        logging.info("Retry Gremlin Query: %s", gremlin_query_retry)

        if self._judge_gremlin_query(judge_llm, user_query, gremlin_query_retry):
            return gremlin_query_retry

        return "ERROR: Could not generate a valid Gremlin query for this request."

    def generate_query(self, question: str) -> str:
        """Generate a Gremlin query from a question."""
        print(f"🔍 GENERATING QUERY FOR: {question}")
        # Use cross-account permissions if needed.
        generator_kwargs = {
            "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "generation_kwargs": {"temperature": 0.0},
        }
        generator_kwargs.update(self.creds_kwargs)
        judge_llm = AmazonBedrockChatGenerator(**generator_kwargs)
        generated_query = self._generate_and_validate_query(
            question, self.rag_pipeline_tool, judge_llm
        )
        print(f"🔍 FINAL GENERATED GREMLIN QUERY: {generated_query}")
        logging.info("FINAL GENERATED GREMLIN QUERY: %s", generated_query)
        return generated_query

    def get_db_response(self, query: str) -> str | dict[str, Any]:
        """Get the response for a query from the Neptune DB."""
        print(f"🔍 EXECUTING NEPTUNE QUERY: {query}")
        response = self.neptune_client.execute_gremlin_query(
            gremlinQuery=query, serializer="application/json"
        )
        print(f"🔍 NEPTUNE RAW RESPONSE: {response}")
        return response
