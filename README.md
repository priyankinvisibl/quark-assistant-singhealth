# Quark Assistant built using the Generic AI-Assistant Framework
## Features
### Threads(now Memories):
* User session separation
* Memory titles can be changed
* Memory summary can be found in metadata/can be triggered
* Memories can be downloaded
* Memories can be shared with other users - can we use other user share features in workspaces
* Message metadata contains attachments/context, path to attachment, and history/summary that was used to answer the prompt
* Message metadata also contains agent and model name
* Messages can have attachments(separate from knowledgebase); attachments can be:
  + Files(pictures, videos*, documents)
  + Public/accessible URLs
  + support for audio*
* Image output ( disclaimer - diagrams are not included yet )
* Group Memories*
* Messages can be flagged with positive/negative by the user*
* Messages can be questioned/replied to, taking context prior to it*
* Audit/Trace*

---
### Knowledgebases:
* Knowledgebase can be created from documents in myfiles/datalocations - knowledgebase/multiple files #p1
* Knowledgebase can be cycled(removed and added) from a given memory 
* Knowledgebase can be shared
* Knowledgebase catalog to show embeddings of all your added, shared and org documents
* Knowledgebase embeddings can be deleted #rbac*
* Knowledgebase embedding can be started as a background job if documents are huge*
* Build a knowledgegraph within a knowledgebase to aid in higher quality output*
* Knowledgebases can be visualized - Data in knowledgebases can be visualized as tables and graphs*
* Knowledgebases can be queried by agents*
* S3 as a document store and retriever with s3 select/Athena*
* Supported file types:
  + application/pdf (structured)
  + text/plain
  + text/markdown
----
### Major Agents:
* Generic Assistant:
  - GeneGPT AKA NCBI Agent:
    + Has now evolved into GeneLLM(it can use gemini, azure, aws) and will be called NCBI Agent
    + GeneHopping is now possible
    + Features: Gene-Disease, Gene-Location, Gene-Name, Gene-SNP, Gene-Alias, Gene-Name conversion, Human-Genome DNA Alignment, Multi-species DNA Alignment, Protein-codign genes, SNP Location, Disease-Gene, SNP-Gene
  - Quark Solutions:
    + Recommendation of available globaljobtemplates and globaltasktemplates*
    + Analytics agent over results for cohort comparsions*
  - Text-Gen
  - QA
  - RE-BC5CDR
  - RE-DDI
* Coding Assistant
  - CodeGen
  - Quark PipelineBuilder*
  - CodeCompletion*
  - AutoBA* - onhold
  
### Minor Agents/Models:
* Task Decider (can be replaced with a local BERT Classification Models)
* Memory Summarizer
* Validator*
* Evaluators/Enrichers*
* Nomenclator*
* ImageGenerator

## Design Changes
* General:
  + Separation of code and configuration
  + Models, prompts, embeddings, storages can be globally configured and stored
  + Vendors, models, prompt, tools can be plugged and played with agents through config file
* Haystack over Langchain:
  + Everything is a node and everything runs in a pipeline
  + Components can be very easily added and removed
  + Custom components can be easily built and controlled
  + Documentation is concise and clean
  + Optimization and Evaluation for RAG
  + Flexibility and performance benefits
  + Better logging
* Opensearch over Zep:
  + Audit
  + Better control
  + Flexibility with embeddings
  + Integrated with other platform resources
  + Summaries can be retrieved
  + Enables sharing and downloading
  
### TODO:
1. Including context of request to agent
2. Requesting specific agents
3. Quality and assurance of agent-models alongside benchmarks
4. Add support for all datalocations and validate files
5. Combine ChatMessage, and Message classes into one
6. Better prompt engineering
7. Add proper auth
8. Make agents more flexible to use context, history and tools
9. Vectorstore customizability
10. Work on agents(especially NCBI and quark solutions), understanding use cases; waiting on https://github.com/deepset-ai/haystack-core-integrations/issues/1022
11. Find valid use case for get_relevant_history
12. Evaluation of responses
~~13. Improve handling of db connections~~
14. Bring in Vertex AI Search and Conversation and Amazon Q
15. https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde
16. Chatting with your Vectorstore
17. User input Validation
18. One opensearch client for vector store, and memory store?
19. ReAct - https://react-lm.github.io/
20. Make pipelines even more dynamic.
21. Pipeline components need to be build from yaml config files
22. Make all calls to storage async
~~23. Generalize document store/ retrievers~~
24. Test with OpensearchEmbeddingRetriever
25. Update code after haystack v2.5 is released with subgraphs
26. Test routers with Message instead of dict
27. Automerging/Hierarchical RAG

'*' - indicates pending work for functionality


