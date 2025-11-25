import json
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd
import spacy
import yaml
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

from src.components.agents.helpers import get_boto3session_haystack
from src.config.types import Memory, Message
from src.gtex.pipeline import QueryEnhancerPipeline


@component
class PreProcessorComponent:
    def __init__(
        self,
        mem_client,
        config,
        schema_path,
        model_path,
        entities_path,
        entities_file_type,
    ):
        self.mem_client = mem_client
        self.config = config
        self.schema_path = schema_path
        self.model_path = model_path
        self.entities_path = entities_path
        self.entities_file_type = entities_file_type

    @component.output_types(action=str, message=Dict[str, Any], final_message=Message)
    def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        user_response = message["content"].strip()
        history, _ = self.mem_client.get_history(message["memory_id"])

        if not history:
            return {"action": "continue", "message": message, "final_message": None}

        last_msg = history[-1]

        # Check for pathway choice response
        if last_msg.role == "assistant" and last_msg.metadata.get("choice_request"):

            pathway_choices = last_msg.metadata.get("pathway_choices", [])
            pathway_ids = last_msg.metadata.get("pathway_ids", [])

            if user_response.isdigit():
                choice_num = int(user_response)
                if 1 <= choice_num <= len(pathway_choices):
                    chosen_pathway = pathway_choices[choice_num - 1]
                    chosen_pathway_id = (
                        pathway_ids[choice_num - 1]
                        if choice_num <= len(pathway_ids)
                        else chosen_pathway
                    )
                    print(
                        f"🔍 USER CHOSE PATHWAY: {choice_num} → '{chosen_pathway}' (ID: {chosen_pathway_id})"
                    )

                    # Get original query from history
                    original_query = ""
                    for msg in reversed(history):
                        if (
                            msg.role == "user"
                            and len(msg.content) > 10
                            and not msg.content.strip().isdigit()
                        ):
                            original_query = msg.content
                            break

                    # Process with chosen pathway
                    processor = GTExProcessorComponent(
                        self.config,
                        self.schema_path,
                        self.model_path,
                        self.entities_path,
                        self.entities_file_type,
                        self.mem_client,
                    )

                    original_message = {**message, "content": original_query}
                    # Pass original pathway info for correct replacement
                    original_pathway = last_msg.metadata.get("original_pathway", "")
                    result = processor.run(
                        {chosen_pathway_id: "Pathway"},
                        original_message,
                        original_pathway_name=original_pathway,
                    )
                    return {
                        "action": "handled",
                        "final_message": result["final_message"],
                    }

        # Check for entity validation response
        if last_msg.role == "assistant" and last_msg.metadata.get("validation_request"):

            stored_entities = last_msg.metadata.get("extracted_entities", {})
            user_response_lower = user_response.lower()

            if user_response_lower in ["yes", "y", "correct", "ok", "right", "good"]:
                # User approved entities - process with GTEx
                processor = GTExProcessorComponent(
                    self.config,
                    self.schema_path,
                    self.model_path,
                    self.entities_path,
                    self.entities_file_type,
                    self.mem_client,
                )

                # Get original query
                original_query = ""
                for msg in reversed(history):
                    if (
                        msg.role == "user"
                        and len(msg.content) > 10
                        and msg.content.lower() not in ["yes", "y", "no", "n"]
                    ):
                        original_query = msg.content
                        break

                original_message = {**message, "content": original_query}
                result = processor.run(stored_entities, original_message)
                return {"action": "handled", "final_message": result["final_message"]}

            elif user_response.startswith("no, entities are:"):
                # User corrected entities - re-extract
                entity_text = user_response.replace("no, entities are:", "").strip()
                corrected_message = {**message, "content": entity_text}
                return {
                    "action": "re_extract",
                    "message": corrected_message,
                    "final_message": None,
                }

            elif user_response_lower in ["no", "n"]:
                # User rejected but didn't provide corrections
                clarification_msg = "Please specify the correct entities: 'NO, entities are: [your entities]'"

                user_message = Message(
                    content=message["content"],  # Preserve original user input
                    origin=message["origin"],
                    memory_id=message["memory_id"],
                    role=message["role"],
                    metadata=message.get("metadata", {}),
                    timestamp=message.get("timestamp"),
                    message_id=message.get("message_id"),
                )
                assistant_message = Message(
                    content=clarification_msg,
                    origin=message["origin"],
                    memory_id=message["memory_id"],
                    role="assistant",
                    metadata={"clarification": True},
                )
                self.mem_client.add_messages(user_message)
                self.mem_client.add_messages(assistant_message)
                return {"action": "handled", "final_message": assistant_message}

        # No special handling needed - continue to supervisor
        return {"action": "continue", "message": message, "final_message": None}


@component
class SupervisorComponent:
    def __init__(self, config):
        # Use model from config instead of hardcoded
        generator_kwargs = {
            "model": config.models.get("aws", {}).get(
                "name", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
            ),
            "generation_kwargs": {"temperature": 0.0},
        }
        generator_kwargs.update(get_boto3session_haystack(config.models.get("aws", {})))
        self.llm = AmazonBedrockChatGenerator(**generator_kwargs)

        # Load supervisor prompt
        with open(config.paths["prompts"].format(prompt="pipeline-supervisor")) as f:
            prompt_data = yaml.safe_load(f)

        system_content = prompt_data["messages"][0]["content"]
        self.template = [
            ChatMessage.from_user(f"{system_content}\n\nUser query: {{content}}")
        ]
        self.prompt_builder = ChatPromptBuilder(
            template=self.template, required_variables=["content"]
        )

    @component.output_types(decision=str, message=Dict[str, Any])
    def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        # Build prompt and get LLM decision
        prompt_result = self.prompt_builder.run(content=message["content"])
        llm_result = self.llm.run(messages=prompt_result["prompt"])

        decision = llm_result["replies"][0].text.strip().lower()
        return {"decision": decision, "message": message}


@component
class StateCheckerComponent:
    def __init__(self, mem_client):
        self.mem_client = mem_client

    @component.output_types(state=str, message=Dict[str, Any], validation_attempt=int)
    def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        history, _ = self.mem_client.get_history(message["memory_id"])

        state = "new_query"
        validation_attempt = 1

        if history:
            # Check if this is a response to the most recent validation request
            last_msg = history[-1]
            if (
                last_msg.role == "assistant"
                and last_msg.metadata.get("validation_request")
                and len(message["content"].strip()) < 50
            ):  # Short responses are likely validation responses
                validation_attempt = last_msg.metadata.get("validation_attempt", 1)
                state = "validation_response"

        return {
            "state": state,
            "message": message,
            "validation_attempt": validation_attempt,
        }


@component
class EntityExtractorComponent:
    def __init__(
        self,
        model_path: str,
        entities_path: str,
        entities_file_type: str,
        mem_client,
        config=None,
        schema_path="",
    ):
        self.model_path = Path(model_path)
        self.entities_path = Path(entities_path)
        self.entities_file_type = entities_file_type
        self.mem_client = mem_client
        self.config = config
        self.schema_path = schema_path

    @component.output_types(final_message=Message)
    def run(
        self, message: Dict[str, Any], validation_attempt: int = 1
    ) -> Dict[str, Any]:
        try:
            # Step 1: Extract quoted pathways first
            import re

            content = message["content"]
            print(f"🔍 ORIGINAL CONTENT: {content}")

            # Extract quoted pathway names: pathway 'name' or pathway "name"
            pathway_pattern = r"pathway\s+['\"]([^'\"]+)['\"]"
            quoted_pathways = re.findall(pathway_pattern, content, re.IGNORECASE)
            print(f"🔍 QUOTED PATHWAYS FOUND: {quoted_pathways}")

            # Extract quoted disease names: disease 'name' or disease "name"
            disease_pattern = r"disease\s+['\"]([^'\"]+)['\"]"
            quoted_diseases = re.findall(disease_pattern, content, re.IGNORECASE)
            print(f"🔍 QUOTED DISEASES FOUND: {quoted_diseases}")

            # Remove quoted pathway and disease parts from content for spaCy processing
            clean_content = re.sub(pathway_pattern, "", content, flags=re.IGNORECASE)
            clean_content = re.sub(disease_pattern, "", clean_content, flags=re.IGNORECASE).strip()
            print(f"🔍 CLEAN CONTENT FOR SPACY: {clean_content}")

            # Step 2: Use spaCy on cleaned content to find genes
            nlp = spacy.load(self.model_path)
            raw_spacy_entities = [ent.text for ent in nlp(clean_content).ents]

            # Filter out common words that should never be genes
            common_words = {
                "tell",
                "associated",
                "with",
                "the",
                "and",
                "or",
                "of",
                "in",
                "to",
                "for",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "must",
                "shall",
                "this",
                "that",
                "these",
                "those",
                "a",
                "an",
                "what",
                "which",
                "who",
                "when",
                "where",
                "why",
                "how",
                "any",
                "some",
                "all",
                "each",
                "every",
                "no",
                "none",
                "both",
                "either",
                "neither",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "pathways",
                "pathway",
                "gene",
                "genes",
                "protein",
                "proteins",
            }

            spacy_entities = [
                ent for ent in raw_spacy_entities if ent.lower() not in common_words
            ]
            print(f"🔍 RAW SPACY ENTITIES: {raw_spacy_entities}")
            print(f"🔍 FILTERED SPACY ENTITIES: {spacy_entities}")

            # Step 3: Look for direct pathway IDs (R-HSA-XXXXX) in original content
            pathway_id_pattern = r"R-HSA-\d+"
            direct_pathway_ids = re.findall(pathway_id_pattern, content)
            print(f"🔍 DIRECT PATHWAY IDS: {direct_pathway_ids}")

            # Step 4: Combine all entities
            all_entities = spacy_entities + quoted_pathways + quoted_diseases + direct_pathway_ids
            print(f"🔍 ALL ENTITIES BEFORE TYPING: {all_entities}")

            # Step 5: Load pathway mapping for fuzzy matching and resolution
            pathway_mapping = {}
            try:
                import json

                mapping_path = self.entities_path / "../pathway_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path) as f:
                        pathway_mapping = json.load(f)
                    print(f"🔍 LOADED {len(pathway_mapping)} PATHWAY MAPPINGS")
            except Exception:
                pass

            # Step 6: Process quoted pathways with fuzzy matching
            resolved_pathways = []
            for quoted_pathway in quoted_pathways:
                print(f"🔍 PROCESSING QUOTED PATHWAY: '{quoted_pathway}'")

                # Try exact match first
                if quoted_pathway in pathway_mapping:
                    resolved_pathways.append(quoted_pathway)
                    print(
                        f"🔍 EXACT MATCH: '{quoted_pathway}' → {pathway_mapping[quoted_pathway]}"
                    )
                else:
                    # Find multiple fuzzy matches
                    from difflib import SequenceMatcher

                    best_match = None
                    best_ratio = 0.0

                    matches = []
                    for pathway_name in pathway_mapping.keys():
                        if not pathway_name.startswith("R-HSA-"):
                            ratio = SequenceMatcher(
                                None, quoted_pathway.lower(), pathway_name.lower()
                            ).ratio()
                            if ratio > 0.7:  # Lower threshold to catch more candidates
                                matches.append((pathway_name, ratio))

                    # Sort by similarity (best first)
                    matches.sort(key=lambda x: x[1], reverse=True)

                    if len(matches) == 1:
                        # Single clear match
                        best_match = matches[0][0]
                        resolved_pathways.append(best_match)
                        print(
                            f"🔍 SINGLE FUZZY MATCH: '{quoted_pathway}' → '{best_match}' (similarity: {matches[0][1]:.2f})"
                        )
                    elif len(matches) > 1:
                        # Multiple matches - ask user to choose
                        print(f"🔍 MULTIPLE MATCHES FOUND: {len(matches)} options")

                        # Create multiple choice prompt with all identified entities
                        # Get entity types first to filter properly
                        final_entities = (
                            spacy_entities + resolved_pathways + direct_pathway_ids
                        )
                        entities_to_types = self._get_entity_types(final_entities)

                        # Show only actual genes identified
                        genes_found = [
                            entity
                            for entity, etype in entities_to_types.items()
                            if etype == "Gene"
                        ]
                        genes_display = (
                            f"Genes: {', '.join(genes_found)}"
                            if genes_found
                            else "Genes: None"
                        )

                        # Show pathway matches with their IDs
                        pathway_matches = []
                        pathway_ids = []  # Store IDs separately
                        for i, (pathway_name, similarity) in enumerate(matches[:5], 1):
                            pathway_id = pathway_mapping.get(pathway_name, "Unknown ID")
                            pathway_matches.append(
                                f"{i}. {pathway_name} (ID: {pathway_id})"
                            )
                            pathway_ids.append(pathway_id)

                        choice_prompt = (
                            f"**Identified Entities:**\n{genes_display}\n\n**Multiple pathway matches found for '{quoted_pathway}':**\n"
                            + "\n".join(pathway_matches)
                            + f"\n\nWhich pathway did you mean? Reply with the number (1-{min(5, len(matches))}) or type the exact pathway name."
                        )

                        # Return early with choice prompt
                        user_message = Message(**message)
                        assistant_message = Message(
                            content=choice_prompt,
                            origin=message["origin"],
                            memory_id=message["memory_id"],
                            role="assistant",
                            metadata={
                                "pathway_choices": [match[0] for match in matches[:5]],
                                "pathway_ids": pathway_ids,  # Store corresponding IDs
                                "original_pathway": quoted_pathway,
                                "choice_request": True,
                            },
                        )
                        self.mem_client.add_messages(user_message)
                        self.mem_client.add_messages(assistant_message)
                        return {"final_message": assistant_message}
                    else:
                        # No matches found
                        resolved_pathways.append(quoted_pathway)  # Keep original
                        print(
                            f"🔍 NO FUZZY MATCH FOUND: Keeping '{quoted_pathway}' as-is"
                        )

            # Step 6.5: Process quoted diseases with fuzzy matching
            resolved_diseases = []
            disease_mapping = {}
            try:
                import json
                disease_mapping_path = self.entities_path / "../disease_mapping.json"
                if disease_mapping_path.exists():
                    with open(disease_mapping_path) as f:
                        disease_mapping = json.load(f)
                    print(f"🔍 LOADED {len(disease_mapping)} DISEASE MAPPINGS")
            except Exception:
                pass

            for quoted_disease in quoted_diseases:
                print(f"🔍 PROCESSING QUOTED DISEASE: '{quoted_disease}'")

                # Try exact match first
                if quoted_disease in disease_mapping:
                    resolved_diseases.append(disease_mapping[quoted_disease])  # Use ID, not name
                    print(f"🔍 EXACT MATCH: '{quoted_disease}' → {disease_mapping[quoted_disease]}")
                else:
                    # Find fuzzy matches
                    from difflib import SequenceMatcher
                    matches = []
                    for disease_name in disease_mapping.keys():
                        if not disease_name.startswith(("OMIM:", "ORPHA:")):
                            ratio = SequenceMatcher(None, quoted_disease.lower(), disease_name.lower()).ratio()
                            if ratio > 0.7:
                                matches.append((disease_name, ratio))

                    matches.sort(key=lambda x: x[1], reverse=True)
                    if matches:
                        best_match = matches[0][0]
                        resolved_diseases.append(disease_mapping[best_match])  # Use ID, not name
                        print(f"🔍 FUZZY MATCH: '{quoted_disease}' → '{best_match}' (ID: {disease_mapping[best_match]})")
                    else:
                        resolved_diseases.append(quoted_disease)
                        print(f"🔍 NO FUZZY MATCH FOUND: Keeping '{quoted_disease}' as-is")

            # Step 7: Combine final entities
            final_entities = spacy_entities + resolved_pathways + resolved_diseases + direct_pathway_ids
            print(f"🔍 FINAL ENTITIES FOR TYPING: {final_entities}")

            # Step 8: Get entity types
            entities_to_types = self._get_entity_types(final_entities)
            print(f"🔍 ENTITIES WITH TYPES: {entities_to_types}")

            # Step 9: Resolve gene aliases and pathway names to IDs
            normalized_entities = {}

            for entity, entity_type in entities_to_types.items():
                if entity_type == "Gene":
                    # Resolve gene alias to official symbol
                    resolved_gene = self._resolve_gene_alias(entity)
                    normalized_entities[resolved_gene] = entity_type
                    if entity != resolved_gene:
                        print(
                            f"🔍 GENE ALIAS RESOLUTION: '{entity}' → '{resolved_gene}'"
                        )
                elif entity_type == "Pathway":
                    if entity.startswith("R-HSA-"):
                        # Already an ID
                        normalized_entities[entity] = entity_type
                    else:
                        # Resolve pathway name to ID
                        if entity in pathway_mapping:
                            resolved_id = pathway_mapping[entity]
                            normalized_entities[resolved_id] = entity_type
                            print(
                                f"🔍 PATHWAY RESOLUTION: '{entity}' → '{resolved_id}'"
                            )
                        else:
                            normalized_entities[entity] = entity_type
                            print(f"🔍 PATHWAY NOT FOUND IN MAPPING: '{entity}'")
                else:
                    normalized_entities[entity] = entity_type

            entities_to_types = normalized_entities
            print(f"🔍 FINAL NORMALIZED ENTITIES: {entities_to_types}")

            # Step 10: Create validation prompt
            # Only validate when user uses specific pattern: pathway "id or name"
            has_quoted_pathway = bool(
                quoted_pathways
            )  # User used quotes like pathway "MAPK signaling"

            has_pathway_entity = any(
                entity_type == "Pathway" for entity_type in entities_to_types.values()
            )

            if has_quoted_pathway and not has_pathway_entity:
                # User mentioned pathway but none was identified
                validation_prompt = "Please provide a complete pathway name or the corresponding pathway id as per reactome."
            elif not entities_to_types:
                # No entities found at all
                validation_prompt = "Entity extraction failed. Please provide the entities you want to analyze: 'NO, entities are: [list]'"
            else:
                # Normal case - entities found
                entity_list = ", ".join(
                    [
                        (
                            f"{entity} ({etype})"
                            if etype not in ["Pathway", "Disease"]
                            else (
                                self._format_pathway_display(entity)
                                if etype == "Pathway"
                                else self._format_disease_display(entity)
                            )
                        )
                        for entity, etype in entities_to_types.items()
                    ]
                )
                validation_prompt = f"I extracted: {entity_list}\n\nCorrect? Reply 'YES' or 'NO, entities are: [list]'"

        except Exception as e:
            # Fallback if extraction fails
            print(f"Entity extraction failed: {e}")

            # Check if user mentioned pathway-related keywords
            # But exclude queries that are asking ABOUT pathways
            content_lower = message["content"].lower()

            # Queries asking about pathways (should not trigger validation)
            asking_about_pathways = any(
                phrase in content_lower
                for phrase in [
                    "which pathways",
                    "what pathways",
                    "pathways are",
                    "pathways associated",
                    "find pathways",
                    "show pathways",
                    "list pathways",
                ]
            )

            if asking_about_pathways:
                # User is asking about pathways, not specifying one - proceed normally
                validation_prompt = "Entity extraction failed. Please provide the entities you want to analyze: 'NO, entities are: [list]'"
            else:
                # Check for pathway keywords that suggest user is mentioning a specific pathway
                pathway_keywords = [
                    "signaling",
                    "signalling",
                    "process",
                    "regulation",
                    "activation",
                    "inhibition",
                ]

                has_pathway_keywords = any(
                    keyword in content_lower for keyword in pathway_keywords
                )

                if has_pathway_keywords:
                    validation_prompt = "Please provide a complete pathway name or the corresponding pathway id as per reactome."
                else:
                    validation_prompt = "Entity extraction failed. Please provide the entities you want to analyze: 'NO, entities are: [list]'"

            entities_to_types = {}

        # Create and save messages directly
        user_message = Message(**message)
        assistant_message = Message(
            content=validation_prompt,
            origin=message["origin"],
            memory_id=message["memory_id"],
            role="assistant",
            metadata={
                "validation_request": True,
                "extracted_entities": entities_to_types,
                "validation_attempt": validation_attempt,
                "original_entities": entities_to_types,  # Store original for fallback
            },
        )

        self.mem_client.add_messages(user_message)
        self.mem_client.add_messages(assistant_message)

        return {"final_message": assistant_message}

    def _format_pathway_display(self, pathway_id):
        """Format pathway display to show both ID and name."""
        try:
            import json

            mapping_path = self.entities_path / "../pathway_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    pathway_mapping = json.load(f)
                # Create reverse mapping to get name from ID
                id_to_name = {v: k for k, v in pathway_mapping.items() if k != v}
                pathway_name = id_to_name.get(pathway_id, "")
                if pathway_name:
                    return f"{pathway_name} - {pathway_id} (Pathway)"
        except Exception:
            pass
        return f"{pathway_id} (Pathway)"

    def _format_disease_display(self, disease_id):
        """Format disease display to show both ID and name."""
        try:
            import json

            mapping_path = self.entities_path / "../disease_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    disease_mapping = json.load(f)
                # Create reverse mapping to get name from ID
                id_to_name = {v: k for k, v in disease_mapping.items() if k != v and not k.startswith(("OMIM:", "ORPHA:"))}
                disease_name = id_to_name.get(disease_id, "")
                if disease_name:
                    return f"{disease_name} - {disease_id} (Disease)"
        except Exception:
            pass
        return f"{disease_id} (Disease)"

    def _resolve_gene_alias(self, gene_symbol):
        """Resolve gene alias to official symbol with multi-level resolution."""
        # Load gene alias mapping
        gene_alias_mapping = {}
        try:
            import json

            mapping_path = self.entities_path / "../gene_alias_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    gene_alias_mapping = json.load(f)
        except Exception:
            return gene_symbol  # Return original if mapping fails

        if gene_symbol not in gene_alias_mapping:
            return gene_symbol  # Return original if not in mapping

        # Multi-level alias resolution
        current = gene_symbol
        visited = set()  # Prevent infinite loops
        resolution_chain = [current]

        while current in gene_alias_mapping and current not in visited:
            visited.add(current)
            next_symbol = gene_alias_mapping[current]

            if next_symbol == current:
                # Found the official symbol
                break

            current = next_symbol
            resolution_chain.append(current)

            # Safety check for infinite loops
            if len(resolution_chain) > 10:
                print(f"Long alias chain detected: {' → '.join(resolution_chain)}")
                break

        official_symbol = current

        if gene_symbol != official_symbol:
            print(f"🔍 RESOLVED GENE ALIAS: {gene_symbol} → {official_symbol}")

        return official_symbol

    def _get_entity_types(self, entities):
        if self.entities_file_type == "csv":
            return self._from_csvs(entities)
        return self._from_txts(entities)

    def _from_csvs(self, entities):
        entities_to_types = {}
        for csv_path in self.entities_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                for entity in entities:
                    if entity not in entities_to_types:
                        info = df[df["id:String"].str.fullmatch(entity, case=False)]
                        if len(info) > 0:
                            entities_to_types[entity] = info[
                                "preferred_id:String"
                            ].values[0]
            except Exception:
                continue
        return entities_to_types

    def _from_txts(self, entities):
        entities_to_types = {}
        print(f"🔍 TYPING ENTITIES: {entities}")

        # Load pathway mapping for enhanced pathway resolution
        pathway_mapping = {}
        disease_mapping = {}
        try:
            import json

            mapping_path = self.entities_path / "../pathway_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    pathway_mapping = json.load(f)
                    
            disease_mapping_path = self.entities_path / "../disease_mapping.json"
            if disease_mapping_path.exists():
                with open(disease_mapping_path) as f:
                    disease_mapping = json.load(f)
        except Exception:
            pass

        for txt_path in self.entities_path.glob("*.txt"):
            try:
                entity_type = txt_path.stem
                print(f"🔍 PROCESSING {entity_type} file: {txt_path}")
                with open(txt_path) as f:
                    file_entities = [line.strip() for line in f.read().splitlines()]
                print(f"🔍 LOADED {len(file_entities)} {entity_type} entities")

                for entity in entities:
                    if entity not in entities_to_types:
                        # Exact match
                        if entity.lower() in [e.lower() for e in file_entities]:
                            entities_to_types[entity] = entity_type
                            print(f"🔍 MATCHED {entity_type}: '{entity}'")
                        
                        # Check if entity appears in parentheses (for Disease IDs like OMIM:167000)
                        elif entity_type == "Disease":
                            for file_entity in file_entities:
                                if f"({entity})" in file_entity:
                                    entities_to_types[entity] = entity_type
                                    print(f"🔍 MATCHED {entity_type} ID: '{entity}' in '{file_entity}'")
                                    break
                                # Also check if entity matches the disease name part (before parentheses)
                                elif "(" in file_entity:
                                    disease_name = file_entity.split("(")[0].strip()
                                    if entity.lower() == disease_name.lower():
                                        entities_to_types[entity] = entity_type
                                        print(f"🔍 MATCHED {entity_type} NAME: '{entity}' → '{file_entity}'")
                                        break
                            
                            # If no exact match, try fuzzy matching for disease names
                            if entity not in entities_to_types:
                                resolved_disease = self._resolve_disease_alias(entity, disease_mapping)
                                if resolved_disease:
                                    entities_to_types[resolved_disease] = entity_type
                                    print(f"🔍 DISEASE RESOLVED: '{entity}' → '{resolved_disease}'")

                        # Enhanced pathway matching with alias resolution
                        elif entity_type == "Pathway":
                            resolved_pathway = self._resolve_pathway_alias(
                                entity, pathway_mapping
                            )
                            if resolved_pathway:
                                # Store the resolved ID as the key, not the original name
                                entities_to_types[resolved_pathway] = entity_type
                                print(
                                    f"🔍 PATHWAY RESOLVED: '{entity}' → '{resolved_pathway}'"
                                )
            except Exception as e:
                print(f"🔍 ERROR PROCESSING {txt_path}: {e}")
                continue

        print(f"🔍 FINAL ENTITY TYPES: {entities_to_types}")
        return entities_to_types

    def _resolve_pathway_alias(self, entity, pathway_mapping):
        """Resolve pathway aliases similar to gene alias resolution."""
        # Direct exact match - return the ID, not the name
        if entity in pathway_mapping:
            return pathway_mapping[entity]

        # Case-insensitive match
        for pathway_name, pathway_id in pathway_mapping.items():
            if entity.lower() == pathway_name.lower():
                return pathway_id

        # Partial matching for common variations
        entity_lower = entity.lower()
        for pathway_name, pathway_id in pathway_mapping.items():
            pathway_lower = pathway_name.lower()
            # Handle common variations: "signaling" vs "signalling"
            if (
                entity_lower.replace("signaling", "signalling") == pathway_lower
                or entity_lower.replace("signalling", "signaling") == pathway_lower
            ):
                return pathway_id

        # Extract R-HSA- pattern if present
        if "R-HSA-" in entity:
            import re

            pathway_match = re.search(r"R-HSA-\d+", entity)
            if pathway_match:
                pathway_id = pathway_match.group()
                if pathway_id in pathway_mapping:
                    return pathway_id

        return None

    def _resolve_disease_alias(self, entity, disease_mapping):
        """Resolve disease aliases similar to pathway alias resolution."""
        # Direct exact match - return the ID, not the name
        if entity in disease_mapping:
            return disease_mapping[entity]

        # Case-insensitive match
        for disease_name, disease_id in disease_mapping.items():
            if entity.lower() == disease_name.lower():
                return disease_id

        # Fuzzy matching for partial matches
        from difflib import SequenceMatcher
        best_match = None
        best_ratio = 0.0
        
        for disease_name, disease_id in disease_mapping.items():
            if not disease_name.startswith(("OMIM:", "ORPHA:")):  # Skip IDs, only match names
                ratio = SequenceMatcher(None, entity.lower(), disease_name.lower()).ratio()
                if ratio > 0.8 and ratio > best_ratio:  # Higher threshold for entity typing
                    best_match = disease_id
                    best_ratio = ratio

        return best_match


@component
class ValidationHandlerComponent:
    def __init__(self, mem_client, config):
        self.mem_client = mem_client

    @component.output_types(
        action=str,
        entities=Dict[str, str],
        message=Dict[str, Any],
        validation_attempt=int,
    )
    def run(
        self, message: Dict[str, Any], validation_attempt: int = 1
    ) -> Dict[str, Any]:
        user_response = message["content"].strip()

        # Check if this is a pathway choice response
        history, _ = self.mem_client.get_history(message["memory_id"])
        is_choice_request = False
        pathway_choices = []
        original_pathway = ""

        print(f"🔍 VALIDATION HANDLER - USER RESPONSE: '{user_response}'")
        print(
            f"🔍 VALIDATION HANDLER - HISTORY LENGTH: {len(history) if history else 0}"
        )

        if history:
            last_msg = history[-1]
            print(f"🔍 LAST MESSAGE ROLE: {last_msg.role}")
            print(f"🔍 LAST MESSAGE METADATA: {last_msg.metadata}")
            if last_msg.role == "assistant" and last_msg.metadata.get("choice_request"):
                is_choice_request = True
                pathway_choices = last_msg.metadata.get("pathway_choices", [])
                original_pathway = last_msg.metadata.get("original_pathway", "")
                print(f"🔍 DETECTED CHOICE REQUEST - CHOICES: {pathway_choices}")

        if is_choice_request:
            # Handle pathway choice response
            if user_response.isdigit():
                choice_num = int(user_response)
                if 1 <= choice_num <= len(pathway_choices):
                    chosen_pathway = pathway_choices[choice_num - 1]
                    print(f"🔍 USER CHOSE PATHWAY: {choice_num} → '{chosen_pathway}'")

                    # Continue with chosen pathway
                    return {
                        "action": "continue_with_pathway",
                        "entities": {chosen_pathway: "Pathway"},
                        "message": message,
                        "validation_attempt": 1,
                    }

            # If not a valid choice, ask again
            return {
                "action": "clarify",
                "entities": {},
                "message": message,
                "validation_attempt": validation_attempt,
            }

        # Regular validation logic
        user_response_lower = user_response.lower()

        # Get stored entities from last assistant message
        stored_entities = {}
        original_entities = {}

        if history:
            for msg in reversed(history):
                if msg.role == "assistant" and msg.metadata.get("extracted_entities"):
                    stored_entities = msg.metadata["extracted_entities"]
                    original_entities = msg.metadata.get(
                        "original_entities", stored_entities
                    )
                    break

        # Simple rule-based parsing instead of LLM
        if user_response_lower in [
            "yes",
            "y",
            "correct",
            "ok",
            "right",
            "good",
            "approve",
        ]:
            return {
                "action": "continue",
                "entities": stored_entities,
                "message": message,
                "validation_attempt": validation_attempt,
            }

        elif user_response.startswith("no, entities are:"):
            # Extract entities from user correction
            entity_text = user_response.replace("no, entities are:", "").strip()

            if validation_attempt < 3:
                # Re-extract entities from corrected text
                return {
                    "action": "re_extract",
                    "entities": {},
                    "message": {
                        "content": entity_text,
                        **{k: v for k, v in message.items() if k != "content"},
                    },
                    "validation_attempt": validation_attempt + 1,
                }
            else:
                # Third attempt - use original entities with explanation
                original_list = ", ".join(
                    [
                        f"{entity} ({etype})"
                        for entity, etype in original_entities.items()
                    ]
                )
                fallback_message = f"""I understand you're not satisfied with the entities. I'll proceed with the original entities I detected from your query: {original_list}

This is based on my analysis of your original question. Processing now..."""

                user_message = Message(**message)
                assistant_message = Message(
                    content=fallback_message,
                    origin=message["origin"],
                    memory_id=message["memory_id"],
                    role="assistant",
                    metadata={"fallback_explanation": True},
                )

                self.mem_client.add_messages(user_message)
                self.mem_client.add_messages(assistant_message)

                return {
                    "action": "continue",
                    "entities": original_entities,
                    "message": message,
                    "validation_attempt": 1,
                }

        elif user_response in ["no", "n", "wrong", "incorrect", "reject"]:
            if validation_attempt < 3:
                return {
                    "action": "clarify",
                    "entities": {},
                    "message": message,
                    "validation_attempt": validation_attempt,
                }
            else:
                # Use original entities after 3 attempts
                original_list = ", ".join(
                    [
                        f"{entity} ({etype})"
                        for entity, etype in original_entities.items()
                    ]
                )
                fallback_message = f"Using original entities: {original_list}"

                user_message = Message(**message)
                assistant_message = Message(
                    content=fallback_message,
                    origin=message["origin"],
                    memory_id=message["memory_id"],
                    role="assistant",
                )

                self.mem_client.add_messages(user_message)
                self.mem_client.add_messages(assistant_message)

                return {
                    "action": "continue",
                    "entities": original_entities,
                    "message": message,
                    "validation_attempt": 1,
                }

        else:  # Unclear response
            return {
                "action": "clarify",
                "entities": {},
                "message": message,
                "validation_attempt": validation_attempt,
            }


@component
class GTExProcessorComponent:
    def __init__(
        self,
        config,
        schema_path: str,
        model_path: str,
        entities_path: str,
        entities_file_type: str,
        mem_client,
    ):
        self.config = config
        self.schema_path = schema_path
        self.model_path = model_path
        self.entities_path = entities_path
        self.entities_file_type = entities_file_type
        self.mem_client = mem_client

    @component.output_types(final_message=Message)
    def run(
        self,
        entities: Dict[str, str],
        message: Dict[str, Any],
        original_pathway_name: str = "",
    ) -> Dict[str, Any]:
        # Use the CURRENT message content, not conversation history
        current_query = message["content"]
        print(f"🔍 CURRENT MESSAGE CONTENT: {current_query}")
        print(f"🔍 VALIDATED ENTITIES: {entities}")

        # For validation responses like "YES", get the original query from history
        if current_query.lower().strip() in ["yes", "y", "no", "n"]:
            history, _ = self.mem_client.get_history(message["memory_id"])
            for msg in reversed(history):
                if (
                    msg.role == "user"
                    and len(msg.content) > 10
                    and msg.content.lower().strip() not in ["yes", "y", "no", "n"]
                ):
                    current_query = msg.content
                    print(f"🔍 FOUND ORIGINAL QUERY FROM HISTORY: {current_query}")
                    break

        # Replace entity names with resolved IDs in the current query
        enhanced_query = current_query
        pathway_mapping = self._load_pathway_mapping()
        disease_mapping = self._load_disease_mapping()
        print(
            f"🔍 LOADED PATHWAY MAPPING FOR REPLACEMENT: {len(pathway_mapping)} entries"
        )
        print(
            f"🔍 LOADED DISEASE MAPPING FOR REPLACEMENT: {len(disease_mapping)} entries"
        )

        for entity_id, entity_type in entities.items():
            if entity_type == "Pathway" and entity_id.startswith("R-HSA-"):
                print(f"🔍 PROCESSING PATHWAY ID: {entity_id}")

                # Use original pathway name if provided, otherwise look it up
                if original_pathway_name:
                    pathway_to_replace = original_pathway_name
                    print(f"🔍 USING ORIGINAL PATHWAY NAME: '{pathway_to_replace}'")

                    if pathway_to_replace.lower() in enhanced_query.lower():
                        # Case-insensitive replacement
                        import re

                        pattern = re.compile(
                            re.escape(pathway_to_replace), re.IGNORECASE
                        )
                        enhanced_query = pattern.sub(entity_id, enhanced_query)
                        print(
                            f"🔍 REPLACED IN QUERY: '{pathway_to_replace}' → '{entity_id}'"
                        )
                    else:
                        print(
                            f"🔍 PATHWAY NAME NOT FOUND IN QUERY: '{pathway_to_replace}' not in '{enhanced_query}'"
                        )
                else:
                    # Fallback to lookup (existing logic)
                    for pathway_name, pathway_id in pathway_mapping.items():
                        if pathway_id == entity_id and not pathway_name.startswith(
                            "R-HSA-"
                        ):
                            print(
                                f"🔍 FOUND PATHWAY NAME FOR ID: '{pathway_name}' → '{entity_id}'"
                            )
                            if pathway_name.lower() in enhanced_query.lower():
                                # Case-insensitive replacement
                                import re

                                pattern = re.compile(
                                    re.escape(pathway_name), re.IGNORECASE
                                )
                                enhanced_query = pattern.sub(entity_id, enhanced_query)
                                print(
                                    f"🔍 REPLACED IN QUERY: '{pathway_name}' → '{entity_id}'"
                                )
                            else:
                                print(
                                    f"🔍 PATHWAY NAME NOT FOUND IN QUERY: '{pathway_name}' not in '{enhanced_query}'"
                                )
                            break
            
            elif entity_type == "Disease":
                print(f"🔍 PROCESSING DISEASE ID: {entity_id}")
                
                # Find the disease name that maps to this ID
                disease_to_replace = None
                for disease_name, disease_id in disease_mapping.items():
                    if disease_id == entity_id and not disease_name.startswith(("OMIM:", "ORPHA:")) and disease_name != entity_id:
                        disease_to_replace = disease_name
                        break
                
                if disease_to_replace and disease_to_replace.lower() in enhanced_query.lower():
                    import re
                    pattern = re.compile(re.escape(disease_to_replace), re.IGNORECASE)
                    enhanced_query = pattern.sub(entity_id, enhanced_query)
                    print(f"🔍 REPLACED IN QUERY: '{disease_to_replace}' → '{entity_id}'")
                else:
                    print(f"🔍 DISEASE NAME NOT FOUND IN QUERY for {entity_id}")

        print(f"🔍 ENHANCED QUERY: {enhanced_query}")

        # Load schema
        schema = self._load_schema()

        gtex_pipeline = QueryEnhancerPipeline(schema=schema, config=self.config)

        # Process with GTEx using the enhanced query with resolved IDs
        result = gtex_pipeline.run(
            prompt=enhanced_query,
            model_path=Path(self.model_path),
            entities_path=Path(self.entities_path),
            entity_file_type=self.entities_file_type,
        )

        # Create and save messages directly - preserve original user message content
        # Don't modify the original message dict, create a clean user message
        original_user_message = Message(
            content=message[
                "content"
            ],  # Use original content, not modified current_query
            origin=message["origin"],
            memory_id=message["memory_id"],
            role=message["role"],
            metadata=message.get("metadata", {}),
            timestamp=message.get("timestamp"),
            message_id=message.get("message_id"),
        )

        assistant_message = Message(
            content=result.nl_db_response,
            origin=message["origin"],
            memory_id=message["memory_id"],
            role="assistant",
            metadata={
                "validated_entities": entities,
                "gremlin_query": result.generated_query,
                "original_query": current_query,  # Store the enhanced query in metadata
                "user_original_input": message[
                    "content"
                ],  # Preserve what user actually typed
            },
        )

        self.mem_client.add_messages(original_user_message)
        self.mem_client.add_messages(assistant_message)

        return {"final_message": assistant_message}
        print(f"🔍 VALIDATED ENTITIES: {entities}")

        # Load schema
        schema = self._load_schema()

        # Check token count before processing
        schema_text = str(schema)
        print(f"🔍 SCHEMA LENGTH: {len(schema_text)} characters")
        print(f"🔍 ORIGINAL QUERY LENGTH: {len(original_query)} characters")
        print(
            f"🔍 ESTIMATED TOKENS: {(len(schema_text) + len(original_query)) // 4}"
        )  # Rough estimate: 4 chars = 1 token

        gtex_pipeline = QueryEnhancerPipeline(schema=schema, config=self.config)

        # Process with GTEx using the enhanced query with resolved IDs
        result = gtex_pipeline.run(
            prompt=enhanced_query,
            model_path=Path(self.model_path),
            entities_path=Path(self.entities_path),
            entity_file_type=self.entities_file_type,
        )

        # Create and save messages directly - preserve original user message content
        original_user_message = Message(
            content=message["content"],  # Use original content, not modified
            origin=message["origin"],
            memory_id=message["memory_id"],
            role=message["role"],
            metadata=message.get("metadata", {}),
            timestamp=message.get("timestamp"),
            message_id=message.get("message_id"),
        )

        assistant_message = Message(
            content=result.nl_db_response,
            origin=message["origin"],
            memory_id=message["memory_id"],
            role="assistant",
            metadata={
                "validated_entities": entities,
                "gremlin_query": result.generated_query,
                "original_query": original_query,  # Store for debugging
                "user_original_input": message[
                    "content"
                ],  # Preserve what user actually typed
            },
        )

        self.mem_client.add_messages(original_user_message)
        self.mem_client.add_messages(assistant_message)

        return {"final_message": assistant_message}

    def _load_schema(self):
        with open(self.schema_path) as f:
            if self.schema_path.endswith(".json"):
                return json.load(f)
            return yaml.safe_load(f)

    def _load_pathway_mapping(self):
        """Load pathway mapping for name-to-ID resolution."""
        try:
            import json

            mapping_path = Path(self.entities_path) / "../pathway_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _load_disease_mapping(self):
        """Load disease mapping for name-to-ID resolution."""
        try:
            import json

            mapping_path = Path(self.entities_path) / "../disease_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}


@component
class ClarificationComponent:
    def __init__(self, mem_client):
        self.mem_client = mem_client

    @component.output_types(final_message=Message)
    def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        # Create and save messages directly
        user_message = Message(**message)
        assistant_message = Message(
            content="Please respond with 'YES' if entities are correct, or 'NO, entities are: [your entities]' to provide corrections.",
            origin=message["origin"],
            memory_id=message["memory_id"],
            role="assistant",
            metadata={"validation_request": True},
        )

        self.mem_client.add_messages(user_message)
        self.mem_client.add_messages(assistant_message)

        return {"final_message": assistant_message}


@component
class RegularChatComponent:
    def __init__(self, config, mem_client):
        self.config = config
        self.mem_client = mem_client

        # Initialize LLM for regular chat
        generator_kwargs = {
            "model": config.models.get("aws", {}).get(
                "name", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
            ),
            "generation_kwargs": {"temperature": 0.7},
        }
        generator_kwargs.update(get_boto3session_haystack(config.models.get("aws", {})))
        self.llm = AmazonBedrockChatGenerator(**generator_kwargs)

        # Simple chat prompt
        self.template = [
            ChatMessage.from_system(
                "You are a helpful AI assistant. Answer questions clearly and concisely."
            ),
            ChatMessage.from_user("{{content}}"),
        ]
        self.prompt_builder = ChatPromptBuilder(
            template=self.template, required_variables=["content"]
        )

    @component.output_types(final_message=Message)
    def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        # Build prompt and get LLM response
        prompt_result = self.prompt_builder.run(content=message["content"])
        llm_result = self.llm.run(messages=prompt_result["prompt"])

        response_content = llm_result["replies"][0].text.strip()

        # Create and save messages
        user_message = Message(**message)
        assistant_message = Message(
            content=response_content,
            origin=message["origin"],
            memory_id=message["memory_id"],
            role="assistant",
            metadata={"pipeline": "regular_chat"},
        )

        self.mem_client.add_messages(user_message)
        self.mem_client.add_messages(assistant_message)

        return {"final_message": assistant_message}


class EnhancedGtexChat:
    def __init__(self, config, mem_client, ks_client):
        self.config = config
        self.mem_client = mem_client
        self.ks_client = ks_client

    def enhanced_gtex_chat(
        self,
        prompt: Message,
        schema_path: str,
        entities_path: str,
        entities_file_type: Literal["csv", "txt"],
        model_path: str,
    ) -> Message:
        """Pure Haystack pipeline for GTEx chat with intelligent entity validation."""

        # Build pipeline
        pipe = Pipeline()

        # Add components
        pipe.add_component(
            "preprocessor",
            PreProcessorComponent(
                self.mem_client,
                self.config,
                schema_path,
                model_path,
                entities_path,
                entities_file_type,
            ),
        )
        pipe.add_component("supervisor", SupervisorComponent(self.config))
        pipe.add_component(
            "entity_extractor_reextract",
            EntityExtractorComponent(
                model_path,
                entities_path,
                entities_file_type,
                self.mem_client,
                self.config,
                schema_path,
            ),
        )
        pipe.add_component(
            "entity_extractor_new",
            EntityExtractorComponent(
                model_path,
                entities_path,
                entities_file_type,
                self.mem_client,
                self.config,
                schema_path,
            ),
        )
        pipe.add_component(
            "regular_chat", RegularChatComponent(self.config, self.mem_client)
        )

        # Add routers
        pipe.add_component(
            "preprocessor_router",
            ConditionalRouter(
                routes=[
                    {
                        "condition": "{{action == 'continue'}}",
                        "output": "{{message}}",
                        "output_name": "continue",
                        "output_type": Dict[str, Any],
                    },
                    {
                        "condition": "{{action == 're_extract'}}",
                        "output": "{{message}}",
                        "output_name": "re_extract",
                        "output_type": Dict[str, Any],
                    },
                    {
                        "condition": "{{action == 'handled'}}",
                        "output": "{{final_message}}",
                        "output_name": "handled",
                        "output_type": Message,
                    },
                ]
            ),
        )

        pipe.add_component(
            "pipeline_router",
            ConditionalRouter(
                routes=[
                    {
                        "condition": "{{decision == 'gtex'}}",
                        "output": "{{message}}",
                        "output_name": "gtex",
                        "output_type": Dict[str, Any],
                    },
                    {
                        "condition": "{{decision != 'gtex'}}",
                        "output": "{{message}}",
                        "output_name": "regular",
                        "output_type": Dict[str, Any],
                    },
                ]
            ),
        )

        # Pipeline connections
        pipe.connect("preprocessor.action", "preprocessor_router.action")
        pipe.connect("preprocessor.message", "preprocessor_router.message")
        pipe.connect("preprocessor.final_message", "preprocessor_router.final_message")

        # Continue to supervisor for new queries
        pipe.connect("preprocessor_router.continue", "supervisor.message")

        # Re-extract path
        pipe.connect(
            "preprocessor_router.re_extract", "entity_extractor_reextract.message"
        )

        # Normal supervisor flow
        pipe.connect("supervisor.decision", "pipeline_router.decision")
        pipe.connect("supervisor.message", "pipeline_router.message")

        # GTEx path - goes through entity extraction
        pipe.connect("pipeline_router.gtex", "entity_extractor_new.message")

        # Regular chat path
        pipe.connect("pipeline_router.regular", "regular_chat.message")

        # Check if preprocessor can handle this directly (pathway choice, validation)
        preprocessor = PreProcessorComponent(
            self.mem_client,
            self.config,
            schema_path,
            model_path,
            entities_path,
            entities_file_type,
        )
        preprocessor_result = preprocessor.run(prompt.to_dict())

        # If preprocessor handled it directly, return the result
        if preprocessor_result.get("action") == "handled":
            return preprocessor_result["final_message"]

        # Run pipeline for cases that need further processing
        result = pipe.run({"preprocessor": {"message": prompt.to_dict()}})

        # Debug: Print result keys
        print(f"🔍 Pipeline result keys: {list(result.keys())}")
        for key, value in result.items():
            print(f"🔍 Result[{key}]: {value}")

        # Extract result from whichever component executed

        # Check for preprocessor handled cases first
        if (
            "preprocessor_router" in result
            and "handled" in result["preprocessor_router"]
        ):
            return result["preprocessor_router"]["handled"]

        # Check if preprocessor_router is in result but with different structure
        if "preprocessor_router" in result:
            print(f"🔍 preprocessor_router result: {result['preprocessor_router']}")

        for component_name in [
            "entity_extractor_reextract",
            "entity_extractor_new",
            "regular_chat",
        ]:
            if component_name in result:
                return result[component_name]["final_message"]

        # Handle re_extract case - return a waiting message
        if "re_extract" in result.get("preprocessor_router", {}):
            return Message(
                content="Processing your corrected entities...",
                origin=prompt.origin,
                memory_id=prompt.memory_id,
                role="assistant",
            )

        # Check if supervisor routed to regular pipeline
        if "pipeline_router" in result and "regular" in result["pipeline_router"]:
            return Message(
                content="This query should be handled by the regular pipeline, not GTEx.",
                origin=prompt.origin,
                memory_id=prompt.memory_id,
                role="assistant",
            )

        # Fallback
        return Message(
            content="Pipeline execution failed",
            origin=prompt.origin,
            memory_id=prompt.memory_id,
            role="assistant",
        )
