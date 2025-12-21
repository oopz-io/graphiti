"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from datetime import datetime
from time import time

from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import ExtractedEdges, MissingFacts
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_fulltext_search_batch, edge_similarity_search_batch
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

logger = logging.getLogger(__name__)

# Module-level storage for profiling data
_profiling_data: dict[str, dict[str, float]] = {}


def get_profiling_data() -> dict[str, dict[str, float]]:
    """Get the profiling data collected from edge operations."""
    return _profiling_data.copy()


def clear_profiling_data():
    """Clear the profiling data storage."""
    _profiling_data.clear()


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=node.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built episodic edges: {episodic_edges}')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityEdge]:
    start_total = time()
    step_start = time()

    # Initialize profiling data for this function
    _profiling_data['extract_edges'] = {}

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    edge_type_signature_map: dict[str, tuple[str, str]] = {
        edge_type: signature
        for signature, edge_types in edge_type_map.items()
        for edge_type in edge_types
    }

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signature': edge_type_signature_map.get(type_name, ('Entity', 'Entity')),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [
            {'id': idx, 'name': node.name, 'entity_types': node.labels}
            for idx, node in enumerate(nodes)
        ],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
        'custom_prompt': '',
        'ensure_ascii': clients.ensure_ascii,
    }

    prep_time = (time() - step_start) * 1000
    _profiling_data['extract_edges']['context_prep'] = prep_time
    logger.debug(f'[PROFILING] extract_edges - Context preparation: {prep_time:.2f}ms')
    step_start = time()

    facts_missed = True
    reflexion_iterations = 0
    llm_calls = 0
    llm_total_time = 0.0

    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        llm_call_start = time()
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges.edge(context),
            response_model=ExtractedEdges,
            max_tokens=extract_edges_max_tokens,
        )
        llm_call_time = (time() - llm_call_start) * 1000
        llm_total_time += llm_call_time
        llm_calls += 1
        logger.debug(f'[PROFILING] extract_edges - LLM call #{llm_calls}: {llm_call_time:.2f}ms')

        edges_data = ExtractedEdges(**llm_response).edges

        context['extracted_facts'] = [edge_data.fact for edge_data in edges_data]

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            reflexion_response = await llm_client.generate_response(
                prompt_library.extract_edges.reflexion(context),
                response_model=MissingFacts,
                max_tokens=extract_edges_max_tokens,
            )

            missing_facts = reflexion_response.get('missing_facts', [])

            custom_prompt = 'The following facts were missed in a previous extraction: '
            for fact in missing_facts:
                custom_prompt += f'\n{fact},'

            context['custom_prompt'] = custom_prompt

            facts_missed = len(missing_facts) != 0

    _profiling_data['extract_edges']['llm_calls'] = llm_calls
    _profiling_data['extract_edges']['llm_total_time'] = llm_total_time
    logger.debug(
        f'[PROFILING] extract_edges - Total LLM time ({llm_calls} calls): {llm_total_time:.2f}ms'
    )
    step_start = time()

    end = time()
    logger.debug(f'Extracted new edges: {edges_data} in {(end - start_total) * 1000} ms')

    if len(edges_data) == 0:
        total_time = (time() - start_total) * 1000
        _profiling_data['extract_edges']['total'] = total_time
        logger.debug(f'[PROFILING] extract_edges - TOTAL: {total_time:.2f}ms (no edges extracted)')
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        source_node_idx = edge_data.source_entity_id
        target_node_idx = edge_data.target_entity_id
        if not (-1 < source_node_idx < len(nodes) and -1 < target_node_idx < len(nodes)):
            logger.warning(
                f'WARNING: source or target node not filled {edge_data.relation_type}. source_node_uuid: {source_node_idx} and target_node_uuid: {target_node_idx} '
            )
            continue
        source_node_uuid = nodes[source_node_idx].uuid
        target_node_uuid = nodes[edge_data.target_entity_id].uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')
        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id,
            fact=edge_data.fact,
            episodes=[episode.uuid],
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge: {edge.name} from (UUID: {edge.source_node_uuid}) to (UUID: {edge.target_node_uuid})'
        )

    edge_creation_time = (time() - step_start) * 1000
    _profiling_data['extract_edges']['edge_creation'] = edge_creation_time
    logger.debug(
        f'[PROFILING] extract_edges - Edge object creation: {edge_creation_time:.2f}ms ({len(edges)} edges)'
    )

    total_time = (time() - start_total) * 1000
    _profiling_data['extract_edges']['total'] = total_time
    logger.debug(f'[PROFILING] extract_edges - TOTAL: {total_time:.2f}ms')

    logger.debug(f'Extracted edges: {[(e.name, e.uuid) for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    save_contradicted_edges: bool = False,
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    from time import time

    start_total = time()
    step_start = time()

    # Initialize profiling data for this function
    _profiling_data['resolve_extracted_edges'] = {}

    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder

    await create_entity_edge_embeddings(embedder, extracted_edges)
    embed_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['create_embeddings'] = embed_time
    logger.debug(f'[PROFILING] resolve_extracted_edges - Create embeddings: {embed_time:.2f}ms')
    step_start = time()

    # OPTIMIZATION: Use batched query to get all edges between node pairs in a single query
    # This reduces N separate queries to 1 query
    node_pairs = [(edge.source_node_uuid, edge.target_node_uuid) for edge in extracted_edges]
    edges_by_pair = await EntityEdge.get_between_nodes_batch(driver, node_pairs)
    
    # Convert to list format matching the original structure
    valid_edges_list: list[list[EntityEdge]] = [
        edges_by_pair.get((edge.source_node_uuid, edge.target_node_uuid), [])
        for edge in extracted_edges
    ]
    get_edges_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['get_existing_edges'] = get_edges_time
    logger.debug(
        f'[PROFILING] resolve_extracted_edges - Get existing edges (batched): {get_edges_time:.2f}ms'
    )
    step_start = time()

    # OPTIMIZATION: Use batched search for related edges on Spanner
    # This reduces N BM25 queries + N similarity queries to just 2 batched queries
    driver = clients.driver
    if driver.provider == GraphProvider.SPANNER and len(extracted_edges) > 1:
        from graphiti_core.search.search_utils import rrf

        # Build exclusion sets for each extracted edge (edges to exclude from results)
        exclusion_sets: list[set[str]] = [
            {edge.uuid for edge in valid_edges}
            for valid_edges in valid_edges_list
        ]

        # Group edges by group_id for batched search
        group_to_indices: dict[str, list[int]] = {}
        for idx, edge in enumerate(extracted_edges):
            if edge.group_id not in group_to_indices:
                group_to_indices[edge.group_id] = []
            group_to_indices[edge.group_id].append(idx)

        # Prepare results structure
        related_edges_lists: list[list[EntityEdge]] = [[] for _ in extracted_edges]

        for group_id, indices in group_to_indices.items():
            # Get facts and vectors for this group
            facts = [extracted_edges[idx].fact for idx in indices]
            vectors = [
                extracted_edges[idx].fact_embedding
                for idx in indices
                if extracted_edges[idx].fact_embedding is not None
            ]

            # Batched BM25 search
            bm25_results = await edge_fulltext_search_batch(
                driver=driver,
                queries=facts,
                group_ids=[group_id],
                limit_per_query=20,  # 2 * DEFAULT_SEARCH_LIMIT
            )

            # Batched similarity search (if we have embeddings)
            if vectors:
                sim_results = await edge_similarity_search_batch(
                    driver=driver,
                    search_vectors=vectors,
                    group_ids=[group_id],
                    limit_per_vector=20,  # 2 * DEFAULT_SEARCH_LIMIT
                )
            else:
                sim_results = {}

            # Build per-fact BM25 result lookup (dedup by fact+uuid)
            fact_to_bm25_results: dict[str, list[EntityEdge]] = {fact: [] for fact in facts}
            seen_per_fact: dict[str, set[str]] = {fact: set() for fact in facts}
            for edge in bm25_results:
                # BM25 results are pooled, need to check which facts could match
                # For simplicity, include in all (RRF will handle ranking)
                for fact in facts:
                    if edge.uuid not in seen_per_fact[fact]:
                        fact_to_bm25_results[fact].append(edge)
                        seen_per_fact[fact].add(edge.uuid)

            # Combine BM25 and similarity results using RRF for each extracted edge
            for local_idx, global_idx in enumerate(indices):
                fact = facts[local_idx]
                exclusion_set = exclusion_sets[global_idx]

                # Get BM25 results for this fact, excluding valid_edges
                bm25_edges = [
                    e for e in fact_to_bm25_results.get(fact, [])
                    if e.uuid not in exclusion_set
                ]

                # Get similarity results for this index, excluding valid_edges
                sim_edges_raw = sim_results.get(local_idx, [])
                sim_edges = [e for e in sim_edges_raw if e.uuid not in exclusion_set]

                # Apply RRF to combine results
                bm25_uuids = [e.uuid for e in bm25_edges]
                sim_uuids = [e.uuid for e in sim_edges]

                if bm25_uuids or sim_uuids:
                    edge_uuid_map = {e.uuid: e for e in bm25_edges + sim_edges}
                    reranked_uuids, _ = rrf([bm25_uuids, sim_uuids])
                    related_edges_lists[global_idx] = [
                        edge_uuid_map[uuid] for uuid in reranked_uuids[:10]
                    ]  # Apply limit

        logger.debug(
            f'[BATCH_SEARCH] resolve_extracted_edges: batched {len(extracted_edges)} related edge searches'
        )
    else:
        # Original behavior for non-Spanner or single edge
        related_edges_results: list[SearchResults] = await semaphore_gather(
            *[
                search(
                    clients,
                    extracted_edge.fact,
                    group_ids=[extracted_edge.group_id],
                    config=EDGE_HYBRID_SEARCH_RRF,
                    search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
                )
                for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
            ]
        )
        related_edges_lists = [result.edges for result in related_edges_results]

    search_related_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['search_related_edges'] = search_related_time
    logger.debug(
        f'[PROFILING] resolve_extracted_edges - Search related edges: {search_related_time:.2f}ms'
    )
    step_start = time()

    # OPTIMIZATION: Use batched search for invalidation candidates on Spanner
    # This reduces N queries to 1 query since all edges have empty SearchFilters()
    if driver.provider == GraphProvider.SPANNER and len(extracted_edges) > 1:
        from graphiti_core.search.search_utils import rrf as rrf_func

        # Group edges by group_id with their indices
        group_to_indices: dict[str, list[int]] = {}
        for idx, edge in enumerate(extracted_edges):
            if edge.group_id not in group_to_indices:
                group_to_indices[edge.group_id] = []
            group_to_indices[edge.group_id].append(idx)

        # Prepare results structure
        edge_invalidation_candidates: list[list[EntityEdge]] = [[] for _ in extracted_edges]

        for group_id, indices in group_to_indices.items():
            # Get facts and vectors for this group
            facts = [extracted_edges[idx].fact for idx in indices]
            vectors = [
                extracted_edges[idx].fact_embedding
                for idx in indices
                if extracted_edges[idx].fact_embedding is not None
            ]

            # Batched BM25 search
            bm25_results = await edge_fulltext_search_batch(
                driver=driver,
                queries=facts,
                group_ids=[group_id],
                limit_per_query=10,  # 2 * limit for RRF
            )

            # Batched similarity search (if we have embeddings)
            if vectors:
                sim_results = await edge_similarity_search_batch(
                    driver=driver,
                    search_vectors=vectors,
                    group_ids=[group_id],
                    limit_per_vector=10,  # 2 * limit for RRF
                )
            else:
                sim_results = {}

            # Build UUID to edge mapping
            all_edges = list(bm25_results) + [e for edges in sim_results.values() for e in edges]
            edge_uuid_map = {e.uuid: e for e in all_edges}

            # Apply RRF for each extracted edge
            for local_idx, global_idx in enumerate(indices):
                # Get BM25 results (pooled)
                bm25_uuids = [e.uuid for e in bm25_results]

                # Get similarity results for this index
                sim_edges = sim_results.get(local_idx, [])
                sim_uuids = [e.uuid for e in sim_edges]

                if bm25_uuids or sim_uuids:
                    reranked_uuids, _ = rrf_func([bm25_uuids, sim_uuids])
                    edge_invalidation_candidates[global_idx] = [
                        edge_uuid_map[uuid] for uuid in reranked_uuids[:5]
                    ]  # Apply limit

        logger.debug(
            f'[BATCH_SEARCH] resolve_extracted_edges: batched {len(extracted_edges)} invalidation candidates'
        )
    else:
        # Original behavior for non-Spanner or single edge
        edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
            *[
                search(
                    clients,
                    extracted_edge.fact,
                    group_ids=[extracted_edge.group_id],
                    config=EDGE_HYBRID_SEARCH_RRF,
                    search_filter=SearchFilters(),
                )
                for extracted_edge in extracted_edges
            ]
        )
        edge_invalidation_candidates = [
            result.edges for result in edge_invalidation_candidate_results
        ]
    
    search_invalidation_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['search_invalidation_candidates'] = (
        search_invalidation_time
    )
    logger.debug(
        f'[PROFILING] resolve_extracted_edges - Search invalidation candidates: {search_invalidation_time:.2f}ms'
    )
    step_start = time()

    logger.debug(
        f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Determine which edge types are relevant for each edge
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    prep_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['prepare_edge_types'] = prep_time
    logger.debug(f'[PROFILING] resolve_extracted_edges - Prepare edge types: {prep_time:.2f}ms')
    step_start = time()

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                    clients.ensure_ascii,
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
                )
            ]
        )
    )

    resolve_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['resolve_individual_edges'] = resolve_time
    logger.debug(
        f'[PROFILING] resolve_extracted_edges - Resolve individual edges (LLM): {resolve_time:.2f}ms'
    )
    step_start = time()

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    contradicted_edge_pairs: list[tuple[EntityEdge, EntityEdge]] = []  # (invalidated, invalidating)

    for result in results:
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

        # Track which edges were invalidated by this resolved edge
        for invalidated_edge in invalidated_edge_chunk:
            contradicted_edge_pairs.append((invalidated_edge, resolved_edge))

    logger.debug(f'Resolved edges: {[(e.name, e.uuid) for e in resolved_edges]}')

    # Save contradicted edges to Spanner if enabled
    if save_contradicted_edges and contradicted_edge_pairs:
        from graphiti_core.driver.spanner_driver import SpannerDriver

        if isinstance(driver, SpannerDriver):
            try:
                # Get group_id from the first resolved edge
                group_id = resolved_edges[0].group_id if resolved_edges else episode.group_id
                await driver.save_contradicted_edges(contradicted_edge_pairs, group_id)
            except Exception as e:
                logger.error(f'Failed to save contradicted edges: {e}', exc_info=True)
        else:
            logger.debug(
                'save_contradicted_edges is enabled but driver is not SpannerDriver, skipping'
            )

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    final_embed_time = (time() - step_start) * 1000
    _profiling_data['resolve_extracted_edges']['final_embeddings'] = final_embed_time
    logger.debug(
        f'[PROFILING] resolve_extracted_edges - Final embeddings: {final_embed_time:.2f}ms'
    )

    total_time = (time() - start_total) * 1000
    _profiling_data['resolve_extracted_edges']['total'] = total_time
    logger.debug(f'[PROFILING] resolve_extracted_edges - TOTAL: {total_time:.2f}ms')

    return resolved_edges, invalidated_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        if (
            edge.invalid_at is not None
            and resolved_edge.valid_at is not None
            and edge.invalid_at <= resolved_edge.valid_at
        ) or (
            edge.valid_at is not None
            and resolved_edge.invalid_at is not None
            and resolved_edge.invalid_at <= edge.valid_at
        ):
            continue
        # New edge invalidates edge
        elif (
            edge.valid_at is not None
            and resolved_edge.valid_at is not None
            and edge.valid_at < resolved_edge.valid_at
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_types: dict[str, type[BaseModel]] | None = None,
    ensure_ascii: bool = True,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    if len(related_edges) == 0 and len(existing_edges) == 0:
        return extracted_edge, [], []

    # Fast path: if the fact text and endpoints already exist verbatim, reuse the matching edge.
    normalized_fact = _normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and _normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge
            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)
            return resolved, [], []

    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'id': edge.uuid, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    invalidation_edge_candidates_context = [
        {'id': i, 'fact': existing_edge.fact} for i, existing_edge in enumerate(existing_edges)
    ]

    edge_types_context = (
        [
            {
                'fact_type_id': i,
                'fact_type_name': type_name,
                'fact_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(edge_types.items())
        ]
        if edge_types is not None
        else []
    )

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
        'edge_types': edge_types_context,
        'ensure_ascii': ensure_ascii,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )
    response_object = EdgeDuplicate(**llm_response)
    duplicate_facts = response_object.duplicate_facts

    duplicate_fact_ids: list[int] = [i for i in duplicate_facts if 0 <= i < len(related_edges)]

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    contradicted_facts: list[int] = response_object.contradicted_facts

    invalidation_candidates: list[EntityEdge] = [
        existing_edges[i] for i in contradicted_facts if 0 <= i < len(existing_edges)
    ]

    fact_type: str = response_object.fact_type
    if fact_type.upper() != 'DEFAULT' and edge_types is not None:
        resolved_edge.name = fact_type

        edge_attributes_context = {
            'episode_content': episode.content,
            'reference_time': episode.valid_at,
            'fact': resolved_edge.fact,
            'ensure_ascii': ensure_ascii,
        }

        edge_model = edge_types.get(fact_type)
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
            )

            resolved_edge.attributes = edge_attributes_response

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.name} is {resolved_edge.name}, in {(end - start) * 1000} ms'
    )

    now = utc_now()

    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, c.valid_at))
        for candidate in invalidation_candidates:
            if (
                candidate.valid_at
                and resolved_edge.valid_at
                and candidate.valid_at.tzinfo
                and resolved_edge.valid_at.tzinfo
                and candidate.valid_at > resolved_edge.valid_at
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )
    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, invalidated_edges, duplicate_edges


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
