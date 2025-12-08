"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law o                # Episodic edges (MENTIONS edges between episodes and entities)
                # Note: EpisodicEdge model doesn't have 'name' field, only base fields
                episodic_edge_columns = ['uuid', 'source_node_uuid', 'target_node_uuid', 'group_id', 'created_at']
                for edge in episodic_edges:
                    edge_dict = edge.model_dump()
                    mutation_data = {
                        'table': 'EpisodicEdge',
                        'columns': episodic_edge_columns,
                        'values': [
                            edge_dict['uuid'],
                            edge_dict['source_node_uuid'],
                            edge_dict['target_node_uuid'],
                            edge_dict['group_id'],
                            edge_dict['created_at'],
                        ]
                    }
                    mutations.append(mutation_data)riting, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Any

from graphiti_core.driver.driver import (
    ENTITY_EDGE_INDEX_NAME,
    ENTITY_INDEX_NAME,
    EPISODE_INDEX_NAME,
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, create_entity_node_embeddings
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


class RawEpisode(BaseModel):
    name: str
    uuid: str | None = Field(default=None)
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id]
            )
            for episode in episodes
        ]
    )
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]] = [
        (episode, previous_episodes_list[i]) for i, episode in enumerate(episodes)
    ]

    return episode_tuples


async def add_nodes_and_edges_bulk(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    use_batch_write: bool | None = None,
):
    """Add nodes and edges to the graph in bulk.

    For Spanner, this supports two modes:

    1. Transaction mode (use_batch_write=False):
       - Uses read-write transactions with retry logic
       - May encounter 409 ABORTED errors under high parallelism
       - Atomic: all mutations succeed or all fail

    2. BatchWrite mode (use_batch_write=True, default for Spanner):
       - Uses blind writes without transactions
       - Zero lock conflicts, linear scalability
       - Each mutation is atomic, but mutations are independent
       - Last-write-wins semantics (safe for upserts)

    Args:
        driver: The graph driver
        episodic_nodes: Episode nodes to insert
        episodic_edges: Episodic edges to insert
        entity_nodes: Entity nodes to insert
        entity_edges: Entity edges to insert
        embedder: Embedder client for generating embeddings
        use_batch_write: For Spanner only. If True, use BatchWrite API for
            conflict-free parallel writes. If False, use transactions.
            If None (default), uses SPANNER_BATCH_WRITE_CONFIG['enabled'].
    """
    # Check if we should use BatchWrite for Spanner
    if driver.provider == GraphProvider.SPANNER:
        from graphiti_core.driver.spanner_driver import SPANNER_BATCH_WRITE_CONFIG

        # Determine whether to use batch write
        should_use_batch_write = use_batch_write
        if should_use_batch_write is None:
            should_use_batch_write = SPANNER_BATCH_WRITE_CONFIG.get('enabled', True)

        if should_use_batch_write:
            # Use BatchWrite mode - conflict-free blind writes
            await _add_nodes_and_edges_bulk_batch_write(
                driver,
                episodic_nodes,
                episodic_edges,
                entity_nodes,
                entity_edges,
                embedder,
            )
            return

    # Default: use transaction mode
    session = driver.session()
    try:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
    finally:
        await session.close()


async def _add_nodes_and_edges_bulk_batch_write(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
):
    """Spanner-specific: Add nodes and edges using BatchWrite API for conflict-free writes.

    This function uses Spanner's BatchWrite API which performs blind writes:
    - No read phase = no shared locks
    - No lock upgrade = no deadlock detection
    - Mutations are applied directly = no 409 ABORTED errors

    This is ideal for parallel batch processing where multiple workers may
    write to overlapping data (e.g., same entities discovered in different episodes).

    Trade-offs:
    - PRO: Zero lock contention, linear scalability with parallelism
    - CON: Last-write-wins semantics, each mutation atomic but not across mutations
    """
    from time import time

    from graphiti_core.driver.spanner_driver import SPANNER_BATCH_WRITE_CONFIG

    logger.info(
        f'[BATCH_WRITE] Starting batch write: {len(episodic_nodes)} episodes, '
        f'{len(entity_nodes)} nodes, {len(entity_edges)} edges, {len(episodic_edges)} episodic edges'
    )

    total_start = time()

    # Step 1: Generate embeddings in parallel (same as transaction mode)
    embed_start = time()

    nodes_needing_embeddings = [node for node in entity_nodes if node.name_embedding is None]
    if nodes_needing_embeddings:
        await semaphore_gather(
            *[node.generate_name_embedding(embedder) for node in nodes_needing_embeddings]
        )

    edges_needing_embeddings = [edge for edge in entity_edges if edge.fact_embedding is None]
    if edges_needing_embeddings:
        await semaphore_gather(
            *[edge.generate_embedding(embedder) for edge in edges_needing_embeddings]
        )

    embed_time = (time() - embed_start) * 1000
    logger.debug(
        f'[BATCH_WRITE] Embedding generation: {embed_time:.2f}ms '
        f'({len(nodes_needing_embeddings)} nodes + {len(edges_needing_embeddings)} edges)'
    )

    # Step 2: Prepare data for mutations
    prep_start = time()

    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)

    nodes = []
    for node in entity_nodes:
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }
        if not bool(driver.aoss_client):
            entity_data['name_embedding'] = node.name_embedding
        entity_data['labels'] = list(set(node.labels + ['Entity']))
        attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
        entity_data['attributes'] = json.dumps(attributes)
        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at if edge.valid_at else edge.created_at,
            'invalid_at': edge.invalid_at,
        }
        if not bool(driver.aoss_client):
            edge_data['fact_embedding'] = edge.fact_embedding
        attributes = convert_datetimes_to_strings(edge.attributes) if edge.attributes else {}
        edge_data['attributes'] = json.dumps(attributes)
        edges.append(edge_data)

    prep_time = (time() - prep_start) * 1000
    logger.debug(f'[BATCH_WRITE] Data preparation: {prep_time:.2f}ms')

    # Step 3: Build mutations list
    mutations_start = time()
    mutations = []

    # Episodic nodes
    if episodes:
        episode_columns = [
            'source',
            'source_description',
            'content',
            'entity_edges',
            'uuid',
            'name',
            'group_id',
            'created_at',
            'valid_at',
        ]
        for episode in episodes:
            mutations.append(
                {
                    'table': 'EpisodicNode',
                    'columns': episode_columns,
                    'values': [
                        episode['source'],
                        episode.get('source_description', ''),
                        episode.get('content', ''),
                        episode.get('entity_edges', []),
                        episode['uuid'],
                        episode['name'],
                        episode['group_id'],
                        episode['created_at'],
                        episode['valid_at'],
                    ],
                }
            )

    # Entity nodes
    if nodes:
        node_columns = [
            'uuid',
            'name',
            'group_id',
            'labels',
            'created_at',
            'name_embedding',
            'summary',
            'attributes',
        ]
        for node in nodes:
            mutations.append(
                {
                    'table': 'EntityNode',
                    'columns': node_columns,
                    'values': [
                        node['uuid'],
                        node['name'],
                        node['group_id'],
                        node.get('labels', []),
                        node['created_at'],
                        node.get('name_embedding', []),
                        node.get('summary', ''),
                        node.get('attributes', '{}'),
                    ],
                }
            )

    # Entity edges
    if edges:
        edge_columns = [
            'uuid',
            'source_node_uuid',
            'target_node_uuid',
            'name',
            'fact',
            'group_id',
            'episodes',
            'created_at',
            'expired_at',
            'valid_at',
            'invalid_at',
            'fact_embedding',
            'attributes',
            'labels',
        ]
        for edge in edges:
            mutations.append(
                {
                    'table': 'EntityEdge',
                    'columns': edge_columns,
                    'values': [
                        edge['uuid'],
                        edge['source_node_uuid'],
                        edge['target_node_uuid'],
                        edge['name'],
                        edge['fact'],
                        edge['group_id'],
                        edge.get('episodes', []),
                        edge['created_at'],
                        edge.get('expired_at'),
                        edge.get('valid_at'),
                        edge.get('invalid_at'),
                        edge.get('fact_embedding', []),
                        edge.get('attributes', '{}'),
                        edge.get('labels', []),
                    ],
                }
            )

    # Episodic edges
    if episodic_edges:
        episodic_edge_columns = [
            'uuid',
            'source_node_uuid',
            'target_node_uuid',
            'group_id',
            'created_at',
        ]
        for edge in episodic_edges:
            edge_dict = edge.model_dump()
            mutations.append(
                {
                    'table': 'EpisodicEdge',
                    'columns': episodic_edge_columns,
                    'values': [
                        edge_dict['uuid'],
                        edge_dict['source_node_uuid'],
                        edge_dict['target_node_uuid'],
                        edge_dict['group_id'],
                        edge_dict['created_at'],
                    ],
                }
            )

    mutations_time = (time() - mutations_start) * 1000
    logger.debug(
        f'[BATCH_WRITE] Mutation building: {mutations_time:.2f}ms ({len(mutations)} mutations)'
    )

    # Step 4: Execute BatchWrite
    session = driver.session()
    try:
        mutations_per_group = SPANNER_BATCH_WRITE_CONFIG.get('mutations_per_group', 1)

        successful, failed = await session.run_mutations_batch_write(  # type: ignore[attr-defined]
            mutations,
            mutations_per_group=mutations_per_group,
        )

        total_time = (time() - total_start) * 1000
        logger.info(
            f'[BATCH_WRITE] Completed in {total_time:.2f}ms: '
            f'{successful} successful, {failed} failed mutations'
        )

        if failed > 0:
            logger.warning(f'[BATCH_WRITE] {failed} mutations failed during batch write')

    finally:
        await session.close()


async def add_nodes_and_edges_bulk_tx(
    tx: GraphDriverSession,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    from time import time

    step_start = time()

    # OPTIMIZATION 1: Batch generate embeddings in parallel before data prep
    # This is much faster than sequential generation
    embed_start = time()

    # Find nodes without embeddings and generate in parallel
    nodes_needing_embeddings = [node for node in entity_nodes if node.name_embedding is None]
    if nodes_needing_embeddings:
        await semaphore_gather(
            *[node.generate_name_embedding(embedder) for node in nodes_needing_embeddings]
        )

    # Find edges without embeddings and generate in parallel
    edges_needing_embeddings = [edge for edge in entity_edges if edge.fact_embedding is None]
    if edges_needing_embeddings:
        await semaphore_gather(
            *[edge.generate_embedding(embedder) for edge in edges_needing_embeddings]
        )

    embed_time = (time() - embed_start) * 1000
    logger.debug(
        f'[PROFILING] Parallel embedding generation: {embed_time:.2f}ms ({len(nodes_needing_embeddings)} nodes + {len(edges_needing_embeddings)} edges)'
    )

    # OPTIMIZATION 2: Prepare data outside transaction for faster commit
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)

    nodes = []

    for node in entity_nodes:
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }

        if not bool(driver.aoss_client):
            entity_data['name_embedding'] = node.name_embedding

        entity_data['labels'] = list(set(node.labels + ['Entity']))
        if driver.provider in (GraphProvider.KUZU, GraphProvider.SPANNER):
            attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
            entity_data['attributes'] = json.dumps(attributes)
        else:
            entity_data.update(node.attributes or {})

        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        # Embedding already generated in parallel above
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
        }

        if not bool(driver.aoss_client):
            edge_data['fact_embedding'] = edge.fact_embedding

        if driver.provider in (GraphProvider.KUZU, GraphProvider.SPANNER):
            attributes = convert_datetimes_to_strings(edge.attributes) if edge.attributes else {}
            edge_data['attributes'] = json.dumps(attributes)

            # Spanner-specific handling for optional timestamp fields
            # Keep expired_at and invalid_at as NULL (None) by default
            # Only set valid_at to creation time if it's None
            if driver.provider == GraphProvider.SPANNER and edge_data['valid_at'] is None:
                edge_data['valid_at'] = edge.created_at  # Default to creation time
            # expired_at and invalid_at remain None (NULL in database)
        else:
            edge_data.update(edge.attributes or {})

        edges.append(edge_data)

    prep_time = (time() - step_start) * 1000
    db_start = time()

    if driver.provider in (GraphProvider.KUZU, GraphProvider.SPANNER):
        logger.debug(f'[PROFILING] Data preparation: {prep_time:.2f}ms')
        logger.info(
            f'[PROFILING] Starting inserts for {len(episodes)} episodes, {len(nodes)} nodes, {len(edges)} entity edges, {len(episodic_edges)} episodic edges'
        )

        if driver.provider == GraphProvider.SPANNER:
            # Use Mutation API for optimal bulk insert performance
            # Mutations are 2-5x faster than Batch DML as they bypass SQL parsing
            logger.info(
                '[PROFILING] Using Spanner MUTATION API - direct bulk inserts without SQL parsing'
            )
            insert_start = time()

            # Build mutations list
            mutations = []

            # Add episode inserts as mutations
            if episodes:
                episode_columns = [
                    'source',
                    'source_description',
                    'content',
                    'entity_edges',
                    'uuid',
                    'name',
                    'group_id',
                    'created_at',
                    'valid_at',
                ]
                for episode in episodes:
                    mutation_data = {
                        'table': 'EpisodicNode',
                        'columns': episode_columns,
                        'values': [
                            episode['source'],
                            episode.get('source_description', ''),
                            episode.get('content', ''),
                            episode.get('entity_edges', []),
                            episode['uuid'],
                            episode['name'],
                            episode['group_id'],
                            episode['created_at'],
                            episode['valid_at'],
                        ],
                    }
                    mutations.append(mutation_data)

            # Add entity node inserts as mutations
            if nodes:
                node_columns = [
                    'uuid',
                    'name',
                    'group_id',
                    'labels',
                    'created_at',
                    'name_embedding',
                    'summary',
                    'attributes',
                ]
                for node in nodes:
                    mutation_data = {
                        'table': 'EntityNode',
                        'columns': node_columns,
                        'values': [
                            node['uuid'],
                            node['name'],
                            node['group_id'],
                            node.get('labels', []),
                            node['created_at'],
                            node.get('name_embedding', []),
                            node.get('summary', ''),
                            node.get('attributes', '{}'),
                        ],
                    }
                    mutations.append(mutation_data)

            # Add entity edge inserts as mutations
            if edges:
                edge_columns = [
                    'uuid',
                    'source_node_uuid',
                    'target_node_uuid',
                    'name',
                    'fact',
                    'group_id',
                    'episodes',
                    'created_at',
                    'expired_at',
                    'valid_at',
                    'invalid_at',
                    'fact_embedding',
                    'attributes',
                    'labels',
                ]
                for edge in edges:
                    mutation_data = {
                        'table': 'EntityEdge',
                        'columns': edge_columns,
                        'values': [
                            edge['uuid'],
                            edge['source_node_uuid'],
                            edge['target_node_uuid'],
                            edge['name'],
                            edge['fact'],
                            edge['group_id'],
                            edge.get('episodes', []),
                            edge['created_at'],
                            edge.get('expired_at'),
                            edge.get('valid_at'),
                            edge.get('invalid_at'),
                            edge.get('fact_embedding', []),
                            edge.get('attributes', '{}'),
                            edge.get('labels', []),
                        ],
                    }
                    mutations.append(mutation_data)

            # Add episodic edge inserts as mutations
            if episodic_edges:
                # Note: EpisodicEdge model doesn't have 'name' field, only base fields
                episodic_edge_columns = [
                    'uuid',
                    'source_node_uuid',
                    'target_node_uuid',
                    'group_id',
                    'created_at',
                ]
                for edge in episodic_edges:
                    edge_dict = edge.model_dump()
                    mutation_data = {
                        'table': 'EpisodicEdge',
                        'columns': episodic_edge_columns,
                        'values': [
                            edge_dict['uuid'],
                            edge_dict['source_node_uuid'],
                            edge_dict['target_node_uuid'],
                            edge_dict['group_id'],
                            edge_dict['created_at'],
                        ],
                    }
                    mutations.append(mutation_data)

            # Execute ALL inserts as mutations
            logger.info(
                f'[PROFILING]   - Executing {len(mutations)} mutations ({len(episodes)} episodes + {len(nodes)} nodes + {len(edges)} entity edges + {len(episodic_edges)} episodic edges)'
            )
            await tx.run_mutations(mutations)  # type: ignore[attr-defined]

            total_time = (time() - insert_start) * 1000
            avg_time_per_mutation = total_time / len(mutations) if mutations else 0
            logger.info(
                f'[PROFILING]   - Mutations completed: {total_time:.2f}ms ({len(mutations)} total mutations, {avg_time_per_mutation:.2f}ms per mutation)'
            )
        else:
            # KUZU - keep original one-by-one logic
            insert_start = time()
            episode_query = get_episode_node_save_bulk_query(driver.provider)
            for episode in episodes:
                await tx.run(episode_query, **episode)
            episodes_time = (time() - insert_start) * 1000
            logger.info(
                f'[PROFILING]   - Episodes inserted: {episodes_time:.2f}ms ({len(episodes)} rows, {episodes_time / len(episodes) if episodes else 0:.2f}ms per row)'
            )

            insert_start = time()
            entity_node_query = get_entity_node_save_bulk_query(driver.provider, nodes)
            for node in nodes:
                await tx.run(entity_node_query, **node)
            nodes_time = (time() - insert_start) * 1000
            logger.info(
                f'[PROFILING]   - Entity nodes inserted: {nodes_time:.2f}ms ({len(nodes)} rows, {nodes_time / len(nodes) if nodes else 0:.2f}ms per row)'
            )

            insert_start = time()
            entity_edge_query = get_entity_edge_save_bulk_query(driver.provider)
            for edge in edges:
                await tx.run(entity_edge_query, **edge)
            entity_edges_time = (time() - insert_start) * 1000
            logger.info(
                f'[PROFILING]   - Entity edges inserted: {entity_edges_time:.2f}ms ({len(edges)} rows, {entity_edges_time / len(edges) if edges else 0:.2f}ms per row)'
            )

            insert_start = time()
            episodic_edge_query = get_episodic_edge_save_bulk_query(driver.provider)
            for edge in episodic_edges:
                await tx.run(episodic_edge_query, **edge.model_dump())
            episodic_edges_time = (time() - insert_start) * 1000
            logger.info(
                f'[PROFILING]   - Episodic edges inserted: {episodic_edges_time:.2f}ms ({len(episodic_edges)} rows, {episodic_edges_time / len(episodic_edges) if episodic_edges else 0:.2f}ms per row)'
            )

        total_db_time = (time() - db_start) * 1000
        logger.debug(f'[PROFILING] Total database write time: {total_db_time:.2f}ms')
    else:
        await tx.run(get_episode_node_save_bulk_query(driver.provider), episodes=episodes)
        await tx.run(
            get_entity_node_save_bulk_query(
                driver.provider, nodes, has_aoss=bool(driver.aoss_client)
            ),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        await tx.run(
            get_entity_edge_save_bulk_query(driver.provider, has_aoss=bool(driver.aoss_client)),
            entity_edges=edges,
        )

        if bool(driver.aoss_client):
            for node_data, entity_node in zip(nodes, entity_nodes, strict=True):
                if node_data.get('uuid') == entity_node.uuid:
                    node_data['name_embedding'] = entity_node.name_embedding

            for edge_data, entity_edge in zip(edges, entity_edges, strict=True):
                if edge_data.get('uuid') == entity_edge.uuid:
                    edge_data['fact_embedding'] = entity_edge.fact_embedding

            await driver.save_to_aoss(EPISODE_INDEX_NAME, episodes)
            await driver.save_to_aoss(ENTITY_INDEX_NAME, nodes)
            await driver.save_to_aoss(ENTITY_EDGE_INDEX_NAME, edges)


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    extracted_nodes_bulk: list[list[EntityNode]] = await semaphore_gather(
        *[
            extract_nodes(clients, episode, previous_episodes, entity_types, excluded_entity_types)
            for episode, previous_episodes in episode_tuples
        ]
    )

    extracted_edges_bulk: list[list[EntityEdge]] = await semaphore_gather(
        *[
            extract_edges(
                clients,
                episode,
                extracted_nodes_bulk[i],
                previous_episodes,
                edge_type_map=edge_type_map,
                group_id=episode.group_id,
                edge_types=edge_types,
            )
            for i, (episode, previous_episodes) in enumerate(episode_tuples)
        ]
    )

    return extracted_nodes_bulk, extracted_edges_bulk


async def dedupe_nodes_bulk(
    clients: GraphitiClients,
    extracted_nodes: list[list[EntityNode]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    embedder = clients.embedder
    min_score = 0.8

    # generate embeddings
    await semaphore_gather(
        *[create_entity_node_embeddings(embedder, nodes) for nodes in extracted_nodes]
    )

    # Find similar results
    dedupe_tuples: list[tuple[list[EntityNode], list[EntityNode]]] = []
    for i, nodes_i in enumerate(extracted_nodes):
        existing_nodes: list[EntityNode] = []
        for j, nodes_j in enumerate(extracted_nodes):
            if i == j:
                continue
            existing_nodes += nodes_j

        candidates_i: list[EntityNode] = []
        for node in nodes_i:
            for existing_node in existing_nodes:
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                node_words = set(node.name.lower().split())
                existing_node_words = set(existing_node.name.lower().split())
                has_overlap = not node_words.isdisjoint(existing_node_words)
                if has_overlap:
                    candidates_i.append(existing_node)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(node.name_embedding or []),
                    normalize_l2(existing_node.name_embedding or []),
                )
                if similarity >= min_score:
                    candidates_i.append(existing_node)

        dedupe_tuples.append((nodes_i, candidates_i))

    # Determine Node Resolutions
    bulk_node_resolutions: list[
        tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_nodes(
                clients,
                dedupe_tuple[0],
                episode_tuples[i][0],
                episode_tuples[i][1],
                entity_types,
                existing_nodes_override=dedupe_tuples[i][1],
            )
            for i, dedupe_tuple in enumerate(dedupe_tuples)
        ]
    )

    # Collect all duplicate pairs sorted by uuid
    duplicate_pairs: list[tuple[str, str]] = []
    for _, _, duplicates in bulk_node_resolutions:
        for duplicate in duplicates:
            n, m = duplicate
            duplicate_pairs.append((n.uuid, m.uuid))

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_pairs)

    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for nodes in extracted_nodes for node in nodes
    }

    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for i, nodes in enumerate(extracted_nodes):
        episode = episode_tuples[i][0]

        nodes_by_episode[episode.uuid] = [
            node_uuid_map[compressed_map.get(node.uuid, node.uuid)] for node in nodes
        ]

    return nodes_by_episode, compressed_map


async def dedupe_edges_bulk(
    clients: GraphitiClients,
    extracted_edges: list[list[EntityEdge]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    _entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    _edge_type_map: dict[tuple[str, str], list[str]],
) -> dict[str, list[EntityEdge]]:
    embedder = clients.embedder
    min_score = 0.6

    # generate embeddings
    await semaphore_gather(
        *[create_entity_edge_embeddings(embedder, edges) for edges in extracted_edges]
    )

    # Find similar results
    dedupe_tuples: list[tuple[EpisodicNode, EntityEdge, list[EntityEdge]]] = []
    for i, edges_i in enumerate(extracted_edges):
        existing_edges: list[EntityEdge] = []
        for j, edges_j in enumerate(extracted_edges):
            if i == j:
                continue
            existing_edges += edges_j

        for edge in edges_i:
            candidates: list[EntityEdge] = []
            for existing_edge in existing_edges:
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                if (
                    edge.source_node_uuid != existing_edge.source_node_uuid
                    or edge.target_node_uuid != existing_edge.target_node_uuid
                ):
                    continue

                edge_words = set(edge.fact.lower().split())
                existing_edge_words = set(existing_edge.fact.lower().split())
                has_overlap = not edge_words.isdisjoint(existing_edge_words)
                if has_overlap:
                    candidates.append(existing_edge)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(edge.fact_embedding or []),
                    normalize_l2(existing_edge.fact_embedding or []),
                )
                if similarity >= min_score:
                    candidates.append(existing_edge)

            dedupe_tuples.append((episode_tuples[i][0], edge, candidates))

    bulk_edge_resolutions: list[
        tuple[EntityEdge, EntityEdge, list[EntityEdge]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_edge(
                clients.llm_client,
                edge,
                candidates,
                candidates,
                episode,
                edge_types,
                clients.ensure_ascii,
            )
            for episode, edge, candidates in dedupe_tuples
        ]
    )

    # For now we won't track edge invalidation
    duplicate_pairs: list[tuple[str, str]] = []
    for i, (_, _, duplicates) in enumerate(bulk_edge_resolutions):
        episode, edge, candidates = dedupe_tuples[i]
        for duplicate in duplicates:
            duplicate_pairs.append((edge.uuid, duplicate.uuid))

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_pairs)

    edge_uuid_map: dict[str, EntityEdge] = {
        edge.uuid: edge for edges in extracted_edges for edge in edges
    }

    edges_by_episode: dict[str, list[EntityEdge]] = {}
    for i, edges in enumerate(extracted_edges):
        episode = episode_tuples[i][0]

        edges_by_episode[episode.uuid] = [
            edge_uuid_map[compressed_map.get(edge.uuid, edge.uuid)] for edge in edges
        ]

    return edges_by_episode


class UnionFind:
    def __init__(self, elements):
        # start each element in its own set
        self.parent = {e: e for e in elements}

    def find(self, x):
        # path‐compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # attach the lexicographically larger root under the smaller
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def compress_uuid_map(duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    all_ids: iterable of all entity IDs (strings)
    duplicate_pairs: iterable of (id1, id2) pairs
    returns: dict mapping each id -> lexicographically smallest id in its duplicate set
    """
    all_uuids = set()
    for pair in duplicate_pairs:
        all_uuids.add(pair[0])
        all_uuids.add(pair[1])

    uf = UnionFind(all_uuids)
    for a, b in duplicate_pairs:
        uf.union(a, b)
    # ensure full path‐compression before mapping
    return {uuid: uf.find(uuid) for uuid in all_uuids}


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges
