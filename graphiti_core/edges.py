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

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.driver.driver import ENTITY_EDGE_INDEX_NAME, GraphDriver, GraphProvider
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.edges.edge_db_queries import (
    COMMUNITY_EDGE_RETURN,
    EPISODIC_EDGE_RETURN,
    EPISODIC_EDGE_SAVE,
    get_community_edge_save_query,
    get_entity_edge_return_query,
    get_entity_edge_save_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.nodes import Node

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        if driver.provider == GraphProvider.KUZU:
            await driver.execute_query(
                """
                MATCH (n)-[e:MENTIONS|HAS_MEMBER {uuid: $uuid}]->(m)
                DELETE e
                """,
                uuid=self.uuid,
            )
            await driver.execute_query(
                """
                MATCH (e:RelatesToNode_ {uuid: $uuid})
                DETACH DELETE e
                """,
                uuid=self.uuid,
            )
        else:
            await driver.execute_query(
                """
                MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
                DELETE e
                """,
                uuid=self.uuid,
            )

            if driver.aoss_client:
                await driver.aoss_client.delete(
                    index=ENTITY_EDGE_INDEX_NAME,
                    id=self.uuid,
                    params={'routing': self.group_id},
                )

        logger.debug(f'Deleted Edge: {self.uuid}')

    @classmethod
    async def delete_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == GraphProvider.KUZU:
            await driver.execute_query(
                """
                MATCH (n)-[e:MENTIONS|HAS_MEMBER]->(m)
                WHERE e.uuid IN $uuids
                DELETE e
                """,
                uuids=uuids,
            )
            await driver.execute_query(
                """
                MATCH (e:RelatesToNode_)
                WHERE e.uuid IN $uuids
                DETACH DELETE e
                """,
                uuids=uuids,
            )
        else:
            await driver.execute_query(
                """
                MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)
                WHERE e.uuid IN $uuids
                DELETE e
                """,
                uuids=uuids,
            )

            if driver.aoss_client:
                await driver.aoss_client.delete_by_query(
                    index=ENTITY_EDGE_INDEX_NAME,
                    body={'query': {'terms': {'uuid': uuids}}},
                )

        logger.debug(f'Deleted Edges: {uuids}')

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str): ...


class EpisodicEdge(Edge):
    async def save(self, driver: GraphDriver):
        if driver.provider == GraphProvider.SPANNER:
            # Spanner uses individual params with exact column names
            result = await driver.execute_query(
                get_episodic_edge_save_bulk_query(driver.provider),
                uuid=self.uuid,
                source_node_uuid=self.source_node_uuid,
                target_node_uuid=self.target_node_uuid,
                group_id=self.group_id,
                created_at=self.created_at,
            )
        else:
            result = await driver.execute_query(
                EPISODIC_EDGE_SAVE,
                episode_uuid=self.source_node_uuid,
                entity_uuid=self.target_node_uuid,
                uuid=self.uuid,
                group_id=self.group_id,
                created_at=self.created_at,
            )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
            RETURN
            """
            + EPISODIC_EDGE_RETURN,
            uuid=uuid,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + EPISODIC_EDGE_RETURN,
            uuids=uuids,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuids[0])
        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + EPISODIC_EDGE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges


class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the edge. Dependent on edge name'
    )

    async def generate_embedding(self, embedder: EmbedderClient):
        start = time()

        text = self.fact.replace('\n', ' ')
        self.fact_embedding = await embedder.create(input_data=[text])

        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.fact_embedding

    async def load_fact_embedding(self, driver: GraphDriver):
        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            RETURN e.fact_embedding AS fact_embedding
        """

        if driver.provider == GraphProvider.NEPTUNE:
            query = """
                MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
                RETURN [x IN split(e.fact_embedding, ",") | toFloat(x)] as fact_embedding
            """
        elif driver.aoss_client:
            resp = await driver.aoss_client.search(
                body={
                    'query': {'multi_match': {'query': self.uuid, 'fields': ['uuid']}},
                    'size': 1,
                },
                index=ENTITY_EDGE_INDEX_NAME,
                params={'routing': self.group_id},
            )

            if resp['hits']['hits']:
                self.fact_embedding = resp['hits']['hits'][0]['_source']['fact_embedding']
                return
            else:
                raise EdgeNotFoundError(self.uuid)

        if driver.provider == GraphProvider.KUZU:
            query = """
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
                RETURN e.fact_embedding AS fact_embedding
            """

        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise EdgeNotFoundError(self.uuid)

        self.fact_embedding = records[0]['fact_embedding']

    async def save(self, driver: GraphDriver):
        edge_data: dict[str, Any] = {
            'source_uuid': self.source_node_uuid,
            'target_uuid': self.target_node_uuid,
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'fact': self.fact,
            'fact_embedding': self.fact_embedding,
            'episodes': self.episodes,
            'created_at': self.created_at,
            'expired_at': self.expired_at,
            'valid_at': self.valid_at,
            'invalid_at': self.invalid_at,
        }

        if driver.provider == GraphProvider.KUZU:
            edge_data['attributes'] = json.dumps(self.attributes)
            result = await driver.execute_query(
                get_entity_edge_save_query(driver.provider, has_aoss=bool(driver.aoss_client)),
                **edge_data,
            )
        elif driver.provider == GraphProvider.SPANNER:
            # Spanner uses individual params with different column names
            result = await driver.execute_query(
                get_entity_edge_save_query(driver.provider),
                uuid=self.uuid,
                source_node_uuid=self.source_node_uuid,
                target_node_uuid=self.target_node_uuid,
                group_id=self.group_id,
                created_at=self.created_at,
                name=self.name,
                fact=self.fact,
                fact_embedding=self.fact_embedding,
                episodes=self.episodes,
                expired_at=self.expired_at,
                valid_at=self.valid_at,
                invalid_at=self.invalid_at,
                attributes=json.dumps(self.attributes) if self.attributes else '{}',
            )
        else:
            edge_data.update(self.attributes or {})

            if driver.aoss_client:
                await driver.save_to_aoss(ENTITY_EDGE_INDEX_NAME, [edge_data])  # pyright: ignore reportAttributeAccessIssue

            result = await driver.execute_query(
                get_entity_edge_save_query(driver.provider),
                edge_data=edge_data,
            )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
            """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + get_entity_edge_return_query(driver.provider),
            uuid=uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_between_nodes(
        cls, driver: GraphDriver, source_node_uuid: str, target_node_uuid: str
    ):
        match_query = """
            MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity {uuid: $source_node_uuid})
                      -[:RELATES_TO]->(e:RelatesToNode_)
                      -[:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
            """
        elif driver.provider == GraphProvider.SPANNER:
            match_query = """
                GRAPH GRAPHITI
                MATCH (n:Entity {uuid: @source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: @target_node_uuid})
            """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + get_entity_edge_return_query(driver.provider),
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges

    @classmethod
    async def get_between_nodes_batch(
        cls,
        driver: GraphDriver,
        node_pairs: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list['EntityEdge']]:
        """
        Batch lookup of edges between multiple node pairs in a single query.
        
        This is an optimization that reduces N separate get_between_nodes calls
        into a single database query, significantly reducing latency and CPU usage.
        
        Args:
            driver: Graph database driver
            node_pairs: List of (source_node_uuid, target_node_uuid) tuples
            
        Returns:
            Dict mapping (source_uuid, target_uuid) -> list of edges between them
        """
        if not node_pairs:
            return {}
        
        # Deduplicate pairs while preserving order for result mapping
        unique_pairs = list(set(node_pairs))
        
        if driver.provider == GraphProvider.SPANNER:
            # Build query with OR conditions for each pair
            # This is more compatible with Spanner's parameter binding
            if len(unique_pairs) == 1:
                # Single pair - simple query
                spanner_query = """
                    SELECT uuid, name, fact, group_id, created_at, expired_at, 
                           valid_at, invalid_at, source_node_uuid, target_node_uuid, 
                           fact_embedding, episodes, attributes, labels
                    FROM EntityEdge
                    WHERE source_node_uuid = @src0 AND target_node_uuid = @tgt0
                """
                params = {'src0': unique_pairs[0][0], 'tgt0': unique_pairs[0][1]}
            else:
                # Multiple pairs - build OR conditions
                # (source_node_uuid = @src0 AND target_node_uuid = @tgt0) OR (...)
                conditions = []
                params = {}
                for i, (src, tgt) in enumerate(unique_pairs):
                    conditions.append(f'(source_node_uuid = @src{i} AND target_node_uuid = @tgt{i})')
                    params[f'src{i}'] = src
                    params[f'tgt{i}'] = tgt
                
                where_clause = ' OR '.join(conditions)
                spanner_query = f"""
                    SELECT uuid, name, fact, group_id, created_at, expired_at, 
                           valid_at, invalid_at, source_node_uuid, target_node_uuid, 
                           fact_embedding, episodes, attributes, labels
                    FROM EntityEdge
                    WHERE {where_clause}
                """
            
            records, _, _ = await driver.execute_query(
                spanner_query,
                routing_='r',
                **params,
            )
        else:
            # For Neo4j and other providers, use UNWIND with list of maps
            if driver.provider == GraphProvider.KUZU:
                match_query = """
                    UNWIND $pairs AS pair
                    MATCH (n:Entity {uuid: pair.source})
                          -[:RELATES_TO]->(e:RelatesToNode_)
                          -[:RELATES_TO]->(m:Entity {uuid: pair.target})
                """
            else:
                match_query = """
                    UNWIND $pairs AS pair
                    MATCH (n:Entity {uuid: pair.source})-[e:RELATES_TO]->(m:Entity {uuid: pair.target})
                """
            
            pairs_param = [{'source': p[0], 'target': p[1]} for p in unique_pairs]
            
            records, _, _ = await driver.execute_query(
                match_query
                + """
                RETURN
                """
                + get_entity_edge_return_query(driver.provider),
                pairs=pairs_param,
                routing_='r',
            )
        
        # Parse results and group by (source, target) pair
        result: dict[tuple[str, str], list[EntityEdge]] = {pair: [] for pair in unique_pairs}
        
        for record in records:
            edge = get_entity_edge_from_record(record, driver.provider)
            key = (edge.source_node_uuid, edge.target_node_uuid)
            if key in result:
                result[key].append(edge)
        
        return result

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if len(uuids) == 0:
            return []

        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            WHERE e.uuid IN $uuids
            RETURN
            """
            + get_entity_edge_return_query(driver.provider),
            uuids=uuids,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
        with_embeddings: bool = False,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        with_embeddings_query: LiteralString = (
            """,
                e.fact_embedding AS fact_embedding
                """
            if with_embeddings
            else ''
        )

        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + get_entity_edge_return_query(driver.provider)
            + with_embeddings_query
            + """
            ORDER BY e.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        if len(edges) == 0:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    @classmethod
    async def get_by_node_uuid(cls, driver: GraphDriver, node_uuid: str):
        match_query = """
            MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(m:Entity)
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + get_entity_edge_return_query(driver.provider),
            node_uuid=node_uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges


class CommunityEdge(Edge):
    async def save(self, driver: GraphDriver):
        result = await driver.execute_query(
            get_community_edge_save_query(driver.provider),
            community_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m)
            RETURN
            """
            + COMMUNITY_EDGE_RETURN,
            uuid=uuid,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Community)-[e:HAS_MEMBER]->(m)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + COMMUNITY_EDGE_RETURN,
            uuids=uuids,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Community)-[e:HAS_MEMBER]->(m)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + COMMUNITY_EDGE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges


# Edge helpers
def get_episodic_edge_from_record(record: Any) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


def get_entity_edge_from_record(record: Any, provider: GraphProvider) -> EntityEdge:
    episodes = record['episodes']
    if provider in (GraphProvider.KUZU, GraphProvider.SPANNER):
        attributes = json.loads(record['attributes']) if record['attributes'] else {}
    else:
        attributes = record['attributes']
        attributes.pop('uuid', None)
        attributes.pop('source_node_uuid', None)
        attributes.pop('target_node_uuid', None)
        attributes.pop('fact', None)
        attributes.pop('fact_embedding', None)
        attributes.pop('name', None)
        attributes.pop('group_id', None)
        attributes.pop('episodes', None)
        attributes.pop('created_at', None)
        attributes.pop('expired_at', None)
        attributes.pop('valid_at', None)
        attributes.pop('invalid_at', None)

    edge = EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=episodes,
        created_at=parse_db_date(record['created_at']),  # type: ignore
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
        attributes=attributes,
    )

    return edge


def get_community_edge_from_record(record: Any):
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


async def create_entity_edge_embeddings(embedder: EmbedderClient, edges: list[EntityEdge]):
    if len(edges) == 0:
        return
    fact_embeddings = await embedder.create_batch([edge.fact for edge in edges])
    for edge, fact_embedding in zip(edges, fact_embeddings, strict=True):
        edge.fact_embedding = fact_embedding
