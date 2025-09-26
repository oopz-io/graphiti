"""
Database query utilities for different graph database backends.

This module provides database-agnostic query generation for Neo4j and FalkorDB,
supporting index creation, fulltext search, and bulk operations.
"""

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphProvider

# Mapping from Neo4j fulltext index names to FalkorDB node labels
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}
# Mapping from fulltext index names to Kuzu node labels
INDEX_TO_LABEL_KUZU_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RelatesToNode_',
}
# Mapping from fulltext index names to Spanner node labels
INDEX_TO_LABEL_SPANNER_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}


def get_range_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        return [
            # Entity node
            'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)',
            # Episodic node
            'CREATE INDEX FOR (n:Episodic) ON (n.uuid, n.group_id, n.created_at, n.valid_at)',
            # Community node
            'CREATE INDEX FOR (n:Community) ON (n.uuid)',
            # RELATES_TO edge
            'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)',
            # MENTIONS edge
            'CREATE INDEX FOR ()-[e:MENTIONS]-() ON (e.uuid, e.group_id)',
            # HAS_MEMBER edge
            'CREATE INDEX FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        ]

    if provider == GraphProvider.KUZU:
        return []

    if provider == GraphProvider.SPANNER:
        # Spanner Graph uses different index syntax
        # These would be property graph schema definitions rather than indexes
        return [
            # Property graph schema creation for Spanner would be done via DDL
            # This is a placeholder - actual Spanner graph schema would be different
            'CREATE PROPERTY GRAPH IF NOT EXISTS graphiti_graph',
            'CREATE OR REPLACE PROPERTY GRAPH graphiti_graph NODE TABLES (Entity, Episodic, Community) EDGE TABLES (RELATES_TO, MENTIONS, HAS_MEMBER)',
        ]

    return [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
        'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
        'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
        'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
        'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
        'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
        'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
        'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
        'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
        'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
    ]


def get_fulltext_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        from typing import cast

        from graphiti_core.driver.falkordb_driver import STOPWORDS

        # Convert to string representation for embedding in queries
        stopwords_str = str(STOPWORDS)

        # Use type: ignore to satisfy LiteralString requirement while maintaining single source of truth
        return cast(
            list[LiteralString],
            [
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Episodic',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'content', 'source', 'source_description', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Entity',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'summary', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Community',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'group_id'
                                                )""",
                """CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
            ],
        )

    if provider == GraphProvider.KUZU:
        return [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]

    if provider == GraphProvider.SPANNER:
        # Spanner Graph fulltext search is integrated into GQL queries
        # No separate fulltext index creation - it's handled by the property graph schema
        return [
            # Placeholder - Spanner handles fulltext search differently in GQL queries
            '-- Spanner Graph fulltext search is handled in GQL queries directly',
        ]

    return [
        """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
        FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]


def get_nodes_query(name: str, query: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', {query}, TOP := $limit)"

    if provider == GraphProvider.SPANNER:
        label = INDEX_TO_LABEL_SPANNER_MAPPING[name]
        # Spanner GQL uses text search in WHERE clauses
        return f'MATCH (n:{label}) WHERE SEARCH(n, {query}) RETURN n LIMIT {limit}'

    return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'


def get_vector_cosine_func_query(vec1, vec2, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        # FalkorDB uses a different syntax for regular cosine similarity and Neo4j uses normalized cosine similarity
        return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'

    if provider == GraphProvider.KUZU:
        return f'array_cosine_similarity({vec1}, {vec2})'

    if provider == GraphProvider.SPANNER:
        # Spanner uses COSINE_DISTANCE which returns distance (lower is better)
        # Convert to similarity by subtracting from 1
        return f'(1 - COSINE_DISTANCE({vec1}, {vec2}))'

    return f'vector.similarity.cosine({vec1}, {vec2})'


def get_relationships_query(name: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', cast($query AS STRING), TOP := $limit)"

    if provider == GraphProvider.SPANNER:
        label = INDEX_TO_LABEL_SPANNER_MAPPING[name]
        # Spanner GQL relationship search
        return f'MATCH ()-[r:{label}]-() WHERE SEARCH(r, $query) RETURN r LIMIT {limit}'

    return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'
