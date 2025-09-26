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
from collections.abc import Coroutine
from typing import Any

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

try:
    from google.cloud import spanner
    from google.cloud.spanner_v1 import transaction as spanner_transaction

    _HAS_SPANNER = True
except ImportError:
    spanner = None  # type: ignore
    spanner_transaction = None  # type: ignore
    _HAS_SPANNER = False


class SpannerDriverSession(GraphDriverSession):
    provider = GraphProvider.SPANNER

    def __init__(self, session, database_name: str):
        self._session = session
        self._database_name = database_name

    async def __aexit__(self, exc_type, exc, tb):
        # Spanner sessions are managed by the pool automatically
        pass

    async def run(self, query: str, **kwargs: Any) -> Any:
        """
        Execute a GQL query using the Spanner session.
        Converts Cypher-like queries to GQL format.
        """
        # Convert Cypher to GQL
        gql_query = self._convert_cypher_to_gql(query, **kwargs)

        def execute_query():
            with self._session.database.snapshot() as snapshot:
                return snapshot.execute_sql(gql_query)

        # For write operations, use a transaction
        if self._is_write_operation(query):

            def execute_write(transaction):
                return transaction.execute_sql(gql_query)

            result = self._session.database.run_in_transaction(execute_write)
        else:
            result = execute_query()

        return self._format_result(result)

    async def close(self):
        """Close the session - handled automatically by Spanner client"""
        pass

    async def execute_write(self, func, *args, **kwargs):
        """Execute a write transaction"""
        return self._session.database.run_in_transaction(func, *args, **kwargs)

    def _convert_cypher_to_gql(self, cypher_query: str, **kwargs) -> str:
        """
        Convert Cypher queries to Google Cloud Spanner GQL format.
        This is a simplified conversion - in production, you'd want a more robust parser.
        """
        # Basic GQL conversion patterns
        gql_query = cypher_query

        # Add GRAPH clause at the beginning if not present
        if not gql_query.strip().upper().startswith('GRAPH'):
            gql_query = f'GRAPH {self._database_name}\n{gql_query}'

        # Convert MATCH patterns
        # Cypher: MATCH (n:Label {prop: value})
        # GQL: MATCH (n:Label {prop: value})
        # GQL is quite similar to Cypher for basic patterns

        # Convert MERGE to CREATE/MATCH pattern
        if 'MERGE' in gql_query.upper():
            # This is a simplified conversion - real implementation would need proper parsing
            gql_query = gql_query.replace('MERGE', 'MATCH')
            logger.warning('MERGE converted to MATCH - may need manual adjustment for upsert logic')

        # Handle parameter substitution
        for param_name, param_value in kwargs.get('params', {}).items():
            if isinstance(param_value, str):
                gql_query = gql_query.replace(f'${param_name}', f"'{param_value}'")
            else:
                gql_query = gql_query.replace(f'${param_name}', str(param_value))

        return gql_query

    def _is_write_operation(self, query: str) -> bool:
        """Check if the query is a write operation"""
        write_keywords = ['CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP']
        query_upper = query.upper()
        return any(keyword in query_upper for keyword in write_keywords)

    def _format_result(self, result) -> Any:
        """Format Spanner result to match expected format"""
        # Convert Spanner result to a format compatible with other drivers
        if hasattr(result, 'rows'):
            return result
        return result


class SpannerDriver(GraphDriver):
    provider = GraphProvider.SPANNER
    fulltext_syntax = ''  # Spanner GQL syntax for fulltext queries

    def __init__(
        self,
        spanner_project_id: str,
        spanner_instance_id: str,
        spanner_database_id: str,
    ):
        super().__init__()

        if not _HAS_SPANNER:
            raise ImportError(
                'Google Cloud Spanner not available. Install with: pip install google-cloud-spanner'
            )

        self.spanner_project_id = spanner_project_id
        self.spanner_instance_id = spanner_instance_id
        self.spanner_database_id = spanner_database_id
        self._database = spanner_database_id

        # Initialize Spanner client
        self.spanner_client = spanner.Client(project=spanner_project_id)  # type: ignore
        self.spanner_instance = self.spanner_client.instance(spanner_instance_id)
        self.spanner_database = self.spanner_instance.database(spanner_database_id)

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> Any:
        """Execute a query against Spanner Graph database"""
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)

        try:
            # Convert Cypher to GQL
            gql_query = self._convert_cypher_to_gql(str(cypher_query_), params)

            # Execute based on query type
            if self._is_write_operation(str(cypher_query_)):

                def execute_write(transaction):
                    return transaction.execute_sql(gql_query)

                result = self.spanner_database.run_in_transaction(execute_write)
            else:
                with self.spanner_database.snapshot() as snapshot:
                    result = snapshot.execute_sql(gql_query)

        except Exception as e:
            logger.error(
                f'Error executing Spanner query: {e}\nQuery: {cypher_query_}\nParams: {params}'
            )
            raise

        return self._format_result(result)

    def session(self, database: str | None = None) -> GraphDriverSession:
        """Create a new database session"""
        database_name = database or self._database

        # Spanner sessions are managed by the client library automatically
        # We create a lightweight wrapper that holds reference to the database
        class SpannerSession:
            def __init__(self, spanner_database):
                self.database = spanner_database

        session = SpannerSession(self.spanner_database)
        return SpannerDriverSession(session, database_name)  # type: ignore

    async def close(self) -> None:
        """Close the Spanner client"""
        try:
            self.spanner_client.close()
        except Exception as e:
            logger.warning(f'Error closing Spanner client: {e}')

    def delete_all_indexes(self) -> Coroutine:
        """Delete all graph indexes in Spanner"""

        # Spanner Graph indexes are managed differently
        # This would involve dropping property graph schema elements
        async def delete_spanner_indexes():
            try:
                # For now, return a placeholder - actual implementation would
                # need to query INFORMATION_SCHEMA for existing indexes
                logger.warning('Spanner index deletion not yet implemented')
                return True
            except Exception as e:
                logger.error(f'Error deleting Spanner indexes: {e}')
                return False

        return delete_spanner_indexes()

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Build fulltext query for Spanner Graph database.
        Spanner Graph uses different syntax for fulltext search.
        """
        # Truncate query if needed
        if len(query) > max_query_length:
            query = query[:max_query_length]

        # Escape special characters for GQL fulltext search
        escaped_query = query.replace("'", "\\'").replace('"', '\\"')

        # Build GQL fulltext search expression
        fulltext_query = f"'{escaped_query}'"

        # Add group_id filtering if specified
        if group_ids:
            group_filter = ' OR '.join([f"group_id = '{gid}'" for gid in group_ids])
            fulltext_query = f'({fulltext_query}) AND ({group_filter})'

        return fulltext_query

    def build_vector_search_query(
        self,
        embedding: list[float],
        similarity_function: str = 'COSINE_DISTANCE',
        limit: int = 10,
        threshold: float | None = None,
        node_labels: list[str] | None = None,
        group_ids: list[str] | None = None,
    ) -> str:
        """
        Build vector similarity search query using Spanner's native vector functions.

        Args:
            embedding: Query vector for similarity search
            similarity_function: 'COSINE_DISTANCE', 'EUCLIDEAN_DISTANCE', or 'DOT_PRODUCT'
            limit: Maximum number of results to return
            threshold: Optional similarity threshold
            node_labels: Filter by node labels
            group_ids: Filter by group IDs

        Returns:
            GQL query string for vector similarity search
        """
        # Convert embedding to Spanner ARRAY format
        embedding_str = '[' + ', '.join(map(str, embedding)) + ']'

        # Choose similarity function
        if similarity_function.upper() == 'DOT_PRODUCT':
            # For dot product, we want higher scores to be better
            similarity_expr = f'DOT_PRODUCT(n.embedding, {embedding_str})'
            order_direction = 'DESC'
        elif similarity_function.upper() == 'EUCLIDEAN_DISTANCE':
            similarity_expr = f'EUCLIDEAN_DISTANCE(n.embedding, {embedding_str})'
            order_direction = 'ASC'
        else:  # Default to COSINE_DISTANCE
            similarity_expr = f'COSINE_DISTANCE(n.embedding, {embedding_str})'
            order_direction = 'ASC'

        # Build WHERE clause conditions
        where_conditions = []

        # Add node label filter
        if node_labels:
            label_conditions = ' OR '.join(
                [f"'{label}' IN UNNEST(LABELS(n))" for label in node_labels]
            )
            where_conditions.append(f'({label_conditions})')

        # Add group ID filter
        if group_ids:
            group_conditions = ' OR '.join([f"n.group_id = '{gid}'" for gid in group_ids])
            where_conditions.append(f'({group_conditions})')

        # Add similarity threshold filter
        if threshold is not None:
            if similarity_function.upper() == 'DOT_PRODUCT':
                where_conditions.append(f'{similarity_expr} >= {threshold}')
            else:  # COSINE_DISTANCE or EUCLIDEAN_DISTANCE
                where_conditions.append(f'{similarity_expr} <= {threshold}')

        # Combine WHERE conditions
        where_clause = ''
        if where_conditions:
            where_clause = 'WHERE ' + ' AND '.join(where_conditions)

        # Build complete query
        query = f"""
        GRAPH {self._database}
        MATCH (n)
        {where_clause}
        RETURN n, {similarity_expr} AS similarity_score
        ORDER BY similarity_score {order_direction}
        LIMIT {limit}
        """

        return query.strip()

    def build_hybrid_search_query(
        self,
        text_query: str,
        embedding: list[float],
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        similarity_function: str = 'COSINE_DISTANCE',
        limit: int = 10,
        node_labels: list[str] | None = None,
        group_ids: list[str] | None = None,
    ) -> str:
        """
        Build hybrid search query combining fulltext and vector search using Spanner's native capabilities.

        Args:
            text_query: Text query for fulltext search
            embedding: Query vector for similarity search
            text_weight: Weight for text similarity score
            vector_weight: Weight for vector similarity score
            similarity_function: Vector similarity function to use
            limit: Maximum number of results to return
            node_labels: Filter by node labels
            group_ids: Filter by group IDs

        Returns:
            GQL query string for hybrid search
        """
        # Convert embedding to Spanner ARRAY format
        embedding_str = '[' + ', '.join(map(str, embedding)) + ']'

        # Escape text query
        escaped_query = text_query.replace("'", "\\'").replace('"', '\\"')

        # Choose vector similarity function
        if similarity_function.upper() == 'DOT_PRODUCT':
            # Normalize dot product score (assuming vectors are normalized)
            vector_similarity = f'(DOT_PRODUCT(n.embedding, {embedding_str}) + 1) / 2'
        elif similarity_function.upper() == 'EUCLIDEAN_DISTANCE':
            # Convert distance to similarity (0-1 scale)
            vector_similarity = f'1 / (1 + EUCLIDEAN_DISTANCE(n.embedding, {embedding_str}))'
        else:  # COSINE_DISTANCE
            # Convert distance to similarity
            vector_similarity = f'1 - COSINE_DISTANCE(n.embedding, {embedding_str})'

        # Build WHERE clause conditions
        where_conditions = []

        # Add node label filter
        if node_labels:
            label_conditions = ' OR '.join(
                [f"'{label}' IN UNNEST(LABELS(n))" for label in node_labels]
            )
            where_conditions.append(f'({label_conditions})')

        # Add group ID filter
        if group_ids:
            group_conditions = ' OR '.join([f"n.group_id = '{gid}'" for gid in group_ids])
            where_conditions.append(f'({group_conditions})')

        # Combine WHERE conditions
        where_clause = ''
        if where_conditions:
            where_clause = 'WHERE ' + ' AND '.join(where_conditions)

        # Build complete hybrid query
        # Note: This assumes nodes have both 'content' text field and 'embedding' vector field
        query = f"""
        GRAPH {self._database}
        MATCH (n)
        {where_clause}
        RETURN n,
               SEARCH(TOKENIZE_FULLTEXT(n.content), '{escaped_query}') AS text_match,
               {vector_similarity} AS vector_similarity,
               ({text_weight} * CAST(SEARCH(TOKENIZE_FULLTEXT(n.content), '{escaped_query}') AS FLOAT64) + 
                {vector_weight} * {vector_similarity}) AS hybrid_score
        WHERE SEARCH(TOKENIZE_FULLTEXT(n.content), '{escaped_query}') = TRUE OR {vector_similarity} > 0.1
        ORDER BY hybrid_score DESC
        LIMIT {limit}
        """

        return query.strip()

    def _convert_cypher_to_gql(self, cypher_query: str, params: dict) -> str:
        """
        Convert Cypher queries to Google Cloud Spanner GQL format.
        This is a basic conversion - production use would need a more sophisticated parser.
        """
        gql_query = cypher_query

        # Add GRAPH clause if not present
        if not gql_query.strip().upper().startswith('GRAPH'):
            gql_query = f'GRAPH {self._database}\n{gql_query}'

        # Convert MERGE operations to MATCH + conditional CREATE
        # This is a simplified approach - real implementation needs proper upsert logic
        if 'MERGE' in gql_query.upper():
            gql_query = gql_query.replace('MERGE', 'MATCH')
            logger.warning('MERGE converted to MATCH - may need manual upsert logic')

        # Handle parameter substitution
        for param_name, param_value in params.items():
            placeholder = f'${param_name}'
            if placeholder in gql_query:
                if isinstance(param_value, str):
                    gql_query = gql_query.replace(placeholder, f"'{param_value}'")
                elif param_value is None:
                    gql_query = gql_query.replace(placeholder, 'NULL')
                else:
                    gql_query = gql_query.replace(placeholder, str(param_value))

        return gql_query

    def _is_write_operation(self, query: str) -> bool:
        """Check if the query performs write operations"""
        write_patterns = ['CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        return any(pattern in query_upper for pattern in write_patterns)

    def _format_result(self, result) -> Any:
        """Format Spanner result to match expected interface"""
        # Convert Spanner results to a format compatible with other graph drivers
        # This may need adjustment based on the actual result format from Spanner
        return result
