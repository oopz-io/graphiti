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

import asyncio
import json
import logging
import random
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from google.api_core import exceptions as google_exceptions
from google.cloud import spanner_v1
from google.cloud.spanner_v1 import types
from google.cloud.spanner_v1.types import spanner, transaction
from google.protobuf import struct_pb2
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

# Retry configuration for handling Spanner transaction conflicts (409 Aborted errors)
# These occur when concurrent transactions conflict on the same rows
SPANNER_RETRY_CONFIG = {
    'max_retries': 5,  # Maximum number of retry attempts
    'initial_delay_ms': 100,  # Initial delay before first retry (100ms)
    'max_delay_ms': 5000,  # Maximum delay between retries (5 seconds)
    'multiplier': 2.0,  # Exponential backoff multiplier
    'jitter': 0.2,  # Random jitter factor (±20%)
}

# Configuration for BatchWrite (blind writes) mode
# BatchWrite eliminates transaction conflicts by performing non-transactional writes
# Each mutation group is atomic, but mutations across groups can be applied in any order
SPANNER_BATCH_WRITE_CONFIG = {
    'enabled': True,  # Enable BatchWrite for bulk operations (eliminates 409 conflicts)
    'mutations_per_group': 1,  # Number of mutations per atomic group (1 = max parallelism)
    'timeout_seconds': 300,  # Timeout for batch write operations (5 minutes)
}

# Configuration for Session Pool
# Adjust these based on your concurrent workload
SPANNER_SESSION_POOL_CONFIG = {
    'min_size': 5,  # Pre-warmed sessions (created on initialization)
    'max_size': 50,  # Maximum concurrent sessions (increase for high concurrency)
    'acquire_timeout_seconds': 30,  # Max time to wait for a session
    'log_exhaustion_as_warning': False,  # Set to True to log pool exhaustion as WARNING
}

# Table name mapping from user tables to Common tables
# Used when use_common_tables=True to route data to Common* tables
USER_TO_COMMON_TABLE_MAPPING = {
    'EntityNode': 'CommonEntityNode',
    'EntityEdge': 'CommonEntityEdge',
    'EpisodicNode': 'CommonEpisodicNode',
    'EpisodicEdge': 'CommonEpisodicEdge',
    'CommunityNode': 'CommonCommunityNode',
    'ContradictedEdge': 'CommonContradictedEdge',
    # Also map the property graph name
    'GRAPHITI': 'COMMON_GRAPHITI',
}


def _rewrite_query_for_common_tables(query: str) -> str:
    """
    Rewrite a SQL query to use Common* tables instead of user tables.

    This performs a simple string replacement of table names. The replacements
    are done in order of longest match first to avoid partial replacements.

    Args:
        query: The original SQL query with user table names

    Returns:
        The query with table names replaced to Common* versions
    """
    result = query
    # Sort by length descending to replace longer names first
    # This prevents 'EntityNode' from being replaced before 'CommonEntityNode' check
    for user_table, common_table in sorted(
        USER_TO_COMMON_TABLE_MAPPING.items(), key=lambda x: len(x[0]), reverse=True
    ):
        # Only replace if not already a Common table
        # Use word boundaries to avoid partial matches
        import re
        # Match table name that is not preceded by 'Common'
        pattern = rf'(?<!Common){user_table}\b'
        result = re.sub(pattern, common_table, result)
    return result


def _parse_insert_or_update_query(
    query: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Parse an INSERT OR UPDATE query and extract table, columns, and values.

    This allows converting SQL upsert queries to BatchWrite mutations for
    conflict-free parallel execution.

    Args:
        query: SQL query string (INSERT OR UPDATE ...)
        params: Query parameters dictionary

    Returns:
        Dictionary with 'table', 'columns', 'values' if parseable, None otherwise

    Example:
        query = "INSERT OR UPDATE EntityEdge (uuid, name) VALUES (@uuid, @name)"
        params = {'uuid': 'abc', 'name': 'test'}
        -> {'table': 'EntityEdge', 'columns': ['uuid', 'name'], 'values': ['abc', 'test']}
    """
    query_upper = query.strip().upper()
    if not query_upper.startswith('INSERT OR UPDATE'):
        return None

    try:
        # Extract table name: INSERT OR UPDATE TableName (columns...)
        # Find the table name between "INSERT OR UPDATE" and "("
        start_idx = len('INSERT OR UPDATE')
        paren_idx = query.find('(', start_idx)
        if paren_idx == -1:
            return None

        table_name = query[start_idx:paren_idx].strip()

        # Extract columns: (col1, col2, col3)
        # Find closing paren for columns
        col_end_idx = query.find(')', paren_idx)
        if col_end_idx == -1:
            return None

        columns_str = query[paren_idx + 1 : col_end_idx]
        columns = [c.strip() for c in columns_str.split(',')]

        # Extract values: VALUES (@param1, @param2, ...)
        values_idx = query_upper.find('VALUES')
        if values_idx == -1:
            return None

        values_start = query.find('(', values_idx)
        values_end = query.find(')', values_start)
        if values_start == -1 or values_end == -1:
            return None

        values_str = query[values_start + 1 : values_end]
        value_placeholders = [v.strip() for v in values_str.split(',')]

        # Resolve parameter values
        values = []
        for placeholder in value_placeholders:
            # Handle @param_name format
            if placeholder.startswith('@'):
                param_name = placeholder[1:]
                if param_name in params:
                    values.append(params[param_name])
                else:
                    # Parameter not found
                    return None
            else:
                # Literal value (shouldn't happen but handle gracefully)
                return None

        if len(columns) != len(values):
            return None

        return {
            'table': table_name,
            'columns': columns,
            'values': values,
        }

    except Exception:
        return None


def _calculate_retry_delay(attempt: int, config: dict = SPANNER_RETRY_CONFIG) -> float:
    """
    Calculate delay before next retry with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration dictionary

    Returns:
        Delay in seconds
    """
    # Calculate base delay with exponential backoff
    delay_ms = config['initial_delay_ms'] * (config['multiplier'] ** attempt)

    # Cap at maximum delay
    delay_ms = min(delay_ms, config['max_delay_ms'])

    # Add jitter (±jitter_factor)
    jitter_range = delay_ms * config['jitter']
    delay_ms += random.uniform(-jitter_range, jitter_range)

    # Ensure non-negative
    delay_ms = max(delay_ms, 0)

    return delay_ms / 1000.0  # Convert to seconds


def _is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (transaction conflict/abort).

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable (409 Aborted), False otherwise
    """
    # Check for google.api_core.exceptions.Aborted (409)
    if isinstance(error, google_exceptions.Aborted):
        return True

    # Check for error message patterns indicating transaction conflicts
    error_str = str(error).lower()
    retryable_patterns = [
        'transaction was aborted',
        'was wounded by a higher priority transaction',
        'conflict on keys',
        'concurrent transaction',
        'aborted due to transient fault',
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


class SessionPool:
    """
    Session pool for Spanner to avoid creating/destroying sessions for every query.

    This pool maintains a collection of reusable Spanner sessions, significantly
    reducing the overhead of session management which can take 350-7,400ms per
    session creation and 500-3,100ms per session deletion.

    With session pooling, session acquisition takes ~1ms instead of creating new sessions.
    """

    def __init__(
        self,
        client: spanner_v1.SpannerAsyncClient,
        database_path: str,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        """
        Initialize the session pool.

        Args:
            client: The Spanner async client
            database_path: Full path to the database
            min_size: Minimum number of sessions to maintain (warmed up).
                      Defaults to SPANNER_SESSION_POOL_CONFIG['min_size']
            max_size: Maximum number of sessions in the pool.
                      Defaults to SPANNER_SESSION_POOL_CONFIG['max_size']
        """
        self.client = client
        self.database_path = database_path
        self.min_size = (
            min_size if min_size is not None else SPANNER_SESSION_POOL_CONFIG['min_size']
        )
        self.max_size = (
            max_size if max_size is not None else SPANNER_SESSION_POOL_CONFIG['max_size']
        )
        self._pool: list[str] = []  # List of available session names
        self._in_use: set[str] = set()  # Set of session names currently in use
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_size)  # Limit concurrent waiters
        self._initialized = False

    async def initialize(self):
        """Pre-create minimum number of sessions for the pool."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return

            logger.info(f'[SESSION POOL] Initializing with {self.min_size} sessions...')
            for i in range(self.min_size):
                try:
                    session_name = await self._create_session()
                    self._pool.append(session_name)
                    logger.debug(f'[SESSION POOL] Pre-created session {i + 1}/{self.min_size}')
                except Exception as e:
                    logger.warning(f'[SESSION POOL] Failed to pre-create session {i + 1}: {e}')

            self._initialized = True
            logger.info(f'[SESSION POOL] Initialized with {len(self._pool)} sessions')

    async def _create_session(self) -> str:
        """Create a new Spanner session and return its name."""
        request = spanner.CreateSessionRequest(database=self.database_path)
        session = await self.client.create_session(request)
        return session.name

    async def _delete_session(self, session_name: str):
        """Delete a Spanner session."""
        try:
            request = spanner.DeleteSessionRequest(name=session_name)
            await self.client.delete_session(request)
        except Exception as e:
            logger.warning(f'[SESSION POOL] Failed to delete session: {e}')

    async def acquire(self) -> str:
        """
        Get a session from the pool or create a new one if needed.

        Uses a semaphore to properly queue waiters when pool is exhausted,
        avoiding recursive retry loops and providing timeout support.

        Returns:
            Session name (string)

        Raises:
            TimeoutError: If session cannot be acquired within timeout
        """
        # Ensure pool is initialized
        if not self._initialized:
            logger.info(
                '[SESSION POOL] acquire() called but pool not initialized - initializing now...'
            )
            await self.initialize()

        timeout = SPANNER_SESSION_POOL_CONFIG['acquire_timeout_seconds']
        start_time = asyncio.get_event_loop().time()

        while True:
            async with self._lock:
                # Try to get a session from the pool
                if self._pool:
                    session_name = self._pool.pop()
                    self._in_use.add(session_name)
                    logger.debug(
                        f'[SESSION POOL] Acquired session from pool '
                        f'(available: {len(self._pool)}, in use: {len(self._in_use)})'
                    )
                    return session_name

                # If pool is empty but we haven't hit max size, create a new session
                total_sessions = len(self._pool) + len(self._in_use)
                if total_sessions < self.max_size:
                    logger.debug(
                        f'[SESSION POOL] Creating new session '
                        f'(total: {total_sessions}/{self.max_size})'
                    )
                    session_name = await self._create_session()
                    self._in_use.add(session_name)
                    return session_name

                # Pool exhausted - log at appropriate level
                if SPANNER_SESSION_POOL_CONFIG['log_exhaustion_as_warning']:
                    logger.warning(
                        f'[SESSION POOL] Pool exhausted ({self.max_size}/{self.max_size}), '
                        'waiting for session...'
                    )
                else:
                    logger.info(
                        f'[SESSION POOL] Pool exhausted ({self.max_size}/{self.max_size}), '
                        'waiting for session...'
                    )

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f'[SESSION POOL] Timeout after {timeout}s waiting for session. '
                    f'Pool size: {self.max_size}, all sessions in use. '
                    'Consider increasing SPANNER_SESSION_POOL_CONFIG["max_size"].'
                )

            # Wait a bit before retry (with exponential backoff capped at 500ms)
            wait_time = min(0.05 * (1 + elapsed), 0.5)
            await asyncio.sleep(wait_time)

    async def release(self, session_name: str):
        """
        Return a session to the pool for reuse.

        Args:
            session_name: The session name to return
        """
        async with self._lock:
            # Remove from in-use set
            self._in_use.discard(session_name)

            # If pool is not full, return session to pool for reuse
            if len(self._pool) < self.max_size:
                self._pool.append(session_name)
                logger.debug(
                    f'[SESSION POOL] Released session to pool (available: {len(self._pool)}, in use: {len(self._in_use)})'
                )
            else:
                # Pool is full, delete the session
                logger.debug('[SESSION POOL] Pool full, deleting session')
                await self._delete_session(session_name)

    async def close(self):
        """Close all sessions in the pool."""
        async with self._lock:
            logger.info(
                f'[SESSION POOL] Closing pool with {len(self._pool)} available sessions and {len(self._in_use)} in-use sessions'
            )

            # Delete all available sessions
            for session_name in self._pool:
                await self._delete_session(session_name)

            # Delete all in-use sessions (they should be returned first, but clean up anyway)
            for session_name in self._in_use:
                await self._delete_session(session_name)

            self._pool.clear()
            self._in_use.clear()
            logger.info('[SESSION POOL] Pool closed')


def _convert_datetimes_for_json(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat().replace('+00:00', 'Z')
    elif isinstance(obj, dict):
        return {key: _convert_datetimes_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_datetimes_for_json(item) for item in obj]
    else:
        return obj


def _extract_value_from_protobuf(value: Any) -> Any:
    """Extract Python value from protobuf Value object."""
    from google.protobuf import struct_pb2

    # Handle different protobuf value types
    if isinstance(value, str | int | float | bool):
        return value
    elif value is None or (
        hasattr(value, 'WhichOneof') and value.WhichOneof('kind') == 'null_value'
    ):
        return None
    elif isinstance(value, struct_pb2.ListValue):
        return [_extract_value_from_protobuf(v) for v in value]
    elif isinstance(value, struct_pb2.Struct):
        return {k: _extract_value_from_protobuf(v) for k, v in value.items()}
    elif hasattr(value, 'string_value'):
        return value.string_value
    elif hasattr(value, 'number_value'):
        return value.number_value
    elif hasattr(value, 'bool_value'):
        return value.bool_value
    elif hasattr(value, 'list_value'):
        return [_extract_value_from_protobuf(v) for v in value.list_value.values]
    elif hasattr(value, 'struct_value'):
        return {k: _extract_value_from_protobuf(v) for k, v in value.struct_value.fields.items()}
    else:
        return value


def _convert_value_for_mutation(value: Any) -> Any:
    """Convert Python value to Spanner mutation-compatible format.

    Args:
        value: Python value to convert

    Returns:
        Value in format suitable for Spanner mutations
    """
    if value is None:
        return None
    elif isinstance(value, datetime):
        # Convert datetime to RFC3339 string
        return value.isoformat().replace('+00:00', 'Z')
    elif isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, list):
        # Convert list elements recursively
        return [_convert_value_for_mutation(item) for item in value]
    elif isinstance(value, dict):
        # JSON already serialized as string in bulk_utils
        return value
    else:
        # Default to string representation
        return str(value)


def _format_spanner_params(
    kwargs: dict[str, Any],
) -> tuple[struct_pb2.Struct, dict[str, types.Type]]:
    """Format parameters for Spanner ExecuteSqlRequest.

    Returns:
        tuple: (params_struct, param_types_dict)
    """
    params = {}
    param_types_t = {}

    # Remove routing_ if present
    kwargs.pop('routing_', None)

    for key, value in kwargs.items():
        param_key = f'{key}'
        if value is None:
            # Handle None/NULL values - pass as None and let Spanner handle it
            params[param_key] = None
            # For None values, we need to infer the type from the parameter name
            # Common timestamp fields that can be None
            if param_key in ('expired_at', 'valid_at', 'invalid_at', 'created_at'):
                param_types_t[param_key] = types.Type(code=types.TypeCode.TIMESTAMP)
            else:
                param_types_t[param_key] = types.Type(code=types.TypeCode.STRING)
        elif isinstance(value, datetime):
            params[param_key] = value.isoformat().replace('+00:00', 'Z')
            param_types_t[param_key] = types.Type(code=types.TypeCode.TIMESTAMP)
        elif isinstance(value, bool):
            params[param_key] = value
            param_types_t[param_key] = types.Type(code=types.TypeCode.BOOL)
        elif isinstance(value, str):
            params[param_key] = value
            # Check if this is a JSON column (attributes is JSON type in Spanner)
            if param_key == 'attributes' and (value.startswith('{') or value.startswith('[')):
                param_types_t[param_key] = types.Type(code=types.TypeCode.JSON)
            else:
                param_types_t[param_key] = types.Type(code=types.TypeCode.STRING)
        elif isinstance(value, int):
            params[param_key] = int(value)
            param_types_t[param_key] = types.Type(code=types.TypeCode.INT64)
        elif isinstance(value, float):
            params[param_key] = value
            param_types_t[param_key] = types.Type(code=types.TypeCode.FLOAT64)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Lists of complex objects (dicts) need to be JSON serialized
                # Convert datetimes to strings first
                converted_value = _convert_datetimes_for_json(value)
                params[param_key] = json.dumps(converted_value)
                param_types_t[param_key] = types.Type(code=types.TypeCode.STRING)
            else:
                params[param_key] = value
                if value:
                    if isinstance(value[0], str):
                        param_types_t[param_key] = types.Type(
                            code=types.TypeCode.ARRAY,
                            array_element_type=types.Type(code=types.TypeCode.STRING),
                        )
                    elif isinstance(value[0], bool):
                        param_types_t[param_key] = types.Type(
                            code=types.TypeCode.ARRAY,
                            array_element_type=types.Type(code=types.TypeCode.BOOL),
                        )
                    elif isinstance(value[0], int):
                        param_types_t[param_key] = types.Type(
                            code=types.TypeCode.ARRAY,
                            array_element_type=types.Type(code=types.TypeCode.INT64),
                        )
                    elif isinstance(value[0], float):
                        param_types_t[param_key] = types.Type(
                            code=types.TypeCode.ARRAY,
                            array_element_type=types.Type(code=types.TypeCode.FLOAT64),
                        )
                    else:
                        param_types_t[param_key] = types.Type(
                            code=types.TypeCode.ARRAY,
                            array_element_type=types.Type(code=types.TypeCode.STRING),
                        )
        else:
            # Default to string representation for unhandled types
            params[param_key] = str(value)
            param_types_t[param_key] = types.Type(code=types.TypeCode.STRING)

    # Convert params dict to protobuf Struct
    # NOTE: INT64 values must be passed as strings to avoid precision loss
    params_struct = struct_pb2.Struct()
    for key, value in params.items():
        # Check the parameter type to determine correct conversion
        param_type = param_types_t.get(key)
        if param_type and param_type.code == types.TypeCode.INT64:
            # INT64 must be passed as string
            params_struct[key] = str(value)
        elif param_type and param_type.code == types.TypeCode.TIMESTAMP:
            # TIMESTAMP is already a string
            params_struct[key] = value
        elif param_type and param_type.code == types.TypeCode.ARRAY:
            # Arrays can be passed as lists directly in protobuf Struct
            if isinstance(value, list):
                element_type = param_type.array_element_type.code
                # For INT64 arrays, convert elements to strings
                if element_type == types.TypeCode.INT64:
                    params_struct[key] = [str(v) for v in value]
                else:
                    # All other arrays (FLOAT64, STRING, BOOL, etc.) can be passed as-is
                    params_struct[key] = value
            else:
                params_struct[key] = value
        elif isinstance(value, str | bool):
            params_struct[key] = value
        elif isinstance(value, int):
            # Default int handling (convert to string for safety)
            params_struct[key] = str(value)
        elif isinstance(value, float):
            params_struct[key] = value
        elif isinstance(value, list):
            # Default list handling - assume it's already in the right format
            params_struct[key] = value

    return params_struct, param_types_t


class SpannerDriverSession(GraphDriverSession):
    """Wrapper around Spanner's session to implement GraphDriverSession interface."""

    provider = GraphProvider.SPANNER

    def __init__(self, client: spanner_v1.SpannerAsyncClient, database_path: str):
        self._client = client
        self._database_path = database_path
        self._session_name = None
        self._current_transaction = None
        self._seqno = 0
        self._parent_driver: Any = None
        self._pending_mutations: list = []

    async def _ensure_session(self):
        """Ensure session is created (lazy initialization)."""
        if self._session_name is None:
            if self._parent_driver and hasattr(self._parent_driver, 'session_pool'):
                # Get session from pool
                self._session_name = await self._parent_driver.session_pool.acquire()
                logger.debug(f'[SESSION] Acquired session from pool: {self._session_name}')
            else:
                # Fallback: create new session directly (for standalone use)
                request = spanner.CreateSessionRequest(database=self._database_path)
                session = await self._client.create_session(request)
                self._session_name = session.name
                logger.debug(f'[SESSION] Created new session directly: {self._session_name}')

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def run(self, query: str, **kwargs: Any) -> Any:
        """Execute a query in this session."""
        await self._ensure_session()

        # Format parameters
        params_struct, param_types_t = _format_spanner_params(kwargs)

        # For read queries, use single use transaction
        if query.strip().upper().startswith('SELECT'):
            request = spanner.ExecuteSqlRequest(
                session=self._session_name,
                sql=query,
                params=params_struct,
                param_types=param_types_t,
            )
            result = await self._client.execute_streaming_sql(request)
            return result

        # For write queries, use transaction
        if not self._current_transaction:
            options = transaction.TransactionOptions(
                read_write=transaction.TransactionOptions.ReadWrite()
            )
            begin_request = spanner.BeginTransactionRequest(
                session=self._session_name, options=options
            )
            transaction_obj = await self._client.begin_transaction(begin_request)
            self._current_transaction = transaction_obj.id
            self._seqno = 0

        try:
            request = spanner.ExecuteSqlRequest(
                session=self._session_name,
                sql=query,
                params=params_struct,
                param_types=param_types_t,
                transaction={'id': self._current_transaction},
                seqno=self._seqno,
            )
            self._seqno += 1
            result = await self._client.execute_streaming_sql(request)
            return result
        except Exception as e:
            if self._current_transaction:
                request = spanner.RollbackRequest(
                    session=self._session_name, transaction_id=self._current_transaction
                )
                await self._client.rollback(request)
                self._current_transaction = None
                self._seqno = 0
            raise e

    async def run_mutations(
        self,
        mutations_data: list[dict[str, Any]],
    ) -> int:
        """Execute mutations using Spanner's Mutation API for bulk inserts.

        This is significantly faster than Batch DML for bulk writes as it bypasses
        SQL parsing and query planning. Expected performance: 2-5x faster than Batch DML.

        Args:
            mutations_data: List of mutation dictionaries, each containing:
                - 'table': Table name (str)
                - 'columns': List of column names (list[str])
                - 'values': List of values matching columns (list[Any])

        Returns:
            Number of mutations committed

        Example:
            mutations = [
                {
                    'table': 'EntityNode',
                    'columns': ['uuid', 'name', 'created_at'],
                    'values': ['uuid-1', 'Alice', datetime.now()]
                },
                {
                    'table': 'EntityEdge',
                    'columns': ['uuid', 'fact'],
                    'values': ['edge-1', 'Alice knows Bob']
                }
            ]
            await session.run_mutations(mutations)
        """
        from time import time

        logger.info(f'[MUTATIONS] run_mutations called with {len(mutations_data)} mutations')

        if not mutations_data:
            return 0

        await self._ensure_session()

        # Start a transaction if not already started
        if not self._current_transaction:
            options = transaction.TransactionOptions(
                read_write=transaction.TransactionOptions.ReadWrite()
            )
            begin_request = spanner.BeginTransactionRequest(
                session=self._session_name, options=options
            )
            transaction_obj = await self._client.begin_transaction(begin_request)
            self._current_transaction = transaction_obj.id
            self._seqno = 0

        try:
            mutation_start = time()

            # Build mutations
            mutations = []
            for mut_data in mutations_data:
                table = mut_data['table']
                columns = mut_data['columns']
                values = mut_data['values']

                # Convert values to mutation-compatible format
                converted_values = [_convert_value_for_mutation(v) for v in values]

                # Create mutation with INSERT_OR_UPDATE for upsert behavior
                # This matches the "INSERT OR UPDATE" SQL behavior we're replacing
                mutation = types.Mutation(
                    insert_or_update=types.Mutation.Write(
                        table=table, columns=columns, values=[converted_values]
                    )
                )
                mutations.append(mutation)

            build_time = (time() - mutation_start) * 1000
            logger.info(f'[MUTATIONS] Built {len(mutations)} mutations in {build_time:.2f}ms')

            # Store mutations for commit
            # Note: We don't commit here - execute_write() will handle the commit
            # to maintain consistency with the transaction management pattern
            if not hasattr(self, '_pending_mutations'):
                self._pending_mutations = []
            self._pending_mutations.extend(mutations)

            total_time = (time() - mutation_start) * 1000
            logger.info(
                f'[MUTATIONS] Prepared {len(mutations)} mutations in {total_time:.2f}ms ({total_time / len(mutations):.2f}ms per mutation)'
            )

            return len(mutations)

        except Exception as e:
            logger.error(f'[MUTATIONS] Error: {e}')
            if self._current_transaction:
                request = spanner.RollbackRequest(
                    session=self._session_name, transaction_id=self._current_transaction
                )
                await self._client.rollback(request)
                self._current_transaction = None
                self._seqno = 0
            raise e

    async def run_mutations_batch_write(
        self,
        mutations_data: list[dict[str, Any]],
        mutations_per_group: int = 1,
    ) -> tuple[int, int]:
        """Execute mutations using Spanner's BatchWrite API for conflict-free blind writes.

        BatchWrite performs non-transactional writes that eliminate lock conflicts:
        - No read phase = no shared locks
        - No lock upgrade = no deadlock detection
        - Mutations are applied directly = no wound-wait algorithm triggered

        This is ideal for parallel batch processing where:
        - Multiple workers may write to the same rows
        - INSERT_OR_UPDATE (upsert) semantics are acceptable
        - Last-write-wins behavior is acceptable

        Trade-offs vs transactions:
        - PRO: Zero lock contention, linear scalability, no 409 errors
        - CON: Last-write-wins (no merge), eventual consistency between groups

        Args:
            mutations_data: List of mutation dictionaries, each containing:
                - 'table': Table name (str)
                - 'columns': List of column names (list[str])
                - 'values': List of values matching columns (list[Any])
            mutations_per_group: Number of mutations per atomic group.
                - 1 = maximum parallelism (each row independent)
                - N = batch N mutations atomically

        Returns:
            Tuple of (successful_mutations, failed_mutations)

        Example:
            mutations = [
                {
                    'table': 'EntityNode',
                    'columns': ['uuid', 'name', 'created_at'],
                    'values': ['uuid-1', 'Alice', datetime.now()]
                },
                {
                    'table': 'EntityEdge',
                    'columns': ['uuid', 'fact'],
                    'values': ['edge-1', 'Alice knows Bob']
                }
            ]
            success, failed = await session.run_mutations_batch_write(mutations)
        """
        from time import time

        logger.info(
            f'[BATCH_WRITE] Starting BatchWrite with {len(mutations_data)} mutations '
            f'({mutations_per_group} per group)'
        )

        if not mutations_data:
            return 0, 0

        await self._ensure_session()

        try:
            mutation_start = time()

            # Build Spanner Mutation objects
            mutations = []
            for mut_data in mutations_data:
                table = mut_data['table']
                columns = mut_data['columns']
                values = mut_data['values']

                # Convert values to mutation-compatible format
                converted_values = [_convert_value_for_mutation(v) for v in values]

                # Create mutation with INSERT_OR_UPDATE for upsert behavior
                # This is idempotent - safe for replays as per Google's recommendation
                mutation = types.Mutation(
                    insert_or_update=types.Mutation.Write(
                        table=table, columns=columns, values=[converted_values]
                    )
                )
                mutations.append(mutation)

            build_time = (time() - mutation_start) * 1000
            logger.debug(f'[BATCH_WRITE] Built {len(mutations)} mutations in {build_time:.2f}ms')

            # Group mutations into MutationGroups
            # Each MutationGroup is atomic, but groups can be applied in any order
            mutation_groups = []
            for i in range(0, len(mutations), mutations_per_group):
                group_mutations = mutations[i : i + mutations_per_group]
                mutation_group = spanner.BatchWriteRequest.MutationGroup(mutations=group_mutations)
                mutation_groups.append(mutation_group)

            logger.info(f'[BATCH_WRITE] Created {len(mutation_groups)} mutation groups')

            # Execute BatchWrite - this is a streaming RPC
            batch_start = time()
            request = spanner.BatchWriteRequest(
                session=self._session_name,
                mutation_groups=mutation_groups,
            )

            successful_count = 0
            failed_count = 0

            # Process streaming responses
            # Each response contains results for one or more mutation groups
            stream = await self._client.batch_write(request=request)
            async for response in stream:
                # response.indexes contains indices of mutation groups in this batch
                # response.status contains the result (OK or error)
                if response.status.code == 0:  # google.rpc.Code.OK
                    # Count mutations in successful groups
                    for idx in response.indexes:
                        if idx < len(mutation_groups):
                            successful_count += len(mutation_groups[idx].mutations)
                    if response.commit_timestamp:
                        logger.debug(
                            f'[BATCH_WRITE] Groups {list(response.indexes)} committed at '
                            f'{response.commit_timestamp}'
                        )
                else:
                    # Count mutations in failed groups
                    for idx in response.indexes:
                        if idx < len(mutation_groups):
                            failed_count += len(mutation_groups[idx].mutations)
                    logger.warning(
                        f'[BATCH_WRITE] Groups {list(response.indexes)} failed: '
                        f'{response.status.message}'
                    )

            batch_time = (time() - batch_start) * 1000
            total_time = (time() - mutation_start) * 1000

            logger.info(
                f'[BATCH_WRITE] Completed: {successful_count} successful, {failed_count} failed '
                f'in {total_time:.2f}ms (batch: {batch_time:.2f}ms)'
            )

            return successful_count, failed_count

        except Exception as e:
            logger.error(f'[BATCH_WRITE] Error: {e}')
            raise

    async def close(self):
        """Close the session."""
        if self._session_name is None:
            return

        # Only try to commit/rollback if there's an active transaction
        # After execute_write() completes, _current_transaction should be None
        if self._current_transaction:
            logger.warning(
                f'[SESSION] close() called with active transaction {self._current_transaction[:20]}... '
                'This should not happen if execute_write() completed successfully. '
                'Attempting to commit remaining transaction.'
            )
            try:
                request = spanner.CommitRequest(
                    session=self._session_name, transaction_id=self._current_transaction
                )
                await self._client.commit(request)
                logger.info('[SESSION] Successfully committed remaining transaction in close()')
            except Exception as e:
                # If commit fails, transaction may already be committed/invalid
                # Log but don't re-raise, just clean up
                logger.warning(
                    f'[SESSION] Cannot commit/rollback in close(): {e}. Transaction may already be closed.'
                )
            finally:
                self._current_transaction = None
                self._seqno = 0

        # Clear any pending mutations
        if hasattr(self, '_pending_mutations'):
            self._pending_mutations = []

        # Return session to pool or delete it
        if self._session_name:
            if self._parent_driver and hasattr(self._parent_driver, 'session_pool'):
                # Return to pool if we have a parent driver with session pool
                await self._parent_driver.session_pool.release(self._session_name)
                logger.debug(f'[SESSION] Returned session to pool: {self._session_name}')
            else:
                # Delete session if it was created directly (no pool)
                try:
                    request = spanner.DeleteSessionRequest(name=self._session_name)
                    await self._client.delete_session(request)
                    logger.debug(f'[SESSION] Deleted direct session: {self._session_name}')
                except Exception as e:
                    logger.warning(f'[SESSION] Failed to delete session: {e}')

        # Clear session reference
        self._session_name = None

    async def execute_write(self, func, *args, **kwargs):
        """Execute a write operation in a transaction with automatic retry on conflicts.

        This method implements exponential backoff retry logic to handle Spanner
        transaction conflicts (409 Aborted errors) that occur when concurrent
        transactions try to modify the same rows.

        Args:
            func: The async function to execute within the transaction
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            The result of func

        Raises:
            The last exception if all retries are exhausted
        """
        max_retries = SPANNER_RETRY_CONFIG['max_retries']
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                return await self._execute_write_once(func, *args, **kwargs)
            except Exception as e:
                last_error = e

                # Check if error is retryable
                if not _is_retryable_error(e):
                    logger.error(f'[TRANSACTION] Non-retryable error: {e}')
                    raise

                # Check if we have retries left
                if attempt >= max_retries:
                    logger.error(
                        f'[TRANSACTION] Transaction aborted after {max_retries} retries. '
                        f'Last error: {e}'
                    )
                    raise

                # Calculate delay and wait
                delay = _calculate_retry_delay(attempt)
                logger.warning(
                    f'[TRANSACTION] Transaction conflict detected (attempt {attempt + 1}/{max_retries + 1}). '
                    f'Retrying in {delay:.2f}s. Error: {str(e)[:200]}'
                )
                await asyncio.sleep(delay)

                # Reset session state for retry
                # Clear any pending mutations from failed attempt
                if hasattr(self, '_pending_mutations'):
                    self._pending_mutations = []
                # Clear transaction state (will be re-created on next attempt)
                self._current_transaction = None
                self._seqno = 0

        # Should not reach here, but just in case
        if last_error is not None:
            raise last_error
        raise RuntimeError('Unexpected state: no error captured but retries exhausted')

    async def _execute_write_once(self, func, *args, **kwargs):
        """Execute a single write attempt (internal method used by execute_write)."""
        await self._ensure_session()

        if not self._current_transaction:
            options = transaction.TransactionOptions(
                read_write=transaction.TransactionOptions.ReadWrite()
            )
            begin_request = spanner.BeginTransactionRequest(
                session=self._session_name, options=options
            )
            transaction_obj = await self._client.begin_transaction(begin_request)
            self._current_transaction = transaction_obj.id
            self._seqno = 0

        try:
            result = await func(self, *args, **kwargs)

            # Commit with mutations if any were accumulated, otherwise regular commit
            if hasattr(self, '_pending_mutations') and self._pending_mutations:
                logger.info(
                    f'[MUTATIONS] Committing {len(self._pending_mutations)} pending mutations'
                )
                request = spanner.CommitRequest(
                    session=self._session_name,
                    transaction_id=self._current_transaction,
                    mutations=self._pending_mutations,
                )
                self._pending_mutations = []
            else:
                request = spanner.CommitRequest(
                    session=self._session_name, transaction_id=self._current_transaction
                )

            # Clear transaction ID BEFORE committing to avoid rollback attempts if commit succeeds
            transaction_id = self._current_transaction
            self._current_transaction = None
            self._seqno = 0

            try:
                await self._client.commit(request)
            except Exception as commit_error:
                # Commit failed - restore transaction ID for rollback attempt
                self._current_transaction = transaction_id
                raise commit_error

            return result
        except Exception as e:
            # Only attempt rollback if there's still an active transaction
            # (i.e., commit never succeeded or never happened)
            if self._current_transaction:
                try:
                    request = spanner.RollbackRequest(
                        session=self._session_name, transaction_id=self._current_transaction
                    )
                    await self._client.rollback(request)
                except Exception as rollback_error:
                    # Rollback can fail if transaction already committed/closed
                    logger.warning(
                        f'[TRANSACTION] Rollback failed: {rollback_error}. Transaction may already be closed.'
                    )
                finally:
                    self._current_transaction = None
                    self._seqno = 0
            # Clear pending mutations on error
            if hasattr(self, '_pending_mutations'):
                self._pending_mutations = []
            raise e


class SpannerDriver(GraphDriver):
    """Google Cloud Spanner implementation of GraphDriver.

    This driver provides flexible schema initialization control. The schema includes:

    - EntityNode table for storing entity nodes
    - EntityEdge table for storing relationships between entities
    - EpisodicNode table for storing episodic memory
    - CommunityNode table for storing community/cluster nodes
    - EpisodicEdge table for linking episodes to entities
    - ContradictedEdge table for tracking edge invalidations (audit trail)
    - Full-text search indexes for content search
    - Property graph definition for graph query support

    Common Tables (Optional, enabled by default):
    ---------------------------------------------
    When enable_common_tables=True, additional tables are created for storing
    shared/common knowledge separate from user-specific data:

    - CommonEntityNode, CommonEntityEdge, CommonEpisodicNode, etc.
    - COMMON_GRAPHITI property graph for graph queries on common data
    
    This separation provides better isolation between user and common data,
    independent scaling, and cleaner data management.

    Schema Initialization:
    ----------------------
    The driver supports two modes for schema initialization:

    1. Automatic (ensure_schema=True): Schema is created during driver creation
       via the create() factory method. This is convenient for development and testing.

    2. Manual (ensure_schema=False, default): Schema initialization is skipped.
       You must create the schema manually or call initialize_schema() explicitly.
       This is recommended for production where you want explicit control.

    The schema initialization is idempotent - it will only create objects that
    don't already exist.

    Usage:
    ------
    For optimal performance with pre-warmed session pool, use the async
    factory method `create()` instead of direct instantiation:

        # Development/testing with automatic schema setup (includes Common tables)
        driver = await SpannerDriver.create(
            project_id, instance_id, database_id,
            ensure_schema=True,
            enable_common_tables=True  # default
        )

        # Production without Common tables
        driver = await SpannerDriver.create(
            project_id, instance_id, database_id,
            ensure_schema=True,
            enable_common_tables=False
        )

        # Production with manual schema control
        driver = await SpannerDriver.create(
            project_id, instance_id, database_id,
            ensure_schema=False  # default
        )
        # Optionally call initialize_schema() when you're ready
        await driver.initialize_schema()

    This will pre-warm the session pool before returning the driver instance.
    """

    provider = GraphProvider.SPANNER
    aoss_client: None = None

    def __init__(
        self,
        project_id: str,
        instance_id: str,
        database_id: str,
        credentials: Any = None,
        session_pool_size: int | None = None,
        ensure_schema: bool = False,
        enable_common_tables: bool = True,
        use_common_tables: bool = False,
    ):
        """Initialize the Spanner driver.

        Note: This constructor does not pre-warm the session pool (since __init__
        cannot be async). For better performance, use the async factory method:

            driver = await SpannerDriver.create(project_id, instance_id, database_id)

        Args:
            project_id: The GCP project ID
            instance_id: The Spanner instance ID
            database_id: The Spanner database ID
            credentials: Optional credentials object
            session_pool_size: Maximum number of sessions in the pool.
                             Defaults to SPANNER_SESSION_POOL_CONFIG['max_size'] (50).
            ensure_schema: If True, schema will be initialized when using the create() factory method.
                         If False (default), schema initialization is skipped entirely.
                         This parameter only takes effect when using create(), not the constructor.
                         Set to True if you want automatic schema setup.
            enable_common_tables: If True (default), create Common* tables for shared/common
                                knowledge storage (CommonEntityNode, CommonEntityEdge, etc.)
                                and the COMMON_GRAPHITI property graph.
                                If False, only user-specific tables are created.
            use_common_tables: If True, route all data operations (INSERT/UPDATE/SELECT) to
                             Common* tables (CommonEntityNode, CommonEntityEdge, etc.) instead
                             of the standard user tables. This is used for "common" mode where
                             shared knowledge is stored separately from user-specific data.
                             If False (default), use standard user tables.
        """
        super().__init__()

        # Create Spanner client
        self.client = spanner_v1.SpannerAsyncClient(credentials=credentials)

        # Store database path
        self.database_path = self.client.database_path(project_id, instance_id, database_id)

        self._database = database_id

        # Determine pool sizes from parameter or config
        max_pool_size = (
            session_pool_size
            if session_pool_size is not None
            else SPANNER_SESSION_POOL_CONFIG['max_size']
        )
        min_pool_size = min(SPANNER_SESSION_POOL_CONFIG['min_size'], max_pool_size)

        # Initialize session pool
        # Pre-warm with sessions to handle concurrent operations during add_episode
        # (typical first episode needs 6-8 concurrent sessions)
        self.session_pool = SessionPool(
            client=self.client,
            database_path=self.database_path,
            min_size=min_pool_size,
            max_size=max_pool_size,
        )

        # Store the schema initialization flag
        # If ensure_schema is True, schema will be initialized on first operation
        # If False, schema initialization is completely skipped
        self._ensure_schema_flag = ensure_schema
        self._schema_initialized = False

        # Store the common tables flag
        # If True, Common* tables and COMMON_GRAPHITI property graph will be created
        # If use_common_tables is True, we need Common* schema to exist
        if use_common_tables and not enable_common_tables:
            logger.info('[SPANNER DRIVER] use_common_tables=True implies enable_common_tables=True')
            enable_common_tables = True
        self._enable_common_tables = enable_common_tables

        # Store the use_common_tables flag
        # If True, route all data operations to Common* tables instead of user tables
        self._use_common_tables = use_common_tables

        # Flag to track if session pool has been pre-warmed
        self._pool_initialized = False

    @classmethod
    async def create(
        cls,
        project_id: str,
        instance_id: str,
        database_id: str,
        credentials: Any = None,
        session_pool_size: int | None = None,
        ensure_schema: bool = False,
        enable_common_tables: bool = True,
        use_common_tables: bool = False,
    ) -> 'SpannerDriver':
        """Async factory method to create a SpannerDriver with pre-warmed session pool.

        This is the recommended way to create a SpannerDriver instance as it will
        pre-initialize the session pool, eliminating the ~11 second cold start on
        first database operation.

        Example:
            driver = await SpannerDriver.create(
                project_id='my-project',
                instance_id='my-instance',
                database_id='my-database',
                ensure_schema=True  # Automatically set up database schema
            )

            # For "common" mode - store data in Common* tables
            driver = await SpannerDriver.create(
                project_id='my-project',
                instance_id='my-instance',
                database_id='my-database',
                use_common_tables=True  # Route data to Common* tables
            )

        Args:
            project_id: The GCP project ID
            instance_id: The Spanner instance ID
            database_id: The Spanner database ID
            credentials: Optional credentials object
            session_pool_size: Maximum number of sessions in the pool.
                             Defaults to SPANNER_SESSION_POOL_CONFIG['max_size'] (50).
            ensure_schema: If True, initialize the database schema during driver creation.
                         If False (default), schema initialization is skipped entirely.
            enable_common_tables: If True (default), Common* tables and COMMON_GRAPHITI
                                property graph will be created when schema is initialized.
                                If False, only the standard user tables are created.
            use_common_tables: If True, route all data operations to Common* tables
                             (CommonEntityNode, CommonEntityEdge, etc.) instead of
                             standard user tables. Use this for "common" mode storage.
                             If False (default), use standard user tables.

        Returns:
            SpannerDriver instance with pre-warmed session pool
        """
        # If use_common_tables is True, we need the Common* schema to exist
        # So automatically enable common tables schema creation
        if use_common_tables and not enable_common_tables:
            logger.info('[SPANNER DRIVER] use_common_tables=True implies enable_common_tables=True')
            enable_common_tables = True
        
        driver = cls(
            project_id, instance_id, database_id, credentials, session_pool_size, ensure_schema,
            enable_common_tables, use_common_tables
        )
        await driver._ensure_pool_initialized()

        # Initialize schema if requested
        if ensure_schema:
            await driver.initialize_schema()
            driver._schema_initialized = True

        return driver

    async def _ensure_pool_initialized(self) -> None:
        """Ensure session pool is pre-warmed before first use."""
        if not self._pool_initialized:
            logger.info('[SPANNER DRIVER] Pool not initialized, initializing now...')
            await self.session_pool.initialize()
            self._pool_initialized = True
            logger.info('[SPANNER DRIVER] Pool initialization complete')
        else:
            logger.info('[SPANNER DRIVER] Pool already initialized, skipping')

    async def initialize_schema(self) -> None:
        """Initialize the database schema if it doesn't exist.

        This method creates all necessary tables, sequences, search indexes, and
        property graph definitions required by Graphiti. It is idempotent - running
        it multiple times is safe as it will skip creation of objects that already exist.

        Note: This method is automatically called during driver creation if
        ensure_schema=True was passed to create(). You can also call it manually
        to set up the schema at a specific time.
        """
        # Skip if schema already initialized
        if self._schema_initialized:
            logger.info('Schema already initialized, skipping')
            return

        try:
            # Check if schema already exists by checking for one of the main tables
            # We check for EntityNode table as the primary indicator
            exists_query = """
            SELECT COUNT(*) as table_count 
            FROM information_schema.tables 
            WHERE table_name = 'EntityNode'
            """

            # Create a session directly for the schema check
            request = spanner.CreateSessionRequest(database=self.database_path)
            session = await self.client.create_session(request)

            try:
                # Format parameters for the existence check
                params_struct, param_types_t = _format_spanner_params({})

                # Execute the existence check query
                request = spanner.ExecuteSqlRequest(
                    session=session.name,
                    sql=exists_query,
                    params=params_struct,
                    param_types=param_types_t,
                )

                rows = []
                field_names = None

                async for partial_result in await self.client.execute_streaming_sql(request):
                    # Get field names from metadata
                    if field_names is None and partial_result.metadata:
                        field_names = [
                            field.name for field in partial_result.metadata.row_type.fields
                        ]

                    # Process the result values
                    if field_names:
                        num_fields = len(field_names)
                        all_values = [
                            _extract_value_from_protobuf(val) for val in partial_result.values
                        ]

                        # Group values into rows
                        for i in range(0, len(all_values), num_fields):
                            row_values = all_values[i : i + num_fields]
                            if len(row_values) == num_fields:
                                row_dict = dict(zip(field_names, row_values, strict=False))
                                rows.append(row_dict)

                # Convert table_count to int since Spanner returns INT64 as string
                table_count = rows[0].get('table_count', 0) if rows and len(rows) > 0 else 0
                if isinstance(table_count, str):
                    table_count = int(table_count)
                schema_exists = table_count > 0

            finally:
                # Clean up the session
                cleanup_request = spanner.DeleteSessionRequest(name=session.name)
                await self.client.delete_session(cleanup_request)

            # If tables already exist, skip their creation and only create property graph
            if not schema_exists:
                logger.info('Creating database schema...')
            else:
                logger.info('Creating missing schema components...')

            # Define the schema DDL statements
            schema_statements = [
                # Create sequences
                """CREATE SEQUENCE EntityNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE EntityEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE EpisodicNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommunityNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE EpisodicEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE ContradictedEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                # Create tables
                """CREATE TABLE EntityNode (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE EntityNodeSequence)),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  labels ARRAY<STRING(256)>,
                  created_at TIMESTAMP NOT NULL,
                  name_embedding ARRAY<FLOAT64>(vector_length=>768),
                  summary STRING(MAX),
                  attributes JSON,
                  name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN,
                  summary_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(summary)) HIDDEN,
                  labels_tokens TOKENLIST AS (TOKEN(labels)) HIDDEN,
                  name_ngrams_tokens TOKENLIST AS (TOKENIZE_NGRAMS(name, ngram_size_min=>3, ngram_size_max=>4)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                """CREATE TABLE EntityEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE EntityEdgeSequence)),
                  source_node_uuid STRING(256),
                  target_node_uuid STRING(256),
                  labels ARRAY<STRING(256)>,
                  fact STRING(MAX),
                  fact_embedding ARRAY<FLOAT64>(vector_length=>768),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  episodes ARRAY<STRING(256)>,
                  created_at TIMESTAMP NOT NULL,
                  expired_at TIMESTAMP,
                  valid_at TIMESTAMP,
                  invalid_at TIMESTAMP,
                  attributes JSON,
                  name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN,
                  fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(fact)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                """CREATE TABLE EpisodicNode (
                    id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE EpisodicNodeSequence)),
                    source STRING(256),
                    source_description STRING(MAX),
                    content STRING(MAX),
                    entity_edges ARRAY<STRING(MAX)>,
                    uuid STRING(256),
                    name STRING(MAX),
                    group_id STRING(256),
                    created_at TIMESTAMP NOT NULL,
                    valid_at TIMESTAMP NOT NULL,
                    content_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(content)) HIDDEN,
                    source_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(source)) HIDDEN,
                    source_description_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(source_description)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                """CREATE TABLE CommunityNode (
                    id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommunityNodeSequence)),
                    uuid STRING(256),
                    name STRING(MAX),
                    name_embedding ARRAY<FLOAT64>(vector_length=>768),
                    group_id STRING(256),
                    created_at TIMESTAMP NOT NULL,
                    summary STRING(MAX),
                    labels ARRAY<STRING(256)>,
                    name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                """CREATE TABLE EpisodicEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE EpisodicEdgeSequence)),
                  source_node_uuid STRING(256),
                  target_node_uuid STRING(256),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  created_at TIMESTAMP NOT NULL
                ) PRIMARY KEY(uuid)""",
                """CREATE TABLE ContradictedEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE ContradictedEdgeSequence)),
                  uuid STRING(256),
                  invalidated_edge_uuid STRING(256),
                  invalidating_edge_uuid STRING(256),
                  invalidated_fact STRING(MAX),
                  invalidating_fact STRING(MAX),
                  invalidated_at TIMESTAMP NOT NULL,
                  group_id STRING(256),
                  invalidated_edge_data JSON,
                  invalidating_edge_data JSON,
                  created_at TIMESTAMP NOT NULL,
                  invalidated_fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(invalidated_fact)) HIDDEN,
                  invalidating_fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(invalidating_fact)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # ============================================================
                # SEARCH INDEXES (Full-Text Search with Partition Isolation)
                # ============================================================
                # Search indexes enable efficient full-text search using SEARCH() function
                # and token-based filtering using ARRAY_INCLUDES() with TOKEN().
                #
                # PARTITION BY group_id:
                # - Isolates search operations to a single tenant/user partition
                # - Queries can only search within a single partition at a time
                # - Dramatically improves performance for multi-tenant workloads
                # - Scales horizontally: 1M+ partitions with independent search spaces
                #
                # STORING clause:
                # - Duplicates columns into the search index for covered queries
                # - Eliminates table lookups after index scan (faster reads)
                # - Trade-off: Increased storage for reduced latency
                #
                # EntityNode search index includes:
                # - name_tokens: Full-text search on entity names
                # - summary_tokens: Full-text search on entity summaries
                # - labels_tokens: TOKEN(labels) for ARRAY_INCLUDES filtering
                # - name_ngrams_tokens: TOKENIZE_NGRAMS for REGEXP_CONTAINS pattern matching
                # Note: uuid (primary key), group_id (partition key), and ARRAY fields
                #       are excluded from STORING as they are implicitly available or not allowed
                """CREATE SEARCH INDEX EntityNode_search_index
                   ON EntityNode(name_tokens, summary_tokens, labels_tokens, name_ngrams_tokens)
                   STORING (name, summary, attributes, created_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX EntityEdge_search_index
                   ON EntityEdge(name_tokens, fact_tokens)
                   STORING (name, fact, source_node_uuid, target_node_uuid, created_at, expired_at, valid_at, invalid_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX EpisodicNode_search_index
                   ON EpisodicNode(content_tokens, source_tokens, source_description_tokens)
                   STORING (name, source, source_description, content, created_at, valid_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX CommunityNode_search_index
                   ON CommunityNode(name_tokens)
                   STORING (name, summary, created_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX ContradictedEdge_search_index
                   ON ContradictedEdge(invalidated_fact_tokens, invalidating_fact_tokens)
                   STORING (invalidated_edge_uuid, invalidating_edge_uuid, invalidated_fact, invalidating_fact, invalidated_at, created_at)
                   PARTITION BY group_id""",
                # Custom Entity Index: Optimized for Graph queries filtering by labels + name pattern
                # Use case: MATCH (p:Entity WHERE SEARCH(labels_tokens, 'User') AND SEARCH_NGRAMS(name_ngrams_tokens, 'pattern'))
                # - labels_tokens: TOKEN(labels) enables efficient ARRAY_INCLUDES via SEARCH()
                # - name_ngrams_tokens: TOKENIZE_NGRAMS enables REGEXP_CONTAINS pattern matching
                # Note: labels (ARRAY) excluded from STORING - not allowed for array fields
                """CREATE SEARCH INDEX EntityNodeCustomEntityIndex
                   ON EntityNode(labels_tokens, name_ngrams_tokens)
                   STORING (name, summary, attributes)
                   PARTITION BY group_id""",
                # ============================================================
                # GLOBAL SEARCH INDEXES (No Partition - for queries without group_id filter)
                # ============================================================
                # These indexes support SEARCH() queries that do NOT include group_id
                # in the same WHERE clause. Spanner requires partitioned indexes to have
                # an equality filter on the partition key alongside SEARCH().
                #
                # Use case: Subqueries that perform SEARCH() first, then filter by group_id
                # in an outer query (current code pattern in search_utils.py)
                #
                # Naming convention: *_global_search_index
                """CREATE SEARCH INDEX EntityNode_global_search_index
                   ON EntityNode(name_tokens, summary_tokens)
                   STORING (name, summary, attributes, created_at, group_id)""",
                """CREATE SEARCH INDEX EntityEdge_global_search_index
                   ON EntityEdge(name_tokens, fact_tokens)
                   STORING (name, fact, source_node_uuid, target_node_uuid, created_at, expired_at, valid_at, invalid_at, group_id)""",
                """CREATE SEARCH INDEX EpisodicNode_global_search_index
                   ON EpisodicNode(content_tokens, source_tokens, source_description_tokens)
                   STORING (name, source, source_description, content, created_at, valid_at, group_id)""",
                """CREATE SEARCH INDEX CommunityNode_global_search_index
                   ON CommunityNode(name_tokens)
                   STORING (name, summary, created_at, group_id)""",
                """CREATE SEARCH INDEX ContradictedEdge_global_search_index
                   ON ContradictedEdge(invalidated_fact_tokens, invalidating_fact_tokens)
                   STORING (invalidated_edge_uuid, invalidating_edge_uuid, invalidated_fact, invalidating_fact, invalidated_at, created_at, group_id)""",
                # ============================================================
                # SECONDARY INDEXES for group_id filtering (Hybrid KNN Strategy)
                # ============================================================
                # These indexes are CRITICAL for the Hybrid KNN (Exact K-Nearest Neighbors)
                # approach used for semantic/vector similarity search.
                #
                # WHY HYBRID KNN INSTEAD OF VECTOR INDEX (ANN)?
                # ---------------------------------------------
                # Given our data profile:
                #   - Per partition: 300 – 10,000 records (EntityEdge per group_id)
                #   - Partitions: 1,000,000+ group_ids (tenants/users)
                #   - Total records: 300M – 10B+ rows
                #
                # A global Vector Index (ANN) would:
                #   - Build a single massive index across ALL groups
                #   - Require post-filtering by group_id (inefficient)
                #   - Provide only ~90-95% recall (approximate results)
                #   - Have limited horizontal scaling
                #
                # The Hybrid KNN approach:
                #   - Uses secondary index to FIRST isolate the partition (group_id)
                #   - Then computes exact COSINE_DISTANCE() on 300-10,000 rows only
                #   - Provides 100% recall (exact nearest neighbors)
                #   - Scales perfectly: query cost = O(partition size), not O(total data)
                #
                # QUERY PATTERN FOR EXACT KNN:
                # ----------------------------
                # SELECT uuid, name, fact,
                #        COSINE_DISTANCE(fact_embedding, @query_embedding) AS distance
                # FROM EntityEdge
                # WHERE group_id = @group_id
                #   AND fact_embedding IS NOT NULL
                # ORDER BY distance
                # LIMIT @top_k;
                #
                # HOW IT WORKS:
                # 1. Spanner uses EntityEdge_group_id_idx to seek directly to partition
                # 2. Scans only 300-10,000 rows within that partition
                # 3. Computes exact cosine distance for each row (trivial at this scale)
                # 4. Returns true top-k nearest neighbors (100% recall)
                #
                # SCALING CHARACTERISTICS:
                # ------------------------
                # | Scale                              | Query Performance           |
                # |------------------------------------|----------------------------|
                # | 1M groups × 1K edges = 1B rows     | ✅ Touches ~1K rows only   |
                # | 10M groups × 5K edges = 50B rows   | ✅ Touches ~5K rows only   |
                # | Query latency                      | Sub-100ms regardless of    |
                # |                                    | total database size        |
                #
                # WHEN WOULD YOU NEED VECTOR INDEX (ANN)?
                # - Cross-group semantic search (find similar edges across ALL groups)
                # - Single partition grows to 100K+ records
                # - For our multi-tenant use case: NOT NEEDED
                """CREATE INDEX EpisodicNode_group_id_idx ON EpisodicNode(group_id)""",
                """CREATE INDEX EntityNode_group_id_idx ON EntityNode(group_id)""",
                """CREATE INDEX EntityEdge_group_id_idx ON EntityEdge(group_id)""",
                """CREATE INDEX EpisodicEdge_group_id_idx ON EpisodicEdge(group_id)""",
                """CREATE INDEX CommunityNode_group_id_idx ON CommunityNode(group_id)""",
                """CREATE INDEX ContradictedEdge_group_id_idx ON ContradictedEdge(group_id)""",
                # ============================================================
                # GRAPH TRAVERSAL INDEXES (source/target node lookups)
                # ============================================================
                # Optimize edge lookups during BFS/MATCH traversals.
                # Used when following edges from node to node in graph queries.
                # Example: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                """CREATE INDEX EntityEdge_source_node_idx ON EntityEdge(source_node_uuid)""",
                """CREATE INDEX EntityEdge_target_node_idx ON EntityEdge(target_node_uuid)""",
                """CREATE INDEX EpisodicEdge_source_node_idx ON EpisodicEdge(source_node_uuid)""",
                """CREATE INDEX EpisodicEdge_target_node_idx ON EpisodicEdge(target_node_uuid)""",
                # ============================================================
                # COMPOSITE INDEXES (group_id + graph traversal)
                # ============================================================
                # Advanced optimization for queries filtering by group_id AND traversing.
                # These allow Spanner to satisfy both predicates from a single index scan.
                # Example: MATCH (e:Episodic)-[]->(:Entity) WHERE e.group_id = 'user_123'
                """CREATE INDEX EntityEdge_group_source_idx ON EntityEdge(group_id, source_node_uuid)""",
                """CREATE INDEX EpisodicEdge_group_source_idx ON EpisodicEdge(group_id, source_node_uuid)""",
            ]

            # ============================================================
            # COMMON TABLES (for shared/common group_id storage)
            # ============================================================
            # These tables mirror the user-specific tables but are dedicated to
            # storing "common" knowledge that is shared across all users.
            # This separation provides:
            # - Better isolation between user and common data
            # - Independent scaling and performance optimization
            # - Cleaner data management and potential different retention policies
            #
            # Only created when enable_common_tables=True (default)
            common_schema_statements = [
                # Create sequences for Common tables
                """CREATE SEQUENCE CommonEntityNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommonEntityEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommonEpisodicNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommonCommunityNodeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommonEpisodicEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                """CREATE SEQUENCE CommonContradictedEdgeSequence OPTIONS (
                    sequence_kind='bit_reversed_positive'
                )""",
                # Common Entity Node table
                """CREATE TABLE CommonEntityNode (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonEntityNodeSequence)),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  labels ARRAY<STRING(256)>,
                  created_at TIMESTAMP NOT NULL,
                  name_embedding ARRAY<FLOAT64>(vector_length=>768),
                  summary STRING(MAX),
                  attributes JSON,
                  name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN,
                  summary_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(summary)) HIDDEN,
                  labels_tokens TOKENLIST AS (TOKEN(labels)) HIDDEN,
                  name_ngrams_tokens TOKENLIST AS (TOKENIZE_NGRAMS(name, ngram_size_min=>3, ngram_size_max=>4)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # Common Entity Edge table
                """CREATE TABLE CommonEntityEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonEntityEdgeSequence)),
                  source_node_uuid STRING(256),
                  target_node_uuid STRING(256),
                  labels ARRAY<STRING(256)>,
                  fact STRING(MAX),
                  fact_embedding ARRAY<FLOAT64>(vector_length=>768),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  episodes ARRAY<STRING(256)>,
                  created_at TIMESTAMP NOT NULL,
                  expired_at TIMESTAMP,
                  valid_at TIMESTAMP,
                  invalid_at TIMESTAMP,
                  attributes JSON,
                  name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN,
                  fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(fact)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # Common Episodic Node table
                """CREATE TABLE CommonEpisodicNode (
                    id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonEpisodicNodeSequence)),
                    source STRING(256),
                    source_description STRING(MAX),
                    content STRING(MAX),
                    entity_edges ARRAY<STRING(MAX)>,
                    uuid STRING(256),
                    name STRING(MAX),
                    group_id STRING(256),
                    created_at TIMESTAMP NOT NULL,
                    valid_at TIMESTAMP NOT NULL,
                    content_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(content)) HIDDEN,
                    source_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(source)) HIDDEN,
                    source_description_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(source_description)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # Common Community Node table
                """CREATE TABLE CommonCommunityNode (
                    id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonCommunityNodeSequence)),
                    uuid STRING(256),
                    name STRING(MAX),
                    name_embedding ARRAY<FLOAT64>(vector_length=>768),
                    group_id STRING(256),
                    created_at TIMESTAMP NOT NULL,
                    summary STRING(MAX),
                    labels ARRAY<STRING(256)>,
                    name_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(name)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # Common Episodic Edge table
                """CREATE TABLE CommonEpisodicEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonEpisodicEdgeSequence)),
                  source_node_uuid STRING(256),
                  target_node_uuid STRING(256),
                  uuid STRING(256),
                  name STRING(MAX),
                  group_id STRING(256),
                  created_at TIMESTAMP NOT NULL
                ) PRIMARY KEY(uuid)""",
                # Common Contradicted Edge table
                """CREATE TABLE CommonContradictedEdge (
                  id INT64 DEFAULT (GET_NEXT_SEQUENCE_VALUE(SEQUENCE CommonContradictedEdgeSequence)),
                  uuid STRING(256),
                  invalidated_edge_uuid STRING(256),
                  invalidating_edge_uuid STRING(256),
                  invalidated_fact STRING(MAX),
                  invalidating_fact STRING(MAX),
                  invalidated_at TIMESTAMP NOT NULL,
                  group_id STRING(256),
                  invalidated_edge_data JSON,
                  invalidating_edge_data JSON,
                  created_at TIMESTAMP NOT NULL,
                  invalidated_fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(invalidated_fact)) HIDDEN,
                  invalidating_fact_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(invalidating_fact)) HIDDEN
                ) PRIMARY KEY(uuid)""",
                # ============================================================
                # COMMON SEARCH INDEXES (Full-Text Search with Partition Isolation)
                # ============================================================
                """CREATE SEARCH INDEX CommonEntityNode_search_index
                   ON CommonEntityNode(name_tokens, summary_tokens, labels_tokens, name_ngrams_tokens)
                   STORING (name, summary, attributes, created_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX CommonEntityEdge_search_index
                   ON CommonEntityEdge(name_tokens, fact_tokens)
                   STORING (name, fact, source_node_uuid, target_node_uuid, created_at, expired_at, valid_at, invalid_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX CommonEpisodicNode_search_index
                   ON CommonEpisodicNode(content_tokens, source_tokens, source_description_tokens)
                   STORING (name, source, source_description, content, created_at, valid_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX CommonCommunityNode_search_index
                   ON CommonCommunityNode(name_tokens)
                   STORING (name, summary, created_at)
                   PARTITION BY group_id""",
                """CREATE SEARCH INDEX CommonContradictedEdge_search_index
                   ON CommonContradictedEdge(invalidated_fact_tokens, invalidating_fact_tokens)
                   STORING (invalidated_edge_uuid, invalidating_edge_uuid, invalidated_fact, invalidating_fact, invalidated_at, created_at)
                   PARTITION BY group_id""",
                # Custom Entity Index for Common tables
                """CREATE SEARCH INDEX CommonEntityNodeCustomEntityIndex
                   ON CommonEntityNode(labels_tokens, name_ngrams_tokens)
                   STORING (name, summary, attributes)
                   PARTITION BY group_id""",
                # ============================================================
                # COMMON GLOBAL SEARCH INDEXES (No Partition)
                # ============================================================
                """CREATE SEARCH INDEX CommonEntityNode_global_search_index
                   ON CommonEntityNode(name_tokens, summary_tokens)
                   STORING (name, summary, attributes, created_at, group_id)""",
                """CREATE SEARCH INDEX CommonEntityEdge_global_search_index
                   ON CommonEntityEdge(name_tokens, fact_tokens)
                   STORING (name, fact, source_node_uuid, target_node_uuid, created_at, expired_at, valid_at, invalid_at, group_id)""",
                """CREATE SEARCH INDEX CommonEpisodicNode_global_search_index
                   ON CommonEpisodicNode(content_tokens, source_tokens, source_description_tokens)
                   STORING (name, source, source_description, content, created_at, valid_at, group_id)""",
                """CREATE SEARCH INDEX CommonCommunityNode_global_search_index
                   ON CommonCommunityNode(name_tokens)
                   STORING (name, summary, created_at, group_id)""",
                """CREATE SEARCH INDEX CommonContradictedEdge_global_search_index
                   ON CommonContradictedEdge(invalidated_fact_tokens, invalidating_fact_tokens)
                   STORING (invalidated_edge_uuid, invalidating_edge_uuid, invalidated_fact, invalidating_fact, invalidated_at, created_at, group_id)""",
                # ============================================================
                # COMMON SECONDARY INDEXES for group_id filtering
                # ============================================================
                """CREATE INDEX CommonEpisodicNode_group_id_idx ON CommonEpisodicNode(group_id)""",
                """CREATE INDEX CommonEntityNode_group_id_idx ON CommonEntityNode(group_id)""",
                """CREATE INDEX CommonEntityEdge_group_id_idx ON CommonEntityEdge(group_id)""",
                """CREATE INDEX CommonEpisodicEdge_group_id_idx ON CommonEpisodicEdge(group_id)""",
                """CREATE INDEX CommonCommunityNode_group_id_idx ON CommonCommunityNode(group_id)""",
                """CREATE INDEX CommonContradictedEdge_group_id_idx ON CommonContradictedEdge(group_id)""",
                # ============================================================
                # COMMON GRAPH TRAVERSAL INDEXES
                # ============================================================
                """CREATE INDEX CommonEntityEdge_source_node_idx ON CommonEntityEdge(source_node_uuid)""",
                """CREATE INDEX CommonEntityEdge_target_node_idx ON CommonEntityEdge(target_node_uuid)""",
                """CREATE INDEX CommonEpisodicEdge_source_node_idx ON CommonEpisodicEdge(source_node_uuid)""",
                """CREATE INDEX CommonEpisodicEdge_target_node_idx ON CommonEpisodicEdge(target_node_uuid)""",
                # ============================================================
                # COMMON COMPOSITE INDEXES (group_id + graph traversal)
                # ============================================================
                """CREATE INDEX CommonEntityEdge_group_source_idx ON CommonEntityEdge(group_id, source_node_uuid)""",
                """CREATE INDEX CommonEpisodicEdge_group_source_idx ON CommonEpisodicEdge(group_id, source_node_uuid)""",
            ]

            # Conditionally include Common* schema statements
            if self._enable_common_tables:
                schema_statements.extend(common_schema_statements)
                logger.info('Common tables feature enabled - will create Common* schema elements')
            else:
                logger.info('Common tables feature disabled - skipping Common* schema elements')

            # Property graph definition - created separately to handle partial schema scenarios
            property_graph_statement = """CREATE PROPERTY GRAPH GRAPHITI
                  NODE TABLES(
                    EntityNode
                      KEY (uuid)
                      LABEL Entity
                      PROPERTIES (
                         id, uuid, name, group_id, labels, created_at, name_embedding, summary, attributes
                      ),
                     EpisodicNode
                      KEY (uuid)
                      LABEL Episodic
                      PROPERTIES (
                        id, source, source_description, content, entity_edges, uuid, name, group_id, created_at, valid_at),
                    CommunityNode
                      KEY (uuid)
                      LABEL Community
                      PROPERTIES (
                        id, uuid, name, name_embedding, group_id, created_at, summary, labels
                      )
                  )
                  EDGE TABLES ( 
                      EpisodicEdge
                      KEY (uuid)
                      SOURCE KEY (source_node_uuid) REFERENCES EpisodicNode (uuid)
                      DESTINATION KEY (target_node_uuid) REFERENCES EntityNode (uuid)
                      LABEL MENTIONS
                      PROPERTIES (
                        id, source_node_uuid, target_node_uuid, uuid, name, group_id, created_at
                      ),
                      EntityEdge
                      KEY (uuid)
                      SOURCE KEY (source_node_uuid) REFERENCES EntityNode (uuid)
                      DESTINATION KEY (target_node_uuid) REFERENCES EntityNode (uuid)
                      LABEL RELATES_TO
                      PROPERTIES (
                        id, source_node_uuid, target_node_uuid, labels, fact, fact_embedding, uuid, name, group_id, episodes, created_at, expired_at, valid_at, invalid_at, attributes)
                  )"""

            # Property graph definition for Common tables - separate graph for shared knowledge
            common_property_graph_statement = """CREATE PROPERTY GRAPH COMMON_GRAPHITI
                  NODE TABLES(
                    CommonEntityNode
                      KEY (uuid)
                      LABEL Entity
                      PROPERTIES (
                         id, uuid, name, group_id, labels, created_at, name_embedding, summary, attributes
                      ),
                     CommonEpisodicNode
                      KEY (uuid)
                      LABEL Episodic
                      PROPERTIES (
                        id, source, source_description, content, entity_edges, uuid, name, group_id, created_at, valid_at),
                    CommonCommunityNode
                      KEY (uuid)
                      LABEL Community
                      PROPERTIES (
                        id, uuid, name, name_embedding, group_id, created_at, summary, labels
                      )
                  )
                  EDGE TABLES ( 
                      CommonEpisodicEdge
                      KEY (uuid)
                      SOURCE KEY (source_node_uuid) REFERENCES CommonEpisodicNode (uuid)
                      DESTINATION KEY (target_node_uuid) REFERENCES CommonEntityNode (uuid)
                      LABEL MENTIONS
                      PROPERTIES (
                        id, source_node_uuid, target_node_uuid, uuid, name, group_id, created_at
                      ),
                      CommonEntityEdge
                      KEY (uuid)
                      SOURCE KEY (source_node_uuid) REFERENCES CommonEntityNode (uuid)
                      DESTINATION KEY (target_node_uuid) REFERENCES CommonEntityNode (uuid)
                      LABEL RELATES_TO
                      PROPERTIES (
                        id, source_node_uuid, target_node_uuid, labels, fact, fact_embedding, uuid, name, group_id, episodes, created_at, expired_at, valid_at, invalid_at, attributes)
                  )"""

            # Execute DDL statements to create schema (tables, sequences, indexes)
            # Execute them individually to handle partial schema scenarios
            for statement in schema_statements:
                try:
                    await self.execute_ddl([statement])
                except Exception as e:
                    error_str = str(e).lower()
                    if 'duplicate' in error_str or 'already exists' in error_str:
                        # Skip duplicate errors - object already exists
                        continue
                    else:
                        # Re-raise other errors
                        raise

            logger.info('Schema tables and indexes created/verified successfully')

            # Execute property graph creation separately
            try:
                await self.execute_ddl([property_graph_statement])
                logger.info('Property graph GRAPHITI created successfully')
            except Exception as e:
                # Check if error is about duplicate/existing property graph
                error_str = str(e).lower()
                if 'duplicate' in error_str or 'already exists' in error_str:
                    logger.info('Property graph GRAPHITI already exists')
                else:
                    logger.error(f'Error creating property graph GRAPHITI: {e}')
                    raise

            # Execute Common property graph creation separately (only if enabled)
            if self._enable_common_tables:
                try:
                    await self.execute_ddl([common_property_graph_statement])
                    logger.info('Property graph COMMON_GRAPHITI created successfully')
                except Exception as e:
                    # Check if error is about duplicate/existing property graph
                    error_str = str(e).lower()
                    if 'duplicate' in error_str or 'already exists' in error_str:
                        logger.info('Property graph COMMON_GRAPHITI already exists')
                    else:
                        logger.error(f'Error creating property graph COMMON_GRAPHITI: {e}')
                        raise

        except Exception as e:
            # Check if error is about duplicate/existing schema objects
            error_str = str(e).lower()
            if 'duplicate' in error_str or 'already exists' in error_str:
                logger.info('Schema objects already exist, skipping creation')
            else:
                logger.error(f'Error initializing schema: {e}')
                raise

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """
        Build indices and constraints in the Spanner database.

        This method is provided for compatibility with the Graphiti interface.
        For Spanner, the schema (including indices and constraints) is automatically
        initialized when the driver is first used.

        Args:
            delete_existing: If True, will recreate the entire schema
        """
        if delete_existing:
            logger.info('Recreating Spanner schema...')
            # For delete_existing=True, we could drop and recreate the schema
            # but this is dangerous for production, so we'll just log a warning
            logger.warning(
                "delete_existing=True is not supported for Spanner driver. Schema will be created if it doesn't exist."
            )
            # Reset the flag so schema can be recreated
            self._schema_initialized = False
        
        # Always ensure schema is initialized (idempotent - skips if already done)
        await self.initialize_schema()

    async def execute_query(
        self, cypher_query_: LiteralString, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        """Execute a query directly without session management.
        
        If use_common_tables=True was set during driver creation, this method
        will automatically rewrite table names in the query to use Common* tables
        (e.g., EntityNode -> CommonEntityNode).
        """
        import time

        start_time = time.perf_counter()

        # Rewrite query to use Common* tables if enabled at driver level
        if self._use_common_tables:
            original_query = cypher_query_
            cypher_query_ = _rewrite_query_for_common_tables(cypher_query_)  # type: ignore
            if original_query != cypher_query_:
                logger.debug(f'[COMMON_TABLES] Rewrote query to use Common* tables')

        # Acquire a session from the pool
        session_start = time.perf_counter()
        session_name = await self.session_pool.acquire()
        session_time = time.perf_counter() - session_start
        logger.debug(
            f'[PROFILING] execute_query - Session acquisition: {session_time * 1000:.2f}ms'
        )

        try:
            # Format parameters
            param_start = time.perf_counter()
            params_struct, param_types_t = _format_spanner_params(kwargs)
            param_time = time.perf_counter() - param_start
            logger.debug(
                f'[PROFILING] execute_query - Parameter formatting: {param_time * 1000:.2f}ms'
            )

            rows = []
            field_names = None

            # For read queries, use single use transaction
            # This includes SELECT queries and GRAPH queries with MATCH (read operations)
            query_upper = cypher_query_.strip().upper()

            # Check for write keywords with word boundaries to avoid false positives (e.g., created_at contains CREATE)
            write_keywords = ['INSERT', 'UPDATE', 'DELETE', 'MERGE']
            found_write_keywords = [
                kw for kw in write_keywords if re.search(r'\b' + kw + r'\b', query_upper)
            ]

            # Detect read queries:
            # - SELECT ... (direct select)
            # - WITH ... SELECT ... (CTE followed by select - common for vector searches)
            # - GRAPH ... MATCH ... (graph read operations without write keywords)
            is_select_query = query_upper.startswith('SELECT')
            is_cte_query = query_upper.startswith('WITH') and 'SELECT' in query_upper
            is_graph_read_query = (
                query_upper.startswith('GRAPH')
                and 'MATCH' in query_upper
                and not found_write_keywords
            )

            is_read_query = (
                (is_select_query or is_cte_query or is_graph_read_query)
                and not found_write_keywords
            )

            # Check if this is a SEARCH query (requires read-only transaction)
            uses_search = 'SEARCH(' in query_upper or 'COSINE_DISTANCE' in query_upper

            if is_read_query:
                exec_start = time.perf_counter()

                # For SEARCH queries, explicitly use read-only single-use transaction
                if uses_search:
                    logger.info('[SEARCH] Detected SEARCH query, using read-only transaction')
                    request = spanner.ExecuteSqlRequest(
                        session=session_name,
                        sql=cypher_query_,
                        params=params_struct,
                        param_types=param_types_t,
                        transaction=transaction.TransactionSelector(
                            single_use=transaction.TransactionOptions(
                                read_only=transaction.TransactionOptions.ReadOnly()
                            )
                        ),
                    )
                else:
                    request = spanner.ExecuteSqlRequest(
                        session=session_name,
                        sql=cypher_query_,
                        params=params_struct,
                        param_types=param_types_t,
                    )
                stream_start = time.perf_counter()
                stream_result = await self.client.execute_streaming_sql(request)
                stream_init_time = time.perf_counter() - stream_start
                logger.debug(
                    f'[PROFILING] execute_query - Stream initialization: {stream_init_time * 1000:.2f}ms'
                )

                fetch_start = time.perf_counter()
                async for partial_result in stream_result:
                    # Get field names from metadata
                    if field_names is None and partial_result.metadata:
                        field_names = [
                            field.name for field in partial_result.metadata.row_type.fields
                        ]

                    # partial_result.values is a FLAT list of all field values
                    # We need to group them by number of fields to reconstruct rows
                    if field_names:
                        num_fields = len(field_names)
                        all_values = [
                            _extract_value_from_protobuf(val) for val in partial_result.values
                        ]

                        # Group values into rows
                        for i in range(0, len(all_values), num_fields):
                            row_values = all_values[i : i + num_fields]
                            if len(row_values) == num_fields:  # Only add complete rows
                                row_dict = dict(zip(field_names, row_values, strict=False))
                                rows.append(row_dict)
                    else:
                        # If no field names, just add raw values
                        rows.extend(
                            [_extract_value_from_protobuf(val) for val in partial_result.values]
                        )

                fetch_time = time.perf_counter() - fetch_start
                exec_time = time.perf_counter() - exec_start
                logger.debug(
                    f'[PROFILING] execute_query - Data fetching: {fetch_time * 1000:.2f}ms ({len(rows)} rows)'
                )
                logger.debug(
                    f'[PROFILING] execute_query - Total query execution: {exec_time * 1000:.2f}ms'
                )

            else:
                # For write queries, check if we can use BatchWrite for INSERT OR UPDATE
                # This eliminates 409 conflicts for parallel upsert operations
                if SPANNER_BATCH_WRITE_CONFIG['enabled']:
                    mutation_data = _parse_insert_or_update_query(cypher_query_, kwargs)
                    if mutation_data:
                        # Use BatchWrite for this INSERT OR UPDATE query
                        logger.debug(
                            f'[BATCH_WRITE] Converting INSERT OR UPDATE to BatchWrite: '
                            f'{mutation_data["table"]}'
                        )
                        session = SpannerDriverSession(self.client, self.database_path)
                        try:
                            successful, failed = await session.run_mutations_batch_write(
                                [mutation_data], mutations_per_group=1
                            )
                            if failed > 0:
                                logger.warning(
                                    f'[BATCH_WRITE] execute_query mutation failed: '
                                    f'{mutation_data["table"]}'
                                )
                            # Return empty result (INSERT OR UPDATE doesn't return rows)
                            return [], None, None
                        finally:
                            await session.close()
                    else:
                        # Parser failed for INSERT OR UPDATE query
                        if 'INSERT OR UPDATE' in cypher_query_.upper():
                            logger.warning(
                                f'[BATCH_WRITE] Failed to parse INSERT OR UPDATE query. '
                                f'Query: {cypher_query_[:200]}... Params: {list(kwargs.keys())}'
                            )

                # Fallback to transactional write for other queries or if BatchWrite is disabled
                # Log the query type to help debug 409 conflicts
                logger.warning(
                    f'[TRANSACTION] Fallback to transactional write (potential 409 source). '
                    f'Query: {cypher_query_[:150]}'
                )
                max_retries = SPANNER_RETRY_CONFIG['max_retries']

                for attempt in range(max_retries + 1):
                    try:
                        rows = []
                        field_names = None

                        options = transaction.TransactionOptions(
                            read_write=transaction.TransactionOptions.ReadWrite()
                        )
                        begin_request = spanner.BeginTransactionRequest(
                            session=session_name, options=options
                        )
                        transaction_obj = await self.client.begin_transaction(begin_request)
                        try:
                            request = spanner.ExecuteSqlRequest(
                                session=session_name,
                                sql=cypher_query_,
                                params=params_struct,
                                param_types=param_types_t,
                                transaction={'id': transaction_obj.id},  # Set transaction ID
                            )
                            async for partial_result in await self.client.execute_streaming_sql(
                                request
                            ):
                                # Get field names from metadata
                                if field_names is None and partial_result.metadata:
                                    field_names = [
                                        field.name
                                        for field in partial_result.metadata.row_type.fields
                                    ]

                                # partial_result.values is a FLAT list of all field values
                                # We need to group them by number of fields to reconstruct rows
                                if field_names:
                                    num_fields = len(field_names)
                                    all_values = [
                                        _extract_value_from_protobuf(val)
                                        for val in partial_result.values
                                    ]

                                    # Group values into rows
                                    for i in range(0, len(all_values), num_fields):
                                        row_values = all_values[i : i + num_fields]
                                        if len(row_values) == num_fields:  # Only add complete rows
                                            row_dict = dict(
                                                zip(field_names, row_values, strict=False)
                                            )
                                            rows.append(row_dict)
                                else:
                                    # If no field names, just add raw values
                                    rows.extend(
                                        [
                                            _extract_value_from_protobuf(val)
                                            for val in partial_result.values
                                        ]
                                    )
                            # Commit the transaction
                            commit_request = spanner.CommitRequest(
                                session=session_name, transaction_id=transaction_obj.id
                            )
                            await self.client.commit(commit_request)
                            # Success - break out of retry loop
                            break

                        except Exception as e:
                            # Rollback on error
                            try:
                                rollback_request = spanner.RollbackRequest(
                                    session=session_name, transaction_id=transaction_obj.id
                                )
                                await self.client.rollback(rollback_request)
                            except Exception:
                                pass  # Ignore rollback errors
                            raise e

                    except Exception as e:
                        # Check if error is retryable
                        if not _is_retryable_error(e):
                            logger.error(f'[TRANSACTION] Non-retryable error in execute_query: {e}')
                            raise

                        # Check if we have retries left
                        if attempt >= max_retries:
                            logger.error(
                                f'[TRANSACTION] Write query aborted after {max_retries} retries. '
                                f'Last error: {e}'
                            )
                            raise

                        # Calculate delay and wait
                        delay = _calculate_retry_delay(attempt)
                        logger.warning(
                            f'[TRANSACTION] Write conflict detected (attempt {attempt + 1}/{max_retries + 1}). '
                            f'Retrying in {delay:.2f}s. Error: {str(e)[:200]}'
                        )
                        await asyncio.sleep(delay)

            return rows, None, None

        finally:
            # Return session to pool instead of deleting it
            cleanup_start = time.perf_counter()
            await self.session_pool.release(session_name)
            cleanup_time = time.perf_counter() - cleanup_start

            total_time = time.perf_counter() - start_time
            logger.debug(
                f'[PROFILING] execute_query - Session release: {cleanup_time * 1000:.2f}ms'
            )
            logger.debug(f'[PROFILING] execute_query - TOTAL TIME: {total_time * 1000:.2f}ms')

    def session(self, database: str | None = None) -> GraphDriverSession:
        """Create and return a new session."""
        session = SpannerDriverSession(self.client, self.database_path)
        # Set a reference to the parent driver so the session can ensure schema initialization
        session._parent_driver = self
        return session

    async def save_contradicted_edges(
        self,
        invalidated_edges: list[tuple[Any, Any]],
        group_id: str,
    ) -> None:
        """
        Save contradicted edges to the ContradictedEdge table.

        This is a Spanner-specific feature that stores edges that have been invalidated
        due to contradictions for audit and analysis purposes.

        Args:
            invalidated_edges: List of tuples (invalidated_edge, invalidating_edge) representing
                             edges that were invalidated and the new edges that invalidated them
            group_id: The group_id for all the edges
        """
        import json
        from datetime import datetime, timezone
        from uuid import uuid4

        if not invalidated_edges:
            return

        logger.info(
            f'[CONTRADICTED EDGES] Saving {len(invalidated_edges)} contradicted edges to Spanner'
        )

        # Prepare mutation data as dictionaries for run_mutations_batch_write
        mutations_data: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        columns = [
            'uuid',
            'invalidated_edge_uuid',
            'invalidating_edge_uuid',
            'invalidated_fact',
            'invalidating_fact',
            'invalidated_at',
            'group_id',
            'invalidated_edge_data',
            'invalidating_edge_data',
            'created_at',
        ]

        for invalidated_edge, invalidating_edge in invalidated_edges:
            # Create a unique UUID for this contradicted edge record
            record_uuid = str(uuid4())

            # Serialize full edge data as JSON for audit trail
            # Use a helper to serialize datetime objects in attributes
            def serialize_value(v: Any) -> Any:
                if isinstance(v, datetime):
                    return v.isoformat()
                if isinstance(v, dict):
                    return {k: serialize_value(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [serialize_value(item) for item in v]
                return v

            invalidated_data = {
                'uuid': invalidated_edge.uuid,
                'name': invalidated_edge.name,
                'fact': invalidated_edge.fact,
                'source_node_uuid': invalidated_edge.source_node_uuid,
                'target_node_uuid': invalidated_edge.target_node_uuid,
                'episodes': invalidated_edge.episodes,
                'created_at': invalidated_edge.created_at.isoformat()
                if invalidated_edge.created_at
                else None,
                'valid_at': invalidated_edge.valid_at.isoformat()
                if invalidated_edge.valid_at
                else None,
                'invalid_at': invalidated_edge.invalid_at.isoformat()
                if invalidated_edge.invalid_at
                else None,
                'expired_at': invalidated_edge.expired_at.isoformat()
                if invalidated_edge.expired_at
                else None,
                'attributes': serialize_value(invalidated_edge.attributes),
            }

            invalidating_data = {
                'uuid': invalidating_edge.uuid,
                'name': invalidating_edge.name,
                'fact': invalidating_edge.fact,
                'source_node_uuid': invalidating_edge.source_node_uuid,
                'target_node_uuid': invalidating_edge.target_node_uuid,
                'episodes': invalidating_edge.episodes,
                'created_at': invalidating_edge.created_at.isoformat()
                if invalidating_edge.created_at
                else None,
                'valid_at': invalidating_edge.valid_at.isoformat()
                if invalidating_edge.valid_at
                else None,
                'attributes': serialize_value(invalidating_edge.attributes),
            }

            # Build mutation dictionary for BatchWrite
            mutations_data.append(
                {
                    'table': 'ContradictedEdge',
                    'columns': columns,
                    'values': [
                        record_uuid,
                        invalidated_edge.uuid,
                        invalidating_edge.uuid,
                        invalidated_edge.fact,
                        invalidating_edge.fact,
                        invalidated_edge.invalid_at if invalidated_edge.invalid_at else now,
                        group_id,
                        json.dumps(invalidated_data),
                        json.dumps(invalidating_data),
                        now,
                    ],
                }
            )

        # Execute mutations using BatchWrite if enabled (conflict-free), otherwise use transaction
        if SPANNER_BATCH_WRITE_CONFIG['enabled']:
            # Use BatchWrite for conflict-free writes
            session = SpannerDriverSession(self.client, self.database_path)
            try:
                successful, failed = await session.run_mutations_batch_write(mutations_data)
                if failed > 0:
                    logger.warning(
                        f'[CONTRADICTED EDGES] BatchWrite: {successful} succeeded, {failed} failed'
                    )
                else:
                    logger.info(
                        f'[CONTRADICTED EDGES] Successfully saved {successful} contradicted edge '
                        'records via BatchWrite'
                    )
            finally:
                await session.close()
        else:
            # Fallback to transactional write - need to build Mutation objects
            mutations = []
            for mut_data in mutations_data:
                converted_values = [
                    _convert_value_for_mutation(v) for v in mut_data['values']
                ]
                mutation = types.Mutation(
                    insert=types.Mutation.Write(
                        table=mut_data['table'],
                        columns=mut_data['columns'],
                        values=[converted_values],
                    )
                )
                mutations.append(mutation)

            session_name = await self.session_pool.acquire()
            try:
                # Begin transaction
                options = transaction.TransactionOptions(
                    read_write=transaction.TransactionOptions.ReadWrite()
                )
                begin_request = spanner.BeginTransactionRequest(
                    session=session_name, options=options
                )
                transaction_obj = await self.client.begin_transaction(begin_request)

                commit_succeeded = False
                try:
                    # Commit with mutations
                    commit_request = spanner.CommitRequest(
                        session=session_name,
                        transaction_id=transaction_obj.id,
                        mutations=mutations,
                    )
                    await self.client.commit(commit_request)
                    commit_succeeded = True
                    logger.info(
                        f'[CONTRADICTED EDGES] Successfully saved {len(mutations)} '
                        'contradicted edge records'
                    )

                except Exception as e:
                    # Only rollback if commit didn't succeed
                    if not commit_succeeded:
                        try:
                            rollback_request = spanner.RollbackRequest(
                                session=session_name, transaction_id=transaction_obj.id
                            )
                            await self.client.rollback(rollback_request)
                            logger.debug('[CONTRADICTED EDGES] Transaction rolled back')
                        except Exception as rollback_error:
                            logger.debug(
                                '[CONTRADICTED EDGES] Rollback failed '
                                f'(transaction may have already ended): {rollback_error}'
                            )

                    logger.error(f'[CONTRADICTED EDGES] Error saving contradicted edges: {e}')
                    raise

            finally:
                await self.session_pool.release(session_name)

    async def close(self) -> None:
        """Close the driver and its connections, including the session pool."""
        logger.info('[SPANNER DRIVER] Closing driver and session pool...')
        await self.session_pool.close()
        logger.info('[SPANNER DRIVER] Driver closed')

    async def execute_ddl(self, ddl_statements: Sequence[str]) -> None:
        """Execute DDL statements using Spanner's DDL API.

        Args:
            ddl_statements: Sequence of DDL statements to execute
        """
        if not ddl_statements:
            return

        # Use update_database_ddl to execute DDL statements
        from google.api_core.exceptions import GoogleAPICallError
        from google.cloud.spanner_admin_database_v1 import DatabaseAdminAsyncClient

        admin_client = DatabaseAdminAsyncClient()

        try:
            operation = await admin_client.update_database_ddl(
                database=self.database_path,
                statements=list(ddl_statements),
            )
            # Wait for the operation to complete
            await operation.result()
            logger.info(f'Successfully executed {len(ddl_statements)} DDL statement(s)')
        except GoogleAPICallError as e:
            # Check if error is about duplicate columns (already exists)
            if 'Duplicate column name' in str(e):
                logger.info('DDL columns already exist, skipping...')
            else:
                raise

    async def delete_all_indexes(self) -> None:
        """Delete all indexes in the database."""
        # Create a session for index operations
        request = spanner.CreateSessionRequest(database=self.database_path)
        session = await self.client.create_session(request)

        try:
            # Get list of indexes
            sql_request = spanner.ExecuteSqlRequest(
                session=session.name,
                sql="SELECT index_name FROM information_schema.indexes WHERE index_type = 'INDEX'",
            )
            result = await self.client.execute_streaming_sql(sql_request)

            # Drop each index
            async with SpannerDriverSession(self.client, self.database_path) as driver_session:
                async for partial_result in result:
                    for value in partial_result.values:
                        index_name = _extract_value_from_protobuf(value)
                        await driver_session.run(f'DROP INDEX {index_name}')

        finally:
            request = spanner.DeleteSessionRequest(name=session.name)
            await self.client.delete_session(request)
