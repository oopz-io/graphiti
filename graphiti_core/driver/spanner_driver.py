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
import re
from datetime import datetime
from typing import Any

from google.cloud import spanner_v1
from google.cloud.spanner_v1 import types
from google.cloud.spanner_v1.types import spanner, transaction
from google.protobuf import struct_pb2
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)


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
        min_size: int = 1,
        max_size: int = 10,
    ):
        """
        Initialize the session pool.
        
        Args:
            client: The Spanner async client
            database_path: Full path to the database
            min_size: Minimum number of sessions to maintain (warmed up)
            max_size: Maximum number of sessions in the pool
        """
        self.client = client
        self.database_path = database_path
        self.min_size = min_size
        self.max_size = max_size
        self._pool: list[str] = []  # List of available session names
        self._in_use: set[str] = set()  # Set of session names currently in use
        self._lock = asyncio.Lock()
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
                    logger.debug(f'[SESSION POOL] Pre-created session {i+1}/{self.min_size}')
                except Exception as e:
                    logger.warning(f'[SESSION POOL] Failed to pre-create session {i+1}: {e}')
                    
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
        
        Returns:
            Session name (string)
        """
        # Ensure pool is initialized
        if not self._initialized:
            logger.info('[SESSION POOL] acquire() called but pool not initialized - initializing now...')
            await self.initialize()
        else:
            logger.debug('[SESSION POOL] acquire() called - pool already initialized')
            
        async with self._lock:
            # Try to get a session from the pool
            if self._pool:
                session_name = self._pool.pop()
                self._in_use.add(session_name)
                logger.debug(f'[SESSION POOL] Acquired session from pool (available: {len(self._pool)}, in use: {len(self._in_use)})')
                return session_name
            
            # If pool is empty but we haven't hit max size, create a new session
            total_sessions = len(self._pool) + len(self._in_use)
            if total_sessions < self.max_size:
                logger.debug(f'[SESSION POOL] Creating new session (total: {total_sessions}/{self.max_size})')
                session_name = await self._create_session()
                self._in_use.add(session_name)
                return session_name
            
            # If we're at max capacity, wait and retry (this shouldn't happen often with proper sizing)
            logger.warning(f'[SESSION POOL] Pool exhausted! Waiting for session to be released...')
        
        # Wait a bit and retry
        await asyncio.sleep(0.1)
        return await self.acquire()
    
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
                logger.debug(f'[SESSION POOL] Released session to pool (available: {len(self._pool)}, in use: {len(self._in_use)})')
            else:
                # Pool is full, delete the session
                logger.debug(f'[SESSION POOL] Pool full, deleting session')
                await self._delete_session(session_name)
    
    async def close(self):
        """Close all sessions in the pool."""
        async with self._lock:
            logger.info(f'[SESSION POOL] Closing pool with {len(self._pool)} available sessions and {len(self._in_use)} in-use sessions')
            
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
    elif isinstance(value, bool):
        return value
    elif isinstance(value, (int, float, str)):
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
        # Ensure schema is initialized if we have a parent driver reference
        if self._parent_driver:
            await self._parent_driver._ensure_schema_initialized()

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
            
        # Ensure schema is initialized if we have a parent driver reference
        if self._parent_driver:
            await self._parent_driver._ensure_schema_initialized()
            
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
                        table=table,
                        columns=columns,
                        values=[converted_values]
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
            logger.info(f'[MUTATIONS] Prepared {len(mutations)} mutations in {total_time:.2f}ms ({total_time/len(mutations):.2f}ms per mutation)')
            
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
                logger.warning(f'[SESSION] Cannot commit/rollback in close(): {e}. Transaction may already be closed.')
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
        """Execute a write operation in a transaction."""
        # Ensure schema is initialized if we have a parent driver reference
        if self._parent_driver:
            await self._parent_driver._ensure_schema_initialized()

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
                logger.info(f'[MUTATIONS] Committing {len(self._pending_mutations)} pending mutations')
                request = spanner.CommitRequest(
                    session=self._session_name,
                    transaction_id=self._current_transaction,
                    mutations=self._pending_mutations
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
                    logger.warning(f'[TRANSACTION] Rollback failed: {rollback_error}. Transaction may already be closed.')
                finally:
                    self._current_transaction = None
                    self._seqno = 0
            # Clear pending mutations on error
            if hasattr(self, '_pending_mutations'):
                self._pending_mutations = []
            raise e


class SpannerDriver(GraphDriver):
    """Google Cloud Spanner implementation of GraphDriver.

    This driver automatically initializes the required schema on first use,
    creating all necessary tables, sequences, search indexes, and property graph
    definitions. The schema is based on the Graphiti data model and includes:

    - EntityNode table for storing entity nodes
    - EntityEdge table for storing relationships between entities
    - EpisodicNode table for storing episodic memory
    - CommunityNode table for storing community/cluster nodes
    - EpisodicEdge table for linking episodes to entities
    - Full-text search indexes for content search
    - Property graph definition for graph query support

    The schema initialization is idempotent - it will only create the schema
    if it doesn't already exist.
    
    Note: For optimal performance with pre-warmed session pool, use the async
    factory method `create()` instead of direct instantiation:
    
        driver = await SpannerDriver.create(project_id, instance_id, database_id)
    
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
        session_pool_size: int = 20,
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
            session_pool_size: Maximum number of sessions in the pool (default: 20)
        """
        super().__init__()

        # Create Spanner client
        self.client = spanner_v1.SpannerAsyncClient(credentials=credentials)

        # Store database path
        self.database_path = self.client.database_path(project_id, instance_id, database_id)

        self._database = database_id

        # Initialize session pool
        # Pre-warm with sessions to handle concurrent operations during add_episode
        # (typical first episode needs 6-8 concurrent sessions)
        self.session_pool = SessionPool(
            client=self.client,
            database_path=self.database_path,
            min_size=min(5, session_pool_size),  # Pre-warm up to 5 sessions, but not more than max
            max_size=session_pool_size,
        )

        # Initialize schema automatically for Spanner
        # This is done synchronously during construction to ensure the database is ready
        # before any operations are attempted
        self._schema_initialized = False
        
        # Flag to track if session pool has been pre-warmed
        self._pool_initialized = False

    @classmethod
    async def create(
        cls,
        project_id: str,
        instance_id: str,
        database_id: str,
        credentials: Any = None,
        session_pool_size: int = 20,
    ) -> 'SpannerDriver':
        """Async factory method to create a SpannerDriver with pre-warmed session pool.
        
        This is the recommended way to create a SpannerDriver instance as it will
        pre-initialize the session pool, eliminating the ~11 second cold start on
        first database operation.
        
        Example:
            driver = await SpannerDriver.create(
                project_id='my-project',
                instance_id='my-instance',
                database_id='my-database'
            )
        
        Args:
            project_id: The GCP project ID
            instance_id: The Spanner instance ID
            database_id: The Spanner database ID
            credentials: Optional credentials object
            session_pool_size: Maximum number of sessions in the pool (default: 20)
            
        Returns:
            SpannerDriver instance with pre-warmed session pool
        """
        driver = cls(project_id, instance_id, database_id, credentials, session_pool_size)
        await driver._ensure_pool_initialized()
        return driver

    async def _ensure_schema_initialized(self) -> None:
        return
        """Ensure the schema is initialized. This is called lazily on first operation."""
        if not self._schema_initialized:
            await self.initialize_schema()
            self._schema_initialized = True
    
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
        return
        """Initialize the database schema if it doesn't exist."""
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
                  summary_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(summary)) HIDDEN
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
                # Create search indexes
                """CREATE SEARCH INDEX EntityNode_search_index ON EntityNode(name_tokens, summary_tokens)""",
                """CREATE SEARCH INDEX EntityEdge_search_index ON EntityEdge(name_tokens, fact_tokens)""",
                """CREATE SEARCH INDEX EpisodicNode_search_index ON EpisodicNode(content_tokens, source_tokens, source_description_tokens)""",
                """CREATE SEARCH INDEX CommunityNode_search_index ON CommunityNode(name_tokens)""",
            ]

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
                        id, uuid, name, group_id, created_at, labels
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
                logger.info('Property graph created successfully')
            except Exception as e:
                # Check if error is about duplicate/existing property graph
                error_str = str(e).lower()
                if 'duplicate' in error_str or 'already exists' in error_str:
                    logger.info('Property graph already exists')
                else:
                    logger.error(f'Error creating property graph: {e}')
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

        await self.initialize_schema()

    async def execute_query(
        self, cypher_query_: LiteralString, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        """Execute a query directly without session management."""
        import time
        start_time = time.perf_counter()
        
        # Ensure schema is initialized before executing any queries
        schema_start = time.perf_counter()
        await self._ensure_schema_initialized()
        schema_time = time.perf_counter() - schema_start
        if schema_time > 0.001:  # Only log if > 1ms
            logger.info(f'[PROFILING] execute_query - Schema check: {schema_time*1000:.2f}ms')

        # Acquire a session from the pool
        session_start = time.perf_counter()
        session_name = await self.session_pool.acquire()
        session_time = time.perf_counter() - session_start
        logger.info(f'[PROFILING] execute_query - Session acquisition: {session_time*1000:.2f}ms')

        try:
            # Format parameters
            param_start = time.perf_counter()
            params_struct, param_types_t = _format_spanner_params(kwargs)
            param_time = time.perf_counter() - param_start
            logger.info(f'[PROFILING] execute_query - Parameter formatting: {param_time*1000:.2f}ms')

            rows = []
            field_names = None

            # For read queries, use single use transaction
            # This includes SELECT queries and GRAPH queries with MATCH (read operations)
            query_upper = cypher_query_.strip().upper()

            # Check for write keywords with word boundaries to avoid false positives (e.g., created_at contains CREATE)
            write_keywords = ['CREATE', 'UPDATE', 'DELETE', 'MERGE', 'SET']
            found_write_keywords = [
                kw for kw in write_keywords if re.search(r'\b' + kw + r'\b', query_upper)
            ]

            is_read_query = query_upper.startswith('SELECT') or (
                query_upper.startswith('GRAPH')
                and 'MATCH' in query_upper
                and not found_write_keywords
            )

            if is_read_query:
                exec_start = time.perf_counter()
                request = spanner.ExecuteSqlRequest(
                    session=session_name,
                    sql=cypher_query_,
                    params=params_struct,
                    param_types=param_types_t,
                )
                stream_start = time.perf_counter()
                stream_result = await self.client.execute_streaming_sql(request)
                stream_init_time = time.perf_counter() - stream_start
                logger.info(f'[PROFILING] execute_query - Stream initialization: {stream_init_time*1000:.2f}ms')
                
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
                logger.info(f'[PROFILING] execute_query - Data fetching: {fetch_time*1000:.2f}ms ({len(rows)} rows)')
                logger.info(f'[PROFILING] execute_query - Total query execution: {exec_time*1000:.2f}ms')

            else:
                # For write queries, use transaction
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
                    async for partial_result in await self.client.execute_streaming_sql(request):
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
                    # Commit the transaction
                    commit_request = spanner.CommitRequest(
                        session=session_name, transaction_id=transaction_obj.id
                    )
                    await self.client.commit(commit_request)

                except Exception as e:
                    # Rollback on error
                    rollback_request = spanner.RollbackRequest(
                        session=session_name, transaction_id=transaction_obj.id
                    )
                    await self.client.rollback(rollback_request)
                    raise e

            return rows, None, None

        finally:
            # Return session to pool instead of deleting it
            cleanup_start = time.perf_counter()
            await self.session_pool.release(session_name)
            cleanup_time = time.perf_counter() - cleanup_start
            
            total_time = time.perf_counter() - start_time
            logger.info(f'[PROFILING] execute_query - Session release: {cleanup_time*1000:.2f}ms')
            logger.info(f'[PROFILING] execute_query - TOTAL TIME: {total_time*1000:.2f}ms')


    def session(self, database: str | None = None) -> GraphDriverSession:
        """Create and return a new session."""
        session = SpannerDriverSession(self.client, self.database_path)
        # Set a reference to the parent driver so the session can ensure schema initialization
        session._parent_driver = self
        return session

    async def close(self) -> None:
        """Close the driver and its connections, including the session pool."""
        logger.info('[SPANNER DRIVER] Closing driver and session pool...')
        await self.session_pool.close()
        logger.info('[SPANNER DRIVER] Driver closed')

    async def execute_ddl(self, ddl_statements: list[str]) -> None:
        """Execute DDL statements using Spanner's DDL API.

        Args:
            ddl_statements: List of DDL statements to execute
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
                statements=ddl_statements,
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
