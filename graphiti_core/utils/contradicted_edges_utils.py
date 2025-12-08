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
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.spanner_driver import SpannerDriver

logger = logging.getLogger(__name__)


async def get_invalidating_edges_by_invalidated_uuids(
    driver: GraphDriver,
    invalidated_edge_uuids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Retrieve invalidating edges information for given invalidated edge UUIDs.

    This function queries the ContradictedEdge table to find what edges invalidated
    the specified edges, providing proof and context for why edges were invalidated.

    Args:
        driver: The graph driver instance (must be SpannerDriver)
        invalidated_edge_uuids: List of invalidated edge UUIDs to look up

    Returns:
        Dictionary with invalidated edge UUID as key, containing:
        {
            'invalidated_edge_uuid': {
                'invalidated': {
                    'uuid': str,           # UUID of the invalidated edge
                    'fact': str,           # The fact that was invalidated
                    'created_at': datetime, # When the invalidated edge was created
                },
                'invalidating': {
                    'uuid': str,           # UUID of the edge that caused invalidation
                    'fact': str,           # The new fact that invalidated the old one
                    'created_at': datetime, # When the invalidating edge was created
                },
                'invalidated_at': datetime,  # When the invalidation occurred
                'contradiction_uuid': str,    # UUID of the contradiction record
            }
        }

    Example:
        >>> invalidated_uuids = ['edge-uuid-1', 'edge-uuid-2']
        >>> results = await get_invalidating_edges_by_invalidated_uuids(driver, invalidated_uuids)
        >>> # Access invalidation proof for a specific edge
        >>> proof = results['edge-uuid-1']
        >>> print(f'Old fact: {proof["invalidated"]["fact"]}')
        >>> print(f'New fact: {proof["invalidating"]["fact"]}')
        >>> print(f'Old edge created: {proof["invalidated"]["created_at"]}')
        >>> print(f'Invalidated at: {proof["invalidated_at"]}')

    Raises:
        ValueError: If driver is not a SpannerDriver instance

    Note:
        This function only works with SpannerDriver as it queries the Spanner-specific
        ContradictedEdge table. Returns empty dict if no contradictions found.
    """
    if not isinstance(driver, SpannerDriver):
        raise ValueError(
            'get_invalidating_edges_by_invalidated_uuids only works with SpannerDriver. '
            f'Got {type(driver).__name__} instead.'
        )

    if not invalidated_edge_uuids:
        logger.debug('No invalidated edge UUIDs provided, returning empty dict')
        return {}

    logger.info(
        f'Retrieving invalidating edges for {len(invalidated_edge_uuids)} invalidated edge(s)'
    )

    # Query the ContradictedEdge table
    query = """
    SELECT 
        uuid AS contradiction_uuid,
        invalidated_edge_uuid,
        invalidating_edge_uuid,
        invalidated_fact,
        invalidating_fact,
        invalidated_at,
        invalidated_edge_data,
        invalidating_edge_data
    FROM ContradictedEdge
    WHERE invalidated_edge_uuid IN UNNEST(@invalidated_uuids)
    ORDER BY invalidated_at DESC
    """

    try:
        records, _, _ = await driver.execute_query(
            query,
            invalidated_uuids=invalidated_edge_uuids,
        )

        import json

        # Build the result dictionary
        result: dict[str, dict[str, Any]] = {}

        for record in records:
            invalidated_uuid = record.get('invalidated_edge_uuid')

            # Skip records with missing UUID
            if not invalidated_uuid:
                continue

            # Parse JSON data to extract created_at timestamps
            invalidated_data = json.loads(record.get('invalidated_edge_data', '{}'))
            invalidating_data = json.loads(record.get('invalidating_edge_data', '{}'))

            # If multiple contradictions exist for the same edge, keep the most recent one
            # (ORDER BY invalidated_at DESC ensures we process most recent first)
            if invalidated_uuid not in result:
                result[invalidated_uuid] = {
                    'invalidated': {
                        'uuid': invalidated_uuid,
                        'fact': record.get('invalidated_fact'),
                        'created_at': invalidated_data.get('created_at'),
                    },
                    'invalidating': {
                        'uuid': record.get('invalidating_edge_uuid'),
                        'fact': record.get('invalidating_fact'),
                        'created_at': invalidating_data.get('created_at'),
                    },
                    'invalidated_at': record.get('invalidated_at'),
                    'contradiction_uuid': record.get('contradiction_uuid'),
                }

        logger.info(
            f'Found {len(result)} contradiction record(s) for {len(invalidated_edge_uuids)} invalidated edge(s)'
        )

        return result

    except Exception as e:
        logger.error(f'Error retrieving invalidating edges: {e}', exc_info=True)
        raise


async def get_all_contradictions_for_group(
    driver: GraphDriver,
    group_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Retrieve all contradictions for a specific group.

    Args:
        driver: The graph driver instance (must be SpannerDriver)
        group_id: The group ID to filter contradictions
        limit: Maximum number of contradictions to retrieve (default: 100)

    Returns:
        List of contradiction records, each containing:
        {
            'contradiction_uuid': str,
            'invalidated': {
                'uuid': str,
                'fact': str,
                'created_at': datetime,  # When the invalidated edge was created
            },
            'invalidating': {
                'uuid': str,
                'fact': str,
                'created_at': datetime,  # When the invalidating edge was created
            },
            'invalidated_at': datetime,
            'created_at': datetime,
        }

    Example:
        >>> contradictions = await get_all_contradictions_for_group(driver, 'group-1')
        >>> for c in contradictions:
        >>>     print(f"{c['invalidated']['fact']} → {c['invalidating']['fact']}")
        >>>     print(f"Old edge created: {c['invalidated']['created_at']}")
        >>>     print(f"New edge created: {c['invalidating']['created_at']}")

    Raises:
        ValueError: If driver is not a SpannerDriver instance
    """
    if not isinstance(driver, SpannerDriver):
        raise ValueError(
            'get_all_contradictions_for_group only works with SpannerDriver. '
            f'Got {type(driver).__name__} instead.'
        )

    logger.info(f'Retrieving up to {limit} contradictions for group_id: {group_id}')

    query = """
    SELECT 
        uuid AS contradiction_uuid,
        invalidated_edge_uuid,
        invalidating_edge_uuid,
        invalidated_fact,
        invalidating_fact,
        invalidated_at,
        created_at,
        invalidated_edge_data,
        invalidating_edge_data
    FROM ContradictedEdge
    WHERE group_id = @group_id
    ORDER BY invalidated_at DESC
    LIMIT @limit
    """

    try:
        records, _, _ = await driver.execute_query(
            query,
            group_id=group_id,
            limit=limit,
        )

        import json

        result = []
        for record in records:
            # Parse JSON data to extract created_at timestamps
            invalidated_data = json.loads(record.get('invalidated_edge_data', '{}'))
            invalidating_data = json.loads(record.get('invalidating_edge_data', '{}'))

            result.append(
                {
                    'contradiction_uuid': record.get('contradiction_uuid'),
                    'invalidated': {
                        'uuid': record.get('invalidated_edge_uuid'),
                        'fact': record.get('invalidated_fact'),
                        'created_at': invalidated_data.get('created_at'),
                    },
                    'invalidating': {
                        'uuid': record.get('invalidating_edge_uuid'),
                        'fact': record.get('invalidating_fact'),
                        'created_at': invalidating_data.get('created_at'),
                    },
                    'invalidated_at': record.get('invalidated_at'),
                    'created_at': record.get('created_at'),
                }
            )

        logger.info(f'Found {len(result)} contradiction record(s) for group_id: {group_id}')

        return result

    except Exception as e:
        logger.error(f'Error retrieving contradictions for group: {e}', exc_info=True)
        raise


async def get_contradiction_details(
    driver: GraphDriver,
    contradiction_uuid: str,
) -> dict[str, Any] | None:
    """
    Retrieve full details of a specific contradiction including complete edge data.

    Args:
        driver: The graph driver instance (must be SpannerDriver)
        contradiction_uuid: UUID of the contradiction record

    Returns:
        Dictionary containing full contradiction details including JSON edge data,
        or None if contradiction not found:
        {
            'contradiction_uuid': str,
            'invalidated': {
                'uuid': str,
                'fact': str,
                'full_data': dict,  # Complete edge data as JSON
            },
            'invalidating': {
                'uuid': str,
                'fact': str,
                'full_data': dict,  # Complete edge data as JSON
            },
            'invalidated_at': datetime,
            'created_at': datetime,
            'group_id': str,
        }

    Example:
        >>> details = await get_contradiction_details(driver, 'contradiction-uuid-1')
        >>> if details:
        >>> # Access complete edge data
        >>>     old_edge_data = details['invalidated']['full_data']
        >>>     print(f"Episodes: {old_edge_data['episodes']}")
        >>>     print(f"Created at: {old_edge_data['created_at']}")

    Raises:
        ValueError: If driver is not a SpannerDriver instance
    """
    if not isinstance(driver, SpannerDriver):
        raise ValueError(
            'get_contradiction_details only works with SpannerDriver. '
            f'Got {type(driver).__name__} instead.'
        )

    logger.info(f'Retrieving contradiction details for UUID: {contradiction_uuid}')

    query = """
    SELECT 
        uuid AS contradiction_uuid,
        invalidated_edge_uuid,
        invalidating_edge_uuid,
        invalidated_fact,
        invalidating_fact,
        invalidated_at,
        created_at,
        group_id,
        invalidated_edge_data,
        invalidating_edge_data
    FROM ContradictedEdge
    WHERE uuid = @contradiction_uuid
    """

    try:
        records, _, _ = await driver.execute_query(
            query,
            contradiction_uuid=contradiction_uuid,
        )

        if not records:
            logger.warning(f'No contradiction found with UUID: {contradiction_uuid}')
            return None

        import json

        record = records[0]

        # Parse JSON data
        invalidated_data = json.loads(record.get('invalidated_edge_data', '{}'))
        invalidating_data = json.loads(record.get('invalidating_edge_data', '{}'))

        result = {
            'contradiction_uuid': record.get('contradiction_uuid'),
            'invalidated': {
                'uuid': record.get('invalidated_edge_uuid'),
                'fact': record.get('invalidated_fact'),
                'full_data': invalidated_data,
            },
            'invalidating': {
                'uuid': record.get('invalidating_edge_uuid'),
                'fact': record.get('invalidating_fact'),
                'full_data': invalidating_data,
            },
            'invalidated_at': record.get('invalidated_at'),
            'created_at': record.get('created_at'),
            'group_id': record.get('group_id'),
        }

        logger.info('Successfully retrieved contradiction details')

        return result

    except Exception as e:
        logger.error(f'Error retrieving contradiction details: {e}', exc_info=True)
        raise
