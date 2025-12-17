"""
Query Profiler for Graphiti Database Operations

This module provides detailed profiling of database queries during add_episode
and other Graphiti operations. It helps identify performance bottlenecks.

Usage:
    from graphiti_core.utils.query_profiler import (
        enable_query_profiling,
        disable_query_profiling,
        get_query_profile,
        print_query_profile_summary,
        reset_query_profile,
    )
    
    # Enable profiling before running add_episode
    enable_query_profiling()
    
    # Run your operations...
    await graphiti.add_episode(...)
    
    # Print results
    print_query_profile_summary()
    
    # Get programmatic access to results
    profile_data = get_query_profile()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-safe profiling state
_profiling_lock = threading.Lock()
_profiling_enabled = False
_query_records: List[Dict[str, Any]] = []


@dataclass
class QueryRecord:
    """Record of a single query execution."""
    operation: str  # e.g., 'retrieve_episodes', 'node_fulltext_search'
    query_type: str  # 'SELECT', 'INSERT', 'SEARCH', 'GRAPH'
    table_or_index: str  # e.g., 'EpisodicNode', 'EntityNode'
    duration_ms: float
    rows_returned: int
    query_preview: str  # First 200 chars of query
    params_preview: Optional[str]  # Summary of params
    timestamp: float
    caller: str  # Function name that initiated the query


def enable_query_profiling():
    """Enable query profiling."""
    global _profiling_enabled
    with _profiling_lock:
        _profiling_enabled = True
        _query_records.clear()
    logger.info("[QUERY_PROFILER] Profiling enabled")


def disable_query_profiling():
    """Disable query profiling."""
    global _profiling_enabled
    with _profiling_lock:
        _profiling_enabled = False
    logger.info("[QUERY_PROFILER] Profiling disabled")


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _profiling_enabled


def reset_query_profile():
    """Clear all profiling data."""
    global _query_records
    with _profiling_lock:
        _query_records.clear()


def record_query(
    operation: str,
    query_type: str,
    table_or_index: str,
    duration_ms: float,
    rows_returned: int = 0,
    query_text: str = "",
    params: Optional[Dict] = None,
    caller: str = "",
):
    """Record a query execution for profiling."""
    if not _profiling_enabled:
        return
    
    with _profiling_lock:
        _query_records.append({
            'operation': operation,
            'query_type': query_type,
            'table_or_index': table_or_index,
            'duration_ms': duration_ms,
            'rows_returned': rows_returned,
            'query_full': query_text,  # Store full query text
            'query_preview': query_text[:200] if query_text else "",  # Keep preview for quick display
            'params_preview': str(params)[:100] if params else None,
            'timestamp': time.time(),
            'caller': caller,
        })


def get_query_profile() -> Dict[str, Any]:
    """Get the current query profile data."""
    with _profiling_lock:
        return {
            'enabled': _profiling_enabled,
            'query_count': len(_query_records),
            'queries': list(_query_records),
        }


def classify_query(query_text: str) -> tuple[str, str, str]:
    """
    Classify a query by operation, type, and table/index.
    
    Returns:
        Tuple of (operation_name, query_type, table_or_index)
    """
    query_upper = query_text.upper().strip()
    
    # Determine query type
    if 'INSERT' in query_upper:
        query_type = 'INSERT'
    elif 'UPDATE' in query_upper:
        query_type = 'UPDATE'
    elif 'DELETE' in query_upper:
        query_type = 'DELETE'
    elif 'MERGE' in query_upper:
        query_type = 'MERGE'
    elif 'SEARCH(' in query_upper:
        query_type = 'SEARCH'
    elif 'COSINE_DISTANCE' in query_upper:
        query_type = 'VECTOR_SEARCH'
    elif query_upper.startswith('GRAPH'):
        query_type = 'GRAPH_QUERY'
    elif query_upper.startswith('SELECT') or query_upper.startswith('WITH'):
        query_type = 'SELECT'
    else:
        query_type = 'OTHER'
    
    # Determine table/index
    table = 'unknown'
    if 'EPISODICNODE' in query_upper or 'EPISODIC' in query_upper:
        table = 'EpisodicNode'
    elif 'ENTITYEDGE' in query_upper or 'RELATES_TO' in query_upper:
        table = 'EntityEdge'
    elif 'ENTITYNODE' in query_upper or '(N:ENTITY)' in query_upper or ':ENTITY' in query_upper:
        table = 'EntityNode'
    elif 'COMMUNITYNODE' in query_upper or 'COMMUNITY' in query_upper:
        table = 'CommunityNode'
    elif 'EPISODICEDGE' in query_upper or 'MENTIONS' in query_upper:
        table = 'EpisodicEdge'
    
    # Determine operation name based on query pattern
    operation = 'unknown'
    
    # retrieve_episodes pattern
    if 'MATCH (E:EPISODIC)' in query_upper and 'VALID_AT' in query_upper:
        operation = 'retrieve_episodes'
    
    # Fulltext search patterns
    elif 'SEARCH(' in query_upper:
        if 'ENTITYEDGE' in query_upper or 'NAME_TOKENS' in query_upper and 'FACT_TOKENS' in query_upper:
            operation = 'edge_fulltext_search'
        elif 'ENTITYNODE' in query_upper:
            operation = 'node_fulltext_search'
        elif 'EPISODICNODE' in query_upper:
            operation = 'episode_fulltext_search'
        else:
            operation = 'fulltext_search'
    
    # Vector similarity search patterns
    elif 'COSINE_DISTANCE' in query_upper:
        if 'FACT_EMBEDDING' in query_upper:
            operation = 'edge_similarity_search'
        elif 'NAME_EMBEDDING' in query_upper:
            operation = 'node_similarity_search'
        else:
            operation = 'similarity_search'
    
    # Edge queries
    elif 'RELATES_TO' in query_upper:
        if 'UUID:' in query_upper or 'UUID =' in query_upper:
            operation = 'get_edges_between_nodes'
        elif 'UUID IN' in query_upper:
            operation = 'get_edges_by_uuids'
        else:
            operation = 'edge_query'
    
    # Node queries
    elif ':ENTITY' in query_upper and query_type == 'GRAPH_QUERY':
        operation = 'node_query'
    
    # Batch operations
    elif query_type in ('INSERT', 'UPDATE', 'MERGE'):
        if 'EPISODICNODE' in query_upper:
            operation = 'insert_episodic_node'
        elif 'ENTITYNODE' in query_upper:
            operation = 'insert_entity_node'
        elif 'ENTITYEDGE' in query_upper:
            operation = 'insert_entity_edge'
        elif 'EPISODICEDGE' in query_upper:
            operation = 'insert_episodic_edge'
        else:
            operation = 'insert_operation'
    
    return operation, query_type, table


def print_query_profile_summary() -> Dict[str, Any]:
    """Print a formatted summary of query profiling data."""
    with _profiling_lock:
        queries = list(_query_records)
    
    if not queries:
        print("\n" + "=" * 90)
        print("QUERY PROFILING SUMMARY - NO QUERIES RECORDED")
        print("=" * 90)
        print("Make sure profiling is enabled before running operations:")
        print("  from graphiti_core.utils.query_profiler import enable_query_profiling")
        print("  enable_query_profiling()")
        print("=" * 90)
        return {'total_queries': 0, 'total_time_ms': 0}
    
    # Group by operation
    by_operation: Dict[str, List[Dict]] = {}
    for q in queries:
        op = q['operation']
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(q)
    
    # Group by query type
    by_type: Dict[str, List[Dict]] = {}
    for q in queries:
        qt = q['query_type']
        if qt not in by_type:
            by_type[qt] = []
        by_type[qt].append(q)
    
    # Group by table
    by_table: Dict[str, List[Dict]] = {}
    for q in queries:
        tbl = q['table_or_index']
        if tbl not in by_table:
            by_table[tbl] = []
        by_table[tbl].append(q)
    
    total_time = sum(q['duration_ms'] for q in queries)
    total_rows = sum(q['rows_returned'] for q in queries)
    
    print("\n" + "=" * 90)
    print("QUERY PROFILING SUMMARY FOR add_episode")
    print("=" * 90)
    print(f"Total queries executed: {len(queries)}")
    print(f"Total query time: {total_time:.2f}ms ({total_time/1000:.2f}s)")
    print(f"Total rows processed: {total_rows}")
    print("=" * 90)
    
    # By Operation (sorted by total time)
    print("\n📊 BY OPERATION (sorted by total time):")
    print("-" * 90)
    print(f"{'Operation':<40} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Rows':>8} {'%':>6}")
    print("-" * 90)
    
    sorted_ops = sorted(
        by_operation.items(),
        key=lambda x: sum(q['duration_ms'] for q in x[1]),
        reverse=True
    )
    
    for op, op_queries in sorted_ops:
        count = len(op_queries)
        total = sum(q['duration_ms'] for q in op_queries)
        avg = total / count
        rows = sum(q['rows_returned'] for q in op_queries)
        pct = (total / total_time) * 100 if total_time > 0 else 0
        
        # Mark high-time operations
        marker = "🔴" if pct > 20 else ("🟡" if pct > 10 else "")
        print(f"{marker}{op:<39} {count:>8} {total:>12.2f} {avg:>10.2f} {rows:>8} {pct:>5.1f}%")
    
    print("-" * 90)
    
    # By Query Type
    print("\n📊 BY QUERY TYPE:")
    print("-" * 60)
    print(f"{'Query Type':<20} {'Count':>8} {'Total(ms)':>12} {'%':>6}")
    print("-" * 60)
    
    sorted_types = sorted(
        by_type.items(),
        key=lambda x: sum(q['duration_ms'] for q in x[1]),
        reverse=True
    )
    
    for qt, qt_queries in sorted_types:
        count = len(qt_queries)
        total = sum(q['duration_ms'] for q in qt_queries)
        pct = (total / total_time) * 100 if total_time > 0 else 0
        print(f"{qt:<20} {count:>8} {total:>12.2f} {pct:>5.1f}%")
    
    print("-" * 60)
    
    # By Table
    print("\n📊 BY TABLE/INDEX:")
    print("-" * 60)
    print(f"{'Table/Index':<20} {'Count':>8} {'Total(ms)':>12} {'%':>6}")
    print("-" * 60)
    
    sorted_tables = sorted(
        by_table.items(),
        key=lambda x: sum(q['duration_ms'] for q in x[1]),
        reverse=True
    )
    
    for tbl, tbl_queries in sorted_tables:
        count = len(tbl_queries)
        total = sum(q['duration_ms'] for q in tbl_queries)
        pct = (total / total_time) * 100 if total_time > 0 else 0
        print(f"{tbl:<20} {count:>8} {total:>12.2f} {pct:>5.1f}%")
    
    print("-" * 60)
    
    # Top 10 Slowest Queries
    print("\n🐢 TOP 10 SLOWEST INDIVIDUAL QUERIES:")
    print("-" * 90)
    
    sorted_queries = sorted(queries, key=lambda x: x['duration_ms'], reverse=True)[:10]
    for i, q in enumerate(sorted_queries, 1):
        print(f"\n{i}. {q['operation']} ({q['query_type']}) - {q['duration_ms']:.2f}ms")
        print(f"   Table: {q['table_or_index']}, Rows: {q['rows_returned']}")
        # Print full query, nicely formatted
        full_query = q.get('query_full', q.get('query_preview', ''))
        if full_query:
            # Clean up whitespace for readability
            clean_query = ' '.join(full_query.split())
            print(f"   Query:")
            # Wrap long queries at 100 chars for readability
            for j in range(0, len(clean_query), 100):
                prefix = "      " if j > 0 else "      "
                print(f"{prefix}{clean_query[j:j+100]}")
    
    print("\n" + "=" * 90)
    
    # Return summary for programmatic use
    return {
        'total_queries': len(queries),
        'total_time_ms': total_time,
        'total_rows': total_rows,
        'by_operation': {
            op: {
                'count': len(qs),
                'total_ms': sum(q['duration_ms'] for q in qs),
                'avg_ms': sum(q['duration_ms'] for q in qs) / len(qs),
                'rows': sum(q['rows_returned'] for q in qs),
            }
            for op, qs in by_operation.items()
        },
        'by_type': {
            qt: {
                'count': len(qs),
                'total_ms': sum(q['duration_ms'] for q in qs),
            }
            for qt, qs in by_type.items()
        },
        'by_table': {
            tbl: {
                'count': len(qs),
                'total_ms': sum(q['duration_ms'] for q in qs),
            }
            for tbl, qs in by_table.items()
        },
        'slowest_queries': [
            {
                'operation': q['operation'],
                'duration_ms': q['duration_ms'],
                'table': q['table_or_index'],
                'rows': q['rows_returned'],
            }
            for q in sorted_queries
        ],
    }
