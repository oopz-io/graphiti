"""
Copyright 2025, Zep Software, Inc.

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
import os
import time
from datetime import datetime, timezone
from logging import INFO
from typing import Any

import functions_framework
from dotenv import load_dotenv
from flask import Request

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.spanner_driver import SpannerDriver
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Spanner connection parameters
spanner_project_id = os.environ.get('SPANNER_PROJECT_ID')
spanner_instance_id = os.environ.get('SPANNER_INSTANCE_ID')
spanner_database_id = os.environ.get('SPANNER_DATABASE_ID')

# API key for Gemini LLM and Embedder
api_key = (
    os.environ.get('GEMINI_API_KEY')
    or os.environ.get('GOOGLE_API_KEY')
    or os.environ.get('GOOGLE_GENAI_API_KEY')
)

if not spanner_project_id or not spanner_instance_id or not spanner_database_id:
    raise ValueError('SPANNER_PROJECT_ID, SPANNER_INSTANCE_ID, and SPANNER_DATABASE_ID must be set')

if not api_key:
    raise ValueError('GEMINI_API_KEY or GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY must be set')


async def process_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Process a list of episodes and return profiling results.

    Args:
        episodes: List of episode dictionaries with 'content', 'type', and 'description' keys

    Returns:
        Dictionary with profiling results and summary
    """
    # Initialize Graphiti with Spanner connection
    # Use await to properly initialize the session pool before any operations
    spanner = await SpannerDriver.create(
        project_id=spanner_project_id,
        instance_id=spanner_instance_id,
        database_id=spanner_database_id,
    )
    graphiti = Graphiti(
        graph_driver=spanner,
        llm_client=GeminiClient(config=LLMConfig(api_key=api_key, model='gemini-2.5-flash')),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(api_key=api_key, embedding_dim=768, embedding_model='gemini-embedding-001')
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(
                api_key=api_key,
                # model="gemini-2.0-flash-exp"
            )
        ),
    )

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        # Add episodes to the graph with profiling
        episode_times = []
        print('\n' + '=' * 80)
        print('PROFILING: add_episode Performance')
        print('=' * 80)

        for i, episode in enumerate(episodes):
            episode_name = f'Freakonomics Radio {i}'
            print(f'\nProcessing episode {i + 1}/{len(episodes)}: {episode_name}')
            print(f'Episode type: {episode["type"].value}')

            # Start timing
            start_time = time.perf_counter()

            await graphiti.add_episode(
                name=episode_name,
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
                profile=True,  # Enable detailed profiling
            )

            # End timing
            elapsed_time = time.perf_counter() - start_time
            episode_times.append(
                {
                    'episode_num': i,
                    'episode_name': episode_name,
                    'episode_type': episode['type'].value,
                    'duration_seconds': elapsed_time,
                }
            )

            print(f'✓ Completed in {elapsed_time:.3f} seconds')

        # Calculate profiling summary
        total_time = sum(e['duration_seconds'] for e in episode_times)
        avg_time = total_time / len(episode_times) if episode_times else 0
        min_time = min(e['duration_seconds'] for e in episode_times) if episode_times else 0
        max_time = max(e['duration_seconds'] for e in episode_times) if episode_times else 0

        # Print profiling summary
        print('\n' + '=' * 80)
        print('PROFILING SUMMARY')
        print('=' * 80)
        print(f'\nTotal episodes processed: {len(episode_times)}')
        print(f'Total time: {total_time:.3f} seconds')
        print(f'Average time per episode: {avg_time:.3f} seconds')
        print(f'Minimum time: {min_time:.3f} seconds')
        print(f'Maximum time: {max_time:.3f} seconds')

        print('\nDetailed breakdown:')
        for ep in episode_times:
            print(
                f'  Episode {ep["episode_num"]}: {ep["duration_seconds"]:.3f}s - '
                f'{ep["episode_name"]} ({ep["episode_type"]})'
            )
        print('=' * 80 + '\n')

        # Return results
        return {
            'success': True,
            'episodes_processed': len(episode_times),
            'total_time_seconds': total_time,
            'average_time_seconds': avg_time,
            'min_time_seconds': min_time,
            'max_time_seconds': max_time,
            'episodes': episode_times,
        }

        """
        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')
        """
    except Exception as e:
        logger.error(f'Error processing episodes: {e}', exc_info=True)
        return {'success': False, 'error': str(e)}
    finally:
        # Close the connection
        await graphiti.close()
        print('\nConnection closed')

async def main():
    """
    Main function for local testing with predefined episodes.
    """
    #################################################
    # ADDING EPISODES
    #################################################
    # Episodes are the primary units of information
    # in Graphiti. They can be text or structured JSON
    # and are automatically processed to extract entities
    # and relationships.
    #################################################

    # Example: Add Episodes
    # Episodes list containing both text and JSON episodes
    episodes = [
        {
            'content': 'Kamala Harris is the Attorney General of California. She was previously '
            'the district attorney for San Francisco.',
            'type': EpisodeType.text,
            'description': 'podcast transcript',
        },
        {
            'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
            'type': EpisodeType.text,
            'description': 'podcast transcript',
        },
        {
            'content': {
                'name': 'Gavin Newsom',
                'position': 'Governor',
                'state': 'California',
                'previous_role': 'Lieutenant Governor',
                'previous_location': 'San Francisco',
            },
            'type': EpisodeType.json,
            'description': 'podcast metadata',
        },
        {
            'content': {
                'name': 'Gavin Newsom',
                'position': 'Governor',
                'term_start': 'January 7, 2019',
                'term_end': 'Present',
            },
            'type': EpisodeType.json,
            'description': 'podcast metadata',
        },
    ]

    result = await process_episodes(episodes)
    print(f'\nResult: {json.dumps(result, indent=2)}')


if __name__ == '__main__':
    asyncio.run(main())
