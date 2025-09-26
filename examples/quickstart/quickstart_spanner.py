#!/usr/bin/env python3
"""
Quickstart example for using Graphiti with Google Cloud Spanner Graph database.

This example demonstrates how to:
1. Connect to Google Cloud Spanner Graph database
2. Initialize Graphiti indices and constraints
3. Add episodes to build the knowledge graph
4. Search the graph using Spanner's native fulltext search
5. Explore graph-based search with native vector similarity using Spanner's built-in functions

Prerequisites:
- Google Cloud Spanner instance with Graph database enabled
- Google Cloud authentication set up
- Required Python packages installed

Note: This example uses Spanner's native vector search capabilities (COSINE_DISTANCE, DOT_PRODUCT,
EUCLIDEAN_DISTANCE) instead of external search services like OpenSearch.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from enum import Enum

from graphiti_core.driver.spanner_driver import SpannerDriver
from graphiti_core.graphiti import Graphiti


class SourceType(Enum):
    PODCAST = 'podcast'
    NEWS_ARTICLE = 'news_article'
    RESEARCH_PAPER = 'research_paper'


# Spanner configuration from environment variables or defaults
spanner_project_id = os.getenv('SPANNER_PROJECT_ID', 'your-gcp-project')
spanner_instance_id = os.getenv('SPANNER_INSTANCE_ID', 'graphiti-instance')
spanner_database_id = os.getenv('SPANNER_DATABASE_ID', 'graphiti-db')
aws_region = os.getenv('AWS_REGION', 'us-east-1')
aws_service = os.getenv('AWS_SERVICE', 'es')


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Google Cloud Spanner and set up Graphiti indices
    # This is required before using other Graphiti functionality
    #################################################

    # Initialize Graphiti with Spanner driver (now with native vector search)
    spanner_driver = SpannerDriver(
        spanner_project_id=spanner_project_id,
        spanner_instance_id=spanner_instance_id,
        spanner_database_id=spanner_database_id,
    )

    graphiti = Graphiti(graph_driver=spanner_driver)

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

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
                'type': SourceType.NEWS_ARTICLE,
                'description': 'News article about Kamala Harris career',
            },
            {
                'content': 'We are in the hottest year on record. There were many record-breaking temperatures. '
                'Climate change is a major factor in this.',
                'type': SourceType.RESEARCH_PAPER,
                'description': 'Research findings on climate change',
            },
            {
                'content': {
                    'researcher': 'Dr. Sarah Johnson',
                    'institution': 'Stanford University',
                    'finding': 'Machine learning models can predict climate patterns with 85% accuracy',
                    'publication_date': '2024-01-15',
                },
                'type': SourceType.RESEARCH_PAPER,
                'description': 'Research paper on ML climate prediction',
            },
            {
                'content': 'Google announced a new quantum computing breakthrough. The quantum computer '
                'solved a complex problem in minutes that would take classical computers years.',
                'type': SourceType.NEWS_ARTICLE,
                'description': 'Tech news about quantum computing',
            },
            {
                'content': 'OpenAI released GPT-4, which shows significant improvements in reasoning '
                'and multimodal capabilities compared to previous versions.',
                'type': SourceType.PODCAST,
                'description': 'Podcast discussion about AI developments',
            },
        ]

        # Add episodes to the graph
        print('Adding episodes to the graph...')
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Episode {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'].value,
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Episode {i} ({episode["type"].value})')

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
        # ADDITIONAL SEARCH EXAMPLES
        #################################################
        # Try different search queries to explore the graph
        #################################################

        print("\nSearching for: 'climate change research'")
        climate_results = await graphiti.search('climate change research')
        print(f'Found {len(climate_results)} results about climate research')

        print("\nSearching for: 'artificial intelligence developments'")
        ai_results = await graphiti.search('artificial intelligence developments')
        print(f'Found {len(ai_results)} results about AI developments')

        print('\n✅ Graphiti Spanner quickstart completed successfully!')
        print('You can now explore more search queries or add additional episodes.')

    except Exception as e:
        print(f'❌ Error during quickstart: {e}')
        raise
    finally:
        # Clean up resources
        await graphiti.close()


if __name__ == '__main__':
    print('🚀 Starting Graphiti Spanner Quickstart...')
    print(
        f'Connecting to Spanner: {spanner_project_id}/{spanner_instance_id}/{spanner_database_id}'
    )

    # Check if required environment variables are set
    if spanner_project_id == 'your-gcp-project':
        print(
            '⚠️  Please set SPANNER_PROJECT_ID environment variable to your Google Cloud Project ID'
        )

    asyncio.run(main())
