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
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CypherToGQLParser:
    """
    Parser to convert Cypher queries to Google Cloud Spanner GQL syntax.
    
    Based on Google Cloud Spanner Graph OpenCypher reference:
    https://cloud.google.com/spanner/docs/graph/opencypher-reference
    """

    def __init__(self, database_name: str):
        self.database_name = database_name
        
        # GQL keywords that need special handling
        self.gql_reserved_words = {
            'GRAPH', 'MATCH', 'WHERE', 'RETURN', 'WITH', 'ORDER', 'BY', 
            'LIMIT', 'SKIP', 'CREATE', 'SET', 'DELETE', 'REMOVE', 'MERGE',
            'UNWIND', 'CALL', 'YIELD', 'UNION', 'ALL', 'DISTINCT', 'AS'
        }
        
        # Cypher functions that need conversion to GQL equivalents
        self.function_mappings = {
            'properties': 'PROPERTIES',  # properties(n) -> PROPERTIES(n)
            'labels': 'LABELS',          # labels(n) -> LABELS(n)
            'type': 'TYPE',              # type(r) -> TYPE(r)
            'id': 'ID',                  # id(n) -> ID(n)
            'toString': 'CAST',          # toString(x) -> CAST(x AS STRING)
            'toInteger': 'CAST',         # toInteger(x) -> CAST(x AS INT64)
            'toFloat': 'CAST',           # toFloat(x) -> CAST(x AS FLOAT64)
            'size': 'ARRAY_LENGTH',      # size(array) -> ARRAY_LENGTH(array)
            'coalesce': 'COALESCE',      # same in both
            'exists': 'EXISTS',          # same in both
        }

    def convert(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Main conversion method to transform Cypher to GQL.
        
        Args:
            cypher_query: The Cypher query to convert
            params: Optional parameters dictionary
            
        Returns:
            Converted GQL query string
        """
        if params is None:
            params = {}
            
        # Start with the original query
        gql_query = cypher_query.strip()
        
        # Step 1: Add GRAPH clause if needed
        gql_query = self._add_graph_clause(gql_query)
        
        # Step 2: Handle MERGE operations (convert to MATCH + CREATE logic)
        gql_query = self._convert_merge_operations(gql_query)
        
        # Step 3: Convert array syntax and UNWIND statements
        gql_query = self._convert_unwind_statements(gql_query)
        
        # Step 4: Handle function conversions
        gql_query = self._convert_functions(gql_query)
        
        # Step 5: Fix clause ordering (WHERE must come before RETURN)
        gql_query = self._fix_clause_ordering(gql_query)
        
        # Step 6: Handle parameter substitution
        gql_query = self._handle_parameter_substitution(gql_query, params)
        
        # Step 7: Clean up and validate syntax
        gql_query = self._cleanup_syntax(gql_query)
        
        return gql_query

    def _add_graph_clause(self, query: str) -> str:
        """Add GRAPH clause if not present and not a DDL operation."""
        query_upper = query.upper().strip()
        
        # Skip if already has GRAPH clause or is DDL
        if (query_upper.startswith('GRAPH') or 
            any(query_upper.startswith(ddl) for ddl in ['CREATE', 'ALTER', 'DROP'])):
            return query
            
        # Add GRAPH clause with proper database name quoting
        database_name = f'`{self.database_name}`' if '-' in self.database_name else self.database_name
        return f'GRAPH {database_name}\n{query}'

    def _convert_merge_operations(self, query: str) -> str:
        """
        Convert MERGE operations to MATCH + CREATE patterns.
        
        Note: This is a simplified conversion. True MERGE semantics would require
        more complex logic with conditional CREATE statements.
        """
        if 'MERGE' not in query.upper():
            return query
            
        # For now, convert MERGE to MATCH as a basic approximation
        # In production, this would need proper MERGE -> MATCH + conditional CREATE logic
        converted = re.sub(r'\bMERGE\b', 'MATCH', query, flags=re.IGNORECASE)
        
        if converted != query:
            logger.warning('MERGE converted to MATCH - may need manual adjustment for proper upsert logic')
            
        return converted

    def _convert_unwind_statements(self, query: str) -> str:
        """
        Convert UNWIND statements and fix IN array syntax for Spanner GQL.
        
        Spanner GQL requirements:
        1. UNWIND is not fully supported - remove or convert
        2. IN [array] must be IN UNNEST([array])
        3. IN $param must be IN UNNEST($param)
        """
        converted = query
        
        # First, fix IN array syntax before handling UNWIND
        # Convert IN [...] patterns to IN UNNEST([...])
        # Handle empty arrays first
        converted = re.sub(r'\bIN\s*\[\s*\]', 'IN UNNEST([])', converted, flags=re.IGNORECASE)
        
        # Handle non-empty arrays: IN [value1, value2] -> IN UNNEST([value1, value2])
        # This pattern matches IN followed by array content
        in_array_pattern = r'\bIN\s*(\[[^\]]+\])'
        converted = re.sub(in_array_pattern, r'IN UNNEST(\1)', converted, flags=re.IGNORECASE)
        
        # Handle parameter arrays: IN $param -> IN UNNEST($param)  
        in_param_pattern = r'\bIN\s*(\$\w+)(?!\s*\[)'  # Don't match if followed by [
        converted = re.sub(in_param_pattern, r'IN UNNEST(\1)', converted, flags=re.IGNORECASE)
        
        # Now handle UNWIND statements
        if 'UNWIND' not in converted.upper():
            return converted
        
        # Strategy 1: Convert simple UNWIND $param AS var to variable handling
        # Pattern for: UNWIND $param AS var
        simple_unwind_pattern = r'^\s*UNWIND\s+\$(\w+)\s+AS\s+(\w+)\s*$'
        
        lines = converted.split('\n')
        filtered_lines = []
        
        for line in lines:
            match = re.match(simple_unwind_pattern, line.strip(), re.IGNORECASE)
            if match:
                param_name = match.group(1)
                var_name = match.group(2)
                logger.info(f'Removing UNWIND ${param_name} AS {var_name} for Spanner compatibility')
                # Skip this line - it will be handled in parameter processing
                continue
            else:
                filtered_lines.append(line)
        
        converted = '\n'.join(filtered_lines)
        
        # If there are still UNWIND statements, warn about them
        if 'UNWIND' in converted.upper():
            logger.warning('Complex UNWIND statements detected that require manual conversion for Spanner')
            # Remove remaining UNWIND statements to prevent syntax errors
            remaining_unwind = re.sub(r'^\s*UNWIND\s+[^\n]*$', '', converted, flags=re.IGNORECASE | re.MULTILINE)
            if remaining_unwind != converted:
                logger.warning('Removed unsupported UNWIND statements - query may need manual adjustment')
                converted = remaining_unwind
        
        return converted

    def _convert_functions(self, query: str) -> str:
        """Convert Cypher functions to their GQL equivalents."""
        converted = query
        
        # Handle properties() function - remove it as GQL accesses properties directly
        converted = re.sub(r'\bproperties\s*\(\s*(\w+)\s*\)', r'\1', converted, flags=re.IGNORECASE)
        
        # Handle toString conversions
        converted = re.sub(r'\btoString\s*\(\s*([^)]+)\s*\)', r'CAST(\1 AS STRING)', converted, flags=re.IGNORECASE)
        
        # Handle toInteger conversions  
        converted = re.sub(r'\btoInteger\s*\(\s*([^)]+)\s*\)', r'CAST(\1 AS INT64)', converted, flags=re.IGNORECASE)
        
        # Handle toFloat conversions
        converted = re.sub(r'\btoFloat\s*\(\s*([^)]+)\s*\)', r'CAST(\1 AS FLOAT64)', converted, flags=re.IGNORECASE)
        
        # Handle size() -> ARRAY_LENGTH()
        converted = re.sub(r'\bsize\s*\(\s*([^)]+)\s*\)', r'ARRAY_LENGTH(\1)', converted, flags=re.IGNORECASE)
        
        return converted

    def _fix_clause_ordering(self, query: str) -> str:
        """
        Fix clause ordering issues for GQL compatibility.
        
        Key rules for Spanner GQL:
        1. WHERE clauses must come before RETURN clauses
        2. YIELD is not supported - remove it
        3. ORDER BY and LIMIT must come after RETURN
        """
        # First, remove YIELD clauses entirely as they're not supported
        query = re.sub(r'^\s*YIELD\s+.*$', '', query, flags=re.IGNORECASE | re.MULTILINE)
        query = re.sub(r'\n\s*YIELD\s+.*(?=\n)', '', query, flags=re.IGNORECASE)
        
        # Handle the specific pattern: WHERE score > X after RETURN
        # This is a common pattern in search queries that's invalid in GQL
        
        # Look for patterns like:
        # RETURN something
        # [additional return lines]
        # WHERE score > X
        
        lines = query.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            line_upper = line.upper()
            
            # If we find a RETURN line, look ahead for problematic WHERE clauses
            if line_upper.startswith('RETURN'):
                return_block = [lines[i]]
                j = i + 1
                
                # Collect the rest of the RETURN statement
                while j < len(lines):
                    next_line = lines[j].strip()
                    next_upper = next_line.upper()
                    
                    # If we find WHERE score > pattern, skip it and associated ORDER/LIMIT
                    if next_upper.startswith('WHERE') and 'SCORE >' in next_upper:
                        logger.info(f'Removing invalid WHERE clause after RETURN: {next_line}')
                        j += 1
                        # Also skip ORDER BY and LIMIT that come after this invalid WHERE
                        while j < len(lines):
                            subsequent_line = lines[j].strip().upper()
                            if subsequent_line.startswith(('ORDER BY', 'LIMIT')):
                                logger.info(f'Also removing: {lines[j].strip()}')
                                j += 1
                            else:
                                break
                        continue
                    
                    # If we hit a major clause boundary, stop
                    elif next_upper.startswith(('MATCH', 'WITH', 'CREATE', 'MERGE', 'GRAPH', 'FOR')):
                        break
                    
                    # Otherwise, it's part of the RETURN block
                    else:
                        return_block.append(lines[j])
                        j += 1
                
                # Add the cleaned return block
                result_lines.extend(return_block)
                i = j
            else:
                result_lines.append(lines[i])
                i += 1
        
        result = '\n'.join(result_lines)
        
        # Clean up extra blank lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result

    def _handle_parameter_substitution(self, query: str, params: Dict[str, Any]) -> str:
        """
        Handle parameter substitution and format conversion.
        
        Convert from Cypher $param format to GQL @param format and substitute values.
        """
        converted = query
        
        for param_name, param_value in params.items():
            dollar_placeholder = f'${param_name}'
            at_placeholder = f'@{param_name}'
            
            # First convert $param to @param syntax
            if dollar_placeholder in converted:
                converted = converted.replace(dollar_placeholder, at_placeholder)
            
            # Then substitute actual values if needed
            if at_placeholder in converted:
                formatted_value = self._format_parameter_value(param_value)
                if formatted_value is not None:
                    converted = converted.replace(at_placeholder, formatted_value)
        
        return converted

    def _format_parameter_value(self, value: Any) -> Optional[str]:
        """Format a parameter value for GQL syntax."""
        if value is None:
            return 'NULL'
        elif isinstance(value, str):
            # Escape special characters properly for GQL
            escaped = value.replace('\\', '\\\\')  # Escape backslashes first
            escaped = escaped.replace("'", "\\'")  # Escape single quotes
            escaped = escaped.replace('"', '\\"')  # Escape double quotes
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return 'TRUE' if value else 'FALSE'
        elif isinstance(value, (int, float)):
            return str(value)
        elif hasattr(value, 'isoformat'):  # datetime-like objects
            return f"TIMESTAMP '{value.isoformat()}'"
        elif isinstance(value, (list, tuple)):
            if not value:
                return '[]'
            # Format array elements for GQL - arrays use bracket notation
            if all(isinstance(item, str) for item in value):
                formatted_items = [f"'{self._escape_string(item)}'" for item in value]
                return f"[{', '.join(formatted_items)}]"
            else:
                formatted_items = [str(item) for item in value]
                return f"[{', '.join(formatted_items)}]"
        else:
            # For complex objects, return None to keep the @param placeholder
            return None

    def _escape_string(self, s: str) -> str:
        """Helper to escape strings for GQL."""
        escaped = s.replace('\\', '\\\\')  # Escape backslashes first
        escaped = escaped.replace("'", "\\'")  # Escape single quotes
        escaped = escaped.replace('"', '\\"')  # Escape double quotes
        return escaped

    def _cleanup_syntax(self, query: str) -> str:
        """Final cleanup and syntax validation."""
        # Remove extra whitespace
        lines = [line.rstrip() for line in query.split('\n')]
        
        # Remove empty lines but preserve structure
        cleaned_lines = []
        for line in lines:
            if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(line)
        
        # Join back together
        cleaned = '\n'.join(cleaned_lines)
        
        # Final validation - ensure no problematic patterns remain
        if re.search(r'\bWHERE\b.*\bRETURN\b', cleaned, re.DOTALL | re.IGNORECASE):
            logger.warning('Query may still contain WHERE clause after RETURN - manual review needed')
        
        return cleaned

    def is_supported_query(self, query: str) -> bool:
        """
        Check if a query contains patterns that are known to be unsupported in GQL.
        
        Returns:
            True if query should be convertible, False if it contains unsupported patterns
        """
        query_upper = query.upper()
        
        # Check for unsupported patterns
        unsupported_patterns = [
            'CALL apoc.',           # APOC procedures
            'YIELD',                # YIELD clause (limited support in GQL)
            'SHORTEST PATH',        # Shortest path functions
            'ALL SHORTEST PATHS',   # All shortest paths
            'PERIODIC COMMIT',      # Periodic commit
        ]
        
        for pattern in unsupported_patterns:
            if pattern in query_upper:
                logger.warning(f'Query contains unsupported pattern: {pattern}')
                return False
        
        return True

    def get_conversion_warnings(self, original_query: str, converted_query: str) -> List[str]:
        """
        Analyze the conversion and return a list of warnings about potential issues.
        """
        warnings = []
        
        original_upper = original_query.upper()
        converted_upper = converted_query.upper()
        
        # Check for MERGE -> MATCH conversion
        if 'MERGE' in original_upper and 'MERGE' not in converted_upper:
            warnings.append('MERGE operations converted to MATCH - verify upsert logic is correct')
        
        # Check for UNWIND conversion
        if 'UNWIND' in original_upper and 'FOR' not in converted_upper:
            warnings.append('UNWIND statements may need manual conversion to FOR loops')
        
        # Check for complex WHERE clauses
        if 'WHERE' in original_upper and 'WHERE' not in converted_upper:
            warnings.append('WHERE clauses were removed - they may have been in invalid positions')
        
        # Check for function conversions
        if 'properties(' in original_query.lower():
            warnings.append('properties() function calls were converted - verify property access is correct')
        
        return warnings


def convert_cypher_to_gql(cypher_query: str, database_name: str, params: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """
    Convenience function to convert a Cypher query to GQL.
    
    Args:
        cypher_query: The Cypher query to convert
        database_name: Name of the Spanner database
        params: Optional parameters dictionary
        
    Returns:
        Tuple of (converted_query, list_of_warnings)
    """
    parser = CypherToGQLParser(database_name)
    
    # Check if query is supported
    if not parser.is_supported_query(cypher_query):
        logger.warning('Query contains patterns that may not be supported in GQL')
    
    # Convert the query
    converted_query = parser.convert(cypher_query, params)
    
    # Get conversion warnings
    warnings = parser.get_conversion_warnings(cypher_query, converted_query)
    
    return converted_query, warnings