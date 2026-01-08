from redis import Redis
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class FalkorGraph:
    def __init__(self, url):
        self.r = Redis.from_url(url, decode_responses=True)
        self.graph = "federated_mem"

    def execute_cypher(self, query):
        """Raw execution wrapper for CRUD operations."""
        try:
            res = self.r.execute_command("GRAPH.QUERY", self.graph, query)
            # logger.info(f"CYPHER RAW: {res}") # Uncomment for verbose debug
            return res
        except Exception as e:
            logger.error(f"Cypher Error: {e} | Query: {query}")
            return []

    def insert_triple(self, s, r, o):
        """Create or Update relationship."""
        # Sanitize
        r_safe = "".join([c for c in r if c.isalnum() or c == '_']).upper()
        s_safe = s.replace("'", "\\'")
        o_safe = o.replace("'", "\\'")
        
        q = f"""
        MERGE (a:Entity {{name: '{s_safe}'}}) 
        MERGE (b:Entity {{name: '{o_safe}'}}) 
        MERGE (a)-[:{r_safe}]->(b)
        """
        self.execute_cypher(q)

    def insert_chunk_link(self, chunk_id, entities, source=None):
        """Link a Chunk node to its contained Entities."""
        # 1. Create Chunk Node (if not exists)
        # Note: We don't store full text in graph to save RAM, just ID and metadata if needed
        # Escape source path just in case
        safe_source = source.replace("'", "\\'") if source else "Unknown"
        q_chunk = f"MERGE (c:Chunk {{id: '{chunk_id}', source: '{safe_source}'}})"
        self.execute_cypher(q_chunk)
        
        # 2. Link Entities
        for entity in entities:
             clean_e = entity.replace("'", "\\'")
             # Connect Chunk -> Entity
             q = f"""
             MATCH (c:Chunk {{id: '{chunk_id}'}})
             MERGE (e:Entity {{name: '{clean_e}'}})
             MERGE (c)-[:MENTIONS]->(e)
             """
             self.execute_cypher(q)

    def query_neighbors(self, entity_name):
        """Read: Find 1-hop neighbors (Case-Insensitive)."""
        safe_name = entity_name.replace("'", "\\'")
        # Search via Case-Insensitive partial match for better recall
        q = f"MATCH (n)-[r]->(m) WHERE toLower(n.name) CONTAINS toLower('{safe_name}') RETURN type(r), m.name LIMIT 15"
        res = self.execute_cypher(q)
        logger.info(f"DEBUG QUERY_NEIGHBORS for {entity_name}: {res}")
        # Parse standard [Header, Data, Stats] format
        if res and len(res) >= 2 and isinstance(res[1], list):
            return res[1] # Return only the data rows [['REL', 'Target'], ...]
        return []

    def get_chunks_for_entity(self, entity_name, file_filter=None):
        """Read: Find chunks that mention this entity (Case-Insensitive), optionally filtered by source."""
        safe_name = entity_name.replace("'", "\\'")
        
        # Base Query
        # (Chunk)-[:MENTIONS]->(Entity)
        # Case-Insensitive matching
        where_clause = f"WHERE toLower(e.name) CONTAINS toLower('{safe_name}')"
        
        # Add Filter if provided
        if file_filter:
             safe_file = file_filter.replace("'", "\\'")
             where_clause += f" AND c.source = '{safe_file}'"
             
        q = f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) {where_clause} RETURN c.id"
        res = self.execute_cypher(q)
        
        # RedisGraph GRAPH.QUERY returns: [Header, DataRows, Stats]
        # Example: [['c.id'], [['chunk-123'], ['chunk-456']], ['Query time...']]
        
        chunk_ids = []
        if res and len(res) >= 2:
            data_rows = res[1] # This is the list of records
            if isinstance(data_rows, list):
                for row in data_rows:
                    if row and len(row) > 0:
                        chunk_ids.append(row[0])
        
        return chunk_ids

    def reset_graph(self):
        """Hard Delete the entire graph key."""
        try:
            self.r.execute_command("GRAPH.DELETE", self.graph)
        except Exception as e:
            # It throws an error if the graph doesn't exist, which is fine
            logger.info(f"Graph Delete Info: {e}")