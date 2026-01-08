import asyncio
import logging
from src.config import settings
from src.modules.graph import FalkorGraph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugGraph")

def inspect_graph():
    print(f"\n🔎 Inspecting Graph Database at {settings.FALKOR_URL}\n")
    
    graph_db = FalkorGraph(settings.FALKOR_URL)
    
    # 0. List Graphs
    try:
        graphs = graph_db.r.execute_command("GRAPH.LIST")
        print(f"📊 Available Graphs: {graphs}")
    except Exception as e:
        print(f"❌ Failed to list graphs: {e}")

    # 1. Count Nodes
    try:
        # Cypher to count all nodes
        res = graph_db.execute_cypher("MATCH (n) RETURN count(n) as count")
        # Result format: [[count]] (nested list from RedisGraph usually)
        # But execute_cypher returns the raw response. 
        # RedisGraph-py output structure: [header, result_set, stats] or similar.
        # Wait, execute_cypher calls self.r.execute_command("GRAPH.QUERY", ...)
        # The result set is usually index 0 or 1 depending on version.
        
        # Let's just print raw res to be safe first
        # print(f"Raw Res: {res}") 
        
        # Standard RedisGraph response: [ [header], [ [row1], [row2] ], [stats] ]
        # If just count: [ [['count']], [[123]], ... ]
        
        # Assuming typical response structure
        if res and len(res) > 0:
            # Stats are usually last. Result set is usually index 1 if header exists.
            # But let's just try to parse safe.
            print(f"📊 Raw Result: {res}")
    except Exception as e:
        print(f"❌ Failed to count nodes: {e}")
        return

    # 2. Sample Nodes
    print("\n📦 Sample Entities (Top 10):")
    res = graph_db.execute_cypher("MATCH (n) RETURN n.name, labels(n) LIMIT 10")
    # Result rows are typically in the second element of the response list
    if res and len(res) > 1:
        rows = res[1] 
        for row in rows:
            print(f"   - {row}")

    # 3. Sample Relationships
    print("\n🔗 Sample Relationships (Top 10):")
    res = graph_db.execute_cypher("MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 10")
    if res and len(res) > 1:
        rows = res[1]
        for row in rows:
            print(f"   - {row}")

if __name__ == "__main__":
    inspect_graph()
