from src.config import settings
from src.modules.graph import FalkorGraph
import time

def test_insert():
    print(f"🔬 Testing Graph Insertion at {settings.FALKOR_URL}...")
    g = FalkorGraph(settings.FALKOR_URL)
    
    # 1. Insert Test Triple
    print("👉 Inserting (TestSubject)-[TEST_RELATION]->(TestObject)")
    g.insert_triple("TestSubject", "TEST_RELATION", "TestObject")
    
    # 2. Wait a moment (persistence is usually instant but let's be safe)
    time.sleep(1)
    
    # 3. Query it back
    print("👀 Querying for 'TestSubject'...")
    res = g.execute_cypher("MATCH (n:Entity {name: 'TestSubject'}) RETURN n.name, labels(n)")
    print(f"📄 Result: {res}")
    
    # Validate
    if res and len(res) > 1 and len(res[1]) > 0:
        print("✅ SUCCESS: Found the inserted node!")
    else:
        print("❌ FAILURE: Node not found.")
        # Check raw list
        print("👉 Listing all nodes:")
        all_nodes = g.execute_cypher("MATCH (n) RETURN n")
        print(f"All Nodes: {all_nodes}")

if __name__ == "__main__":
    test_insert()
