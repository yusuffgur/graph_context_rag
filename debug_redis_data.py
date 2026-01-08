import redis

import redis

r = redis.Redis(host='localhost', port=6380, decode_responses=True)

print("--- Checking Graph Existence ---")
keys = r.keys("*")
print(f"Keys in Redis: {keys}")

GRAPH_KEY = "federated_mem"

print(f"\n--- Checking Graph '{GRAPH_KEY}' Stats ---")
try:
    # Match all nodes limit 10
    q = "MATCH (n) RETURN n LIMIT 10"
    res = r.execute_command("GRAPH.QUERY", GRAPH_KEY, q)
    print(f"Sample Nodes raw: {res}")
    
    # Match all edges limit 10
    q_edge = "MATCH ()-[r]->() RETURN type(r) LIMIT 10"
    res_edge = r.execute_command("GRAPH.QUERY", GRAPH_KEY, q_edge)
    print(f"Sample Edges raw: {res_edge}")
    
except Exception as e:
    print(f"Error querying graph: {e}")
