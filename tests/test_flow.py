import requests
import time
import os

# CONFIG
BASE_URL = "http://127.0.0.1:8000"
TEST_FILE_PATH = "tests/test_doc.txt"

def setup_dummy_file():
    os.makedirs("tests", exist_ok=True)
    with open(TEST_FILE_PATH, "w") as f:
        f.write("AifaTurkey is a leading AI company based in Istanbul. They specialize in Graph RAG systems.")

def test_api_flow():
    print(f"🚀 Starting API Integration Test...")
    
    # 1. Upload
    print(f"Submitting {TEST_FILE_PATH}...")
    with open(TEST_FILE_PATH, "rb") as f:
        files = {'files': (os.path.basename(TEST_FILE_PATH), f, "text/plain")}
        resp = requests.post(f"{BASE_URL}/upload", files=files)
        
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    batch_id = data['batch_id']
    print(f"✅ Uploaded! Batch ID: {batch_id}")

    # 2. Wait for Processing (Simulate Polling / Stream)
    print("⏳ Waiting for worker processing (15s)...")
    time.sleep(15) 
    
    # 3. Query
    query = "Where is AifaTurkey based?"
    print(f"❓ Querying: {query}")
    resp = requests.get(f"{BASE_URL}/query", params={"q": query})
    
    if resp.status_code == 200:
        ans = resp.json()
        print(f"✅ Answer: {ans['answer']}")
        print(f"🔍 Sources: {ans['sources']}")
        print(f"🕸️ Graph Context: {ans.get('graph_context')}")
        
        # LOGICAL ASSERTION
        if "Istanbul" not in ans['answer']:
            raise AssertionError(f"❌ Verification Failed. Expected 'Istanbul' in answer, got: '{ans['answer']}'")
        else:
            print("🎉 LOGICAL CHECK PASSED: Answer contains 'Istanbul'")
            
    else:
        print(f"❌ Query failed: {resp.text}")
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    setup_dummy_file()
    try:
        test_api_flow()
    except Exception as e:
        print(f"❌ Test Crash: {e}")
