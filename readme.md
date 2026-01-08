# Enterprise Federated Contextual & Graph RAG

A production-grade Federated Memory system that combines **Contextual RAG** (situating chunks with global summaries) and **Graph RAG** (extracting knowledge graphs) to enable complex reasoning over massive document sets.

## 🚀 Features

* **Hybrid Search:** Combines Vector Search (Qdrant) and Graph Search (FalkorDB).
* **Contextual RAG:** Automatically generates "Contextual Headers" for every text chunk to improve retrieval precision.
* **Resilience:** Full retry logic (Tenacity) for LLMs and DB connections; Dead Letter Queues for failed jobs.
* **Real-Time Feedback:** Server-Sent Events (SSE) via Redis Pub/Sub for live progress tracking.
* **Data Sovereignty:** Uses Local LLMs (Ollama) for high-volume processing and Cloud LLMs (GPT-4) only for final synthesis.

---

## 🛠️ Prerequisites

* **Docker & Docker Compose** (for infrastructure)
* **Python 3.10+**
* **Git**

---

## 📥 Installation

1.  **Clone & Enter Directory**
    ```bash
    mkdir federated_rag_system
    cd federated_rag_system
    # (Copy all provided code files into this folder structure)
    ```

2.  **Set Up Environment Variables**
    Create a `.env` file:
    ```ini
    OPENAI_API_KEY=sk-proj-your-key-here
    OLLAMA_URL=http://localhost:11434
    QDRANT_URL=http://localhost:6333
    FALKOR_URL=redis://localhost:6380
    REDIS_URL=redis://localhost:6379
    KAFKA_BOOTSTRAP=localhost:9092
    LOG_LEVEL=INFO
    ```

3.  **Start Infrastructure**
    This spins up Qdrant, FalkorDB, Redis, Redpanda (Kafka), and Ollama.
    ```bash
    docker-compose up -d
    ```

4.  **Initialize Models**
    Wait 30 seconds for containers to stabilize, then run the setup script to download the local LLM (Mistral).
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

5.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🧪 Usage Guide

### 1. Upload Documents (Batch Ingestion)
To start processing, upload a batch of PDF or TXT files. The system offloads processing to a background queue instantly.

* **Endpoint:** `POST /upload`
* **Body:** `multipart/form-data` with key `files` (Select multiple files)
* **Response:**
    ```json
    {
      "status": "Queued",
      "batch": "550e8400-e29b-41d4-a716-446655440000",
      "count": 5
    }
    ```

### 2. Monitor Progress (Real-Time Streams)
Instead of polling, subscribe to the Server-Sent Events (SSE) endpoint to watch the worker process files live.

* **Endpoint:** `GET /stream/{batch_id}`
* **Client Example (Browser Console):**
    ```javascript
    const evtSource = new EventSource("http://localhost:8000/stream/<batch_id>");
    evtSource.onmessage = function(event) {
        console.log("Update:", JSON.parse(event.data));
    }
    ```
* **Event Data:**
    ```json
    {
      "status": "PROCESSING",
      "file": "contract_v1.pdf",
      "progress": "Extracting Knowledge Graph..."
    }
    ```

### 3. Query the Memory (Hybrid Search)
Perform a federated search that queries both the Vector DB (for semantic context) and the Graph DB (for relationship hops).

* **Endpoint:** `GET /query?q=Your Question Here`
* **Process:**
    1.  **Vector Search:** Finds top 5 relevant text chunks from Qdrant.
    2.  **Entity Extraction:** Uses Local LLM to identify the core entity in your question.
    3.  **Graph Traversal:** Queries FalkorDB for 1-hop neighbors of that entity.
    4.  **Synthesis:** Cloud LLM (GPT-4o) generates the final answer using both contexts.

---

## 🏗️ System Architecture

The system is designed as a decoupled microservices architecture.

| Service | Technology | Port | Role | Persistence |
| :--- | :--- | :--- | :--- | :--- |
| **API Gateway** | FastAPI | `8000` | Accepts requests, manages SSE streams. | Stateless |
| **Message Queue** | Redpanda (Kafka) | `9092` | Buffers high-volume uploads to prevent crashes. | Disk |
| **Worker Engine** | Python | N/A | Consumes queue, runs RAG pipeline. | Stateless |
| **Vector Memory** | Qdrant | `6333` | Stores text chunks + embeddings. | Docker Volume |
| **Graph Memory** | FalkorDB | `6380` | Stores entities & relationships. | AOF (Append Only File) |
| **State & Cache** | Redis | `6379` | Tracks job status & Pub/Sub for SSE. | Docker Volume |
| **Local Inference** | Ollama | `11434` | Runs Mistral for cost-effective processing. | Model Volume |

---

## 🔧 Troubleshooting

### 1. "Connection Refused" to Databases
* **Cause:** Containers aren't ready yet.
* **Fix:** Check status with `docker-compose ps`. If a container is restarting, check logs: `docker-compose logs falkordb`.

### 2. Ollama Context Window Error
* **Cause:** The model hasn't been pulled or the probe failed.
* **Fix:** Run the setup script manually again:
    ```bash
    docker exec ollama_service ollama pull mistral
    ```

### 3. Worker Crash on "Empty File"
* **Cause:** `UnstructuredLoader` couldn't parse a specific file type.
* **Fix:** The worker is designed to catch this, log it to `REDIS` as `FAILED`, and move to the next file. Check the worker console logs for the specific traceback.

---

## ⚙️ Customization

### Changing the LLM
To use a different local model (e.g., Llama 3), update `.env` and `setup.sh`:

1.  **Edit `.env`:**
    ```ini
    SMALL_MODEL=llama3
    ```
2.  **Pull the model:**
    ```bash
    docker exec ollama_service ollama pull llama3
    ```

### Using Azure Blob Storage
To switch from local temp storage to Azure:
1.  Update `src/utils/ingestion.py` to use `AzureBlobStorageFileLoader`.
2.  Update `main.py` to upload the stream directly to Azure Blob instead of `shutil.copyfileobj`.

---

## 📜 License
This project is provided as-is for Enterprise RAG architectural reference.

## 🏃 Running the System

You need two terminal windows running simultaneously.

### Terminal 1: The Worker Engine
Processes files from the Kafka queue, runs the RAG pipeline, and updates Redis status.
```bash
python worker.py
uvicorn main:app --reload --port 8000
```

federated_memory_system/
├── docker-compose.yml         # Infrastructure (DBs, Queue, LLM)
├── .env                       # Secrets & Configs
├── requirements.txt           # Python Dependencies
├── setup_ollama.sh            # Setup Script
├── main.py                    # API Gateway (FastAPI)
├── worker.py                  # Background Processor (Kafka Consumer)
└── src/
    ├── __init__.py
    ├── config.py              # Central Configuration
    ├── logging_config.py      # Central Logging Logic
    ├── prompts.py             # Optimized Prompts
    ├── utils/
    │   ├── __init__.py
    │   ├── context_manager.py # Dynamic Context Detection
    │   ├── ingestion.py       # OCR & Loading
    │   └── processing.py      # Text Splitting & Summarization
    └── modules/
    │   ├── __init__.py
    │   ├── llm.py             # Resilient LLM Service
    │   ├── graph.py           # FalkorDB Adapter
    │   └── vector.py          # Qdrant Adapter
    └── services/
    │   ├── __init__.py
    │   ├── retrieval.py       # Query Logic + Retries
    │   └── notification.py    # Redis Pub/Sub for User Subscription