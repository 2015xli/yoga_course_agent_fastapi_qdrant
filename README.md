# 🧘‍♀️ Yoga Course Recommendation System

This repository contains a fully-featured, **card-driven multi-agent system** that recommends yoga courses tailored to a user’s natural-language request and physical constraints.

Key technologies:

* **Google Agent Dev Kit (ADK)** – each agent ships with a `card.yaml` for automatic discovery and HTTP schema negotiation.
* **Qdrant (vector store)** – provides high-recall semantic search over poses, categories and courses. A **single local Qdrant server** is launched on-demand and shared by every process to avoid file-lock issues.
* **Neo4j (graph DB)** – stores rich relationships between poses, challenges, anatomy, and courses.
* **FastAPI** – powers the HTTP endpoints for the pose-checker and the combined course-finder / category-recommender agent.
* **OpenAI / DeepSeek LLMs** – used for query understanding, candidate filtering, and dynamic course composition.

The pipeline first tries to find an existing course (`find_courses` action). If none passes a strict pose-safety check, it composes a brand-new sequence (`compose_course` action) that is then validated pose-by-pose.

---

## 🗂️ Directory Overview

```
.
├── agents/
│   └── course_finder_adk/       # FastAPI server + card for both actions
├── services/
│   └── pose_checker/           # Pose suitability HTTP micro-service
├── qdrant_db/                  # Auto-created local Qdrant storage
├── data/array_*.json           # Source data for graph + vector store
├── build_graphrag.py           # One-shot script: populate Neo4j & Qdrant
├── yoga_application_runner.py  # Main orchestration entry-point
├── yoga_models.py              # Shared light-weight dataclasses
└── ...
```

---

## ⚡ Quickstart

1. **Clone & install deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt   # or pip install -e . if you create one
   ```

2. **Set environment variables**
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=12345678
   # Choose ONE of the following LLM keys
   export DEEPSEEK_API_KEY=...   # or
   export OPENAI_API_KEY=...
   ```

3. **Load data** (Neo4j + Qdrant)
   ```bash
   python build_graphrag.py
   ```

4. **Run the whole application**
   ```bash
   python yoga_application_runner.py \
       --query "I need a 30-minute strength session, but no headstands." \
       --api deepseek   # or openai
   ```
   The runner will:
   1. Spawn the pose-checker server (random port)
   2. Discover `card.yaml`, start the agent FastAPI server, and wire the two actions automatically
   3. Query Qdrant + Neo4j for candidate courses → validate → fallback to composition if needed

---

## 🔍 Agents & Actions

| Agent (dir)                        | ADK Action ID | Endpoint            | Purpose |
|------------------------------------|---------------|---------------------|---------|
| `agents/course_finder_adk`         | `find_courses`| POST `/find-courses`| Return up to *k* existing course names matching user intent |
| `agents/course_recommender_adk`         | `compose_course` | POST `/compose-course` | Compose a brand-new pose sequence if no existing course passes validation |

Agents are specified **only** by their `card.yaml`; the runner never hardcodes paths or schemas.

---

## 🏗️ Development Tips

* **Rapid reload** – run agent server with `uvicorn ... --reload` during dev.
* **Qdrant UI** – launch `qdrant` binary separately to inspect collections at `http://localhost:6333/dashboard`.
* **Logs** – set `LOG_LEVEL=DEBUG` for verbose tracing.

---

## 📜 License

MIT – see `LICENSE`.


## Functionality Overview

The system operates through a coordinated set of components:

1.  **Data Ingestion (`build_graphrag.py`)**: Populates Neo4j with yoga pose, attribute, category, challenge, and course data, and builds ChromaDB collections for semantic search.
2.  **Pose Suitability Checking (`services/pose_checker/server.py`)**: A FastAPI service that checks if a specific yoga pose is suitable for a user based on their contraindications and poses to avoid. It can also suggest replacement poses from the knowledge graph.
3.  **Existing Course Recommendation (`agents/course_finder/agent.py`)**: An agent that searches for pre-defined yoga courses that semantically match a user's query.
4.  **Dynamic Course Composition (`agents/category_recommender/agent.py`)**: An agent that composes a new yoga pose sequence based on the user's objectives and relevant yoga categories.
5.  **Orchestration (`yoga_application_runner.py`)**: The main application runner that orchestrates the entire process. It first attempts to find and validate an existing course. If no suitable existing course is found, it falls back to composing a new course, validating each pose in the sequence for user suitability.

## Using Instructions

To set up and run the application, follow these steps:

### Prerequisites

*   **Python 3.11+**: Ensure you have a compatible Python version installed.
*   **Neo4j Database**: A running Neo4j instance (e.g., via Docker or local installation). The application expects it to be accessible at `bolt://localhost:7687` with user `neo4j` and password `12345678` by default (configurable via environment variables `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).
*   **LLM API Key**: An API key for either OpenAI or DeepSeek (configurable via environment variables `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`).

### Installation

1.  **Clone the repository** 
2.  **Navigate to the project root directory**: `/home/xli/NAS/home/bin/yoga-info-processing/`
3.  **Make or enter your python virtual environment**
4.  **Install Python dependencies**:
    ```bash
    pip install neo4j openai chromadb-client uvicorn fastapi requests
    ```

### Building the Knowledge Graph and ChromaDB

Before running the application, you need to populate the databases:

```bash
python build_graphrag.py
```
This script will connect to your Neo4j instance, clear existing data, load data from the `array_*.json` files, and build the ChromaDB collections.

### Running the Application

Execute the main runner script with your user query:

```bash
bash
python yoga_application_runner.py --query "I need a 30-minute session for back pain, but I have a weak neck and can't do headstands." --api deepseek
```

*   Replace the `--query` value with your desired yoga request.
*   Choose `--api openai` or `--api deepseek` based on your configured LLM API key.

The runner will start the pose checker API server in the background, interact with the course finder and category recommender agents, validate the poses, and print the recommended yoga sequence to the console.

## Explanation of Files

Here's a breakdown of the files and directories in this project:

*   **`array_anatomical.json`**: JSON data file containing anatomical information (likely for pose descriptions).
*   **`array_attribute.json`**: JSON data file listing various yoga attributes.
*   **`array_category.json`**: JSON data file listing different yoga categories.
*   **`array_challenge.json`**: JSON data file listing different challenge levels for yoga.
*   **`array_course.json`**: JSON data file containing definitions of pre-existing yoga courses, including their sequences.
*   **`array_health_issue.json`**: JSON data file containing health issues (likely for contraindications).
*   **`array_pose.json`**: JSON data file containing detailed information about individual yoga poses.
*   **`build_graphrag.py`**:
    *   **Purpose**: Script responsible for building and populating the Neo4j knowledge graph and ChromaDB vector databases from the `array_*.json` files. It's the initial setup script for the data layer.
*   **`check_yoga_pose.py`**:
    *   **Purpose**: Contains the core logic for checking the suitability of a single yoga pose against user-defined contraindications and a list of poses to avoid. It can also find a replacement pose from the Neo4j graph if the original is unsuitable. This file is imported and used by the `pose_checker` service.
*   **`get_course_candidates_for_query.py`**:
    *   **Purpose**: Implements the `CourseFinder` class, which uses LLMs and ChromaDB to semantically search for existing yoga courses that match a user's query. It retrieves course descriptions from Neo4j and filters candidates. This file is imported and used by the `course_finder` agent.
*   **`get_user_query_key_info.prompt`**:
    *   **Purpose**: A prompt template used by LLMs to extract structured information (objectives, contraindications, poses to avoid, etc.) from a user's natural language query.
*   **`prompt_to_write_app.txt`**: (Likely a historical prompt used during development, not part of the application's runtime logic.)
*   **`recommend_course_from_category.py`**:
    *   **Purpose**: Implements the `CategoryCourseRecommender` class, which uses LLMs and Neo4j to dynamically compose a new yoga pose sequence based on user objectives and related yoga categories. This file is imported and used by the `category_recommender` agent.
*   **`summary.txt`**: (Likely a historical summary, not part of the application's runtime logic.)
*   **`yoga_application_runner.py`**:
    *   **Purpose**: The main entry point and orchestrator of the entire application. It starts the `pose_checker` FastAPI server, coordinates with the `course_finder` and `category_recommender` agents, validates pose sequences, and handles fallbacks and retries.
*   **`chroma_db/`**:
    *   **Purpose**: Directory containing the persistent data for the ChromaDB vector database.
*   **`neo4j_db/`**:
    *   **Purpose**: Directory containing configuration and data for the Neo4j database (e.g., `neo4j.dump` for a pre-built database, or `relate.project.json` for project metadata).
*   **`agents/`**:
    *   **Purpose**: Top-level directory for all agent-specific modules.
    *   **`agents/course_finder/`**:
        *   **`agents/course_finder/agent.py`**: Implements the `CourseFinderAgent` class, which wraps the `get_course_candidates_for_query.py` logic and defines the agent's API (request/response data structures).
    *   **`agents/category_recommender/`**:
        *   **`agents/category_recommender/agent.py`**: Implements the `CategoryRecommenderAgent` class, which wraps the `recommend_course_from_category.py` logic and defines the agent's API.
*   **`services/`**:
    *   **Purpose**: Top-level directory for all shared service modules.
    *   **`services/pose_checker/`**:
        *   **`services/pose_checker/server.py`**: Implements the FastAPI server for the pose checking service. It exposes an API endpoint that utilizes the `check_yoga_pose.py` logic.

```
