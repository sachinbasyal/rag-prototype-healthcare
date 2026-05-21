# Enterprise Corrective RAG (CRAG) Pipeline for Healthcare

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-orange)
![Local Execution](https://img.shields.io/badge/Privacy-100%25_Local-brightgreen)

## Project Overview
This repository contains an end-to-end **Corrective Retrieval-Augmented Generation (CRAG)** prototype designed specifically for the healthcare sector. Developed as a master's capstone project, this application serves as an interactive healthcare assistant capable of securely querying medical policies, patient protocols, and billing documents.

To simulate the strict data privacy requirements (HIPAA) of enterprise healthcare systems and financial institutions, this pipeline is engineered to run **100% locally on CPU**. No data is ever sent to external APIs like OpenAI. 

Instead of a traditional, linear pipeline, this project utilizes a state-machine architecture to evaluate its own retrieved context and intercept hallucinations before they reach the user, guaranteeing the deterministic reliability required in regulated environments.

## Core Technology Stack
This project leverages the modern, modularized AI ecosystem:
* **Orchestration & Agents:** `langgraph` (State machine routing) & `langchain-core` (LCEL).
* **Dedicated Integrations (LangChain 0.3+):** Utilizes `langchain-ollama` and `langchain-huggingface` for modernized LLM and embedding management.
* **Local Generative AI:** `Ollama` (Running highly quantized models like `phi3` or `qwen2.5:1.5b`).
* **Vector Storage & Retrieval:** `faiss-cpu` integrated via `langchain-community`.
* **Embeddings & PyTorch:** `sentence-transformers` and `transformers` powered by local PyTorch to run the `all-MiniLM-L6-v2` embedding model directly on local hardware.
* **Evaluation & Tracking:** `mlflow` for logging latency and tracking generation metrics.

## Agentic Workflow (LangGraph Architecture)
The system operates as a self-correcting agent using a cyclic graph structure rather than a static script.

1. **Document Ingestion (LangChain):** Medical policies are parsed, chunked via `RecursiveCharacterTextSplitter`, embedded using local HuggingFace transformers, and stored in a FAISS vector database.
2. **Retrieve Node:** Fetches top-k relevant chunks using LangChain's retrievers based on the user's healthcare query.
3. **Grade Context Node (The First Guardrail):** The LLM evaluates the retrieved chunks. *"Are these documents actually relevant to the question?"* * If all documents are deemed irrelevant, the graph safely ends the workflow to prevent hallucination.
4. **Conditional Edge Routing:** If documents are deemed irrelevant, the graph safely intercepts the flow and abstains from answering to prevent hallucination.
5. **Generate Node:** The LLM drafts an answer grounded *strictly* in the filtered medical context.
6. **Hallucination Grader Node (The Final Guardrail):** A secondary LLM pass checks the generated output against the retrieved documents. If ungrounded claims are detected, the agent safely abstains from answering.

## Installation & Local Setup
## Setup Steps:
#### Step 1: Create environment
    python -m venv .venv

#### Windows
    .\.venv\Scripts\activate

#### Mac/Linux
    source .venv/bin/activate

#### Step 2: Install necessary packages
*Note: Due to the inclusion of sentence-transformers, this will download local PyTorch binaries. The installation may take several minutes depending on your CPU and network speed.*

    pip install -r requirements.txt

#### Step 3: Ensure Ollama is Running
You must have `Ollama` installed on your local machine. Pull your preferred lightweight model:

    ollama run phi3

## Usage
#### Build the Vector Database:

    python src/build_index.py

This script will ingest the mock healthcare documents from /data/raw and serialize a FAISS index to the /artifacts folder.

#### Run the Agent:
    python app/main_agent.py

#### Streamlit UI Interface:
 Deploy a local Streamlit interface for non-technical user evaluation.

    streamlit run app/streamlit_app.py

This will automatically pop open a new tab in your web browser (usually at http://localhost:8501).

#### *Snapshot:*

![alt text](<streamlit_UI.png>)

## Future Roadmap
[ ] Implement Corrective Query Rewriting (looping back to retrieval if documents fail grading).

[ ] Integrate MLflow explicitly into the LangGraph nodes for step-by-step latency tracking.


## About the Author
Sachin Basyal is an Applied 
AI/ML Engineer and Data Analytics Professional pursuing an M.S. in Computer Science with a specialization in Data Analytics. He specializes in bridging the gap between over a decade of enterprise IT leadership and cutting-edge predictive science, architecting scalable machine learning models and actionable business intelligence solutions.

🔗 [LinkedIn Profile](https://www.linkedin.com/in/sachin-basyal-3421602b/) | 🌐 [Project Portfolio](https://sachinbasyal.com/da-projects)