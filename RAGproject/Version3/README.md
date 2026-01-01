# ArXiv-based Research Ideation and Paper Drafting System

This repository contains the implementation of a **RAG-based research ideation and paper drafting system**, developed as part of the **SC6117 Capstone Project**.  
The system automatically mines research gaps from arXiv papers, generates and evaluates research ideas, and produces a structured paper draft through an interactive web interface.

---

## 1. Project Overview

### Motivation
Academic researchers often spend significant time:
- surveying large volumes of literature,
- identifying meaningful research gaps,
- and drafting initial paper structures.

Existing tools provide text generation but lack **evidence grounding** and **systematic ideation workflows**.  
This project addresses the problem by building an **evidence-driven research assistant grounded in arXiv literature**.

### Key Capabilities
- Research gap mining from arXiv papers
- Evidence-based idea generation
- Automatic idea evaluation
- Structured paper draft generation
- Interactive web-based user interface

---

## 2. System Architecture

The system follows a **frontend–backend separation** with an asynchronous RAG pipeline:

```

Browser (Frontend UI)
|
|  REST API (/api/tasks)
v
FastAPI Backend (api_server.py)
|
|  Background Tasks
v
RAG Pipeline
├── gap_miner.py
├── idea_generator.py
├── idea_evaluator.py
├── paper_draft_writer.py
|
v
Task Results (JSON)

```

### Components
- **Frontend**:  
  - Single-page UI (`static/index.html`)
  - Allows users to input research topics and monitor progress
- **Backend**:  
  - FastAPI server (`api_server.py`)
  - Manages tasks, progress tracking, and API endpoints
- **RAG Pipeline**:  
  - Retrieves papers from arXiv
  - Extracts structured content from PDFs
  - Grounds generation on retrieved evidence

---

## 3. Technical Stack

- **Programming Language**: Python 3.9
- **Backend Framework**: FastAPI
- **Frontend**: HTML + Bootstrap + JavaScript
- **LLM Interface**: OpenAI-compatible API (via `llm_client.py`)
- **Document Processing**: PDF parsing and structured extraction
- **Deployment Mode**: Local server with REST API

---

## 4. Repository Structure

```

RAGproject/Version3/
├── api_server.py                # FastAPI backend
├── gap_miner.py                 # Research gap mining
├── idea_generator.py            # Idea generation
├── idea_evaluator.py            # Idea evaluation
├── paper_draft_writer.py        # Paper draft generation
├── paper_retriever.py           # arXiv paper retrieval
├── paper_analyzer.py            # Paper analysis utilities
├── pdf_processor_fullscan.py    # Full PDF processing
├── pdf_utils.py                 # PDF utilities
├── pipeline_service.py          # Pipeline orchestration
├── llm_client.py                # LLM API client
├── static/
│   └── index.html               # Frontend UI
├── .gitignore                   # Git ignore rules

```

---

## **5. Setup Instructions**

This project is implemented in **Python 3.9** and is designed to run in a local environment using a virtual environment.

### 5.1 Prerequisites

* Python **3.9**
* Internet connection (for arXiv access and LLM API calls)

---

### 5.2 Create and Activate Virtual Environment

From the project root directory:

```bash
python -m venv venv
```

Activate the environment:


```bash
venv\Scripts\activate
```


---

### 5.3 Install Dependencies

Install the required Python packages:

```bash
pip install fastapi uvicorn requests tqdm
```

> Additional packages may be required depending on the PDF processing backend and LLM provider.

---

### 5.4 Environment Variables

Create a `.env` file in the `Version3` directory (do **not** commit this file):

```env
OPENAI_API_KEY=your_api_key_here
```

The LLM interface is implemented in `llm_client.py` and reads the API key from environment variables.

---

## **6. System Architecture and Pipeline**

The system implements an **end-to-end research ideation and paper drafting pipeline**, grounded in **real arXiv papers and their PDF content**.

### Overall Pipeline Flow

```
User Topic
   ↓
arXiv Paper Retrieval
   ↓
Paper Analysis (LLM-based)
   ↓
Research Gap Mining
   ↓
Idea Generation
   ↓
Idea Evaluation & Ranking
   ↓
Supporting Paper Retrieval + PDF Full Scan
   ↓
Paper Draft Generation (Markdown)
```

Each stage produces structured outputs that are consumed by subsequent stages, enabling modular execution and debugging.

---

## **7. Core Modules**

### 7.1 Paper Retrieval (`paper_retriever.py`)

* Queries arXiv based on user-provided topics
* Retrieves recent and relevant papers
* Outputs paper metadata including title, abstract, authors, and PDF URL

---

### 7.2 Paper Analysis (`paper_analyzer.py`)

* Uses LLMs to extract:

  * research tasks
  * methodological details
  * contributions
  * limitations
* Enforces structured outputs to support downstream gap mining

---

### 7.3 Research Gap Mining (`gap_miner.py`)

* Aggregates limitations and future work across papers
* Produces **actionable and non-trivial research gaps**
* Avoids generic gaps through structural constraints

---

### 7.4 Idea Generation (`idea_generator.py`)

* Generates concrete research ideas from mined gaps
* Each idea includes:

  * problem statement
  * proposed method
  * experimental design
  * evaluation metrics

---

### 7.5 Idea Evaluation (`idea_evaluator.py`)

* Scores ideas based on:

  * novelty
  * feasibility
  * grounding in prior work
  * potential impact
* Outputs ranked ideas for selection

---

### 7.6 PDF Full Scan (`pdf_processor_fullscan.py`)

* Performs full PDF processing for selected supporting papers
* Extracts:

  * methods
  * equations
  * experimental settings
  * limitations
* Uses caching to avoid redundant computation

---

### 7.7 Paper Draft Generation (`paper_draft_writer.py`)

* Generates a structured academic paper draft
* Sections include:

  * Abstract
  * Introduction
  * Related Work
  * Methodology
  * Experiments
  * Conclusion
* Drafts are grounded in **real PDF-extracted content** to reduce hallucination

---

## **8. Running the System**

### 8.1 Command-Line Execution

Run the full pipeline using:

```bash
python paper_draft_writer.py
```

You will be prompted to enter a research topic.
Intermediate results (gaps, ideas, evaluations) are generated sequentially.

---

### 8.2 Web Interface (Optional)

A lightweight web interface is provided for interactive usage.

#### Start Backend Server

```bash
python -m uvicorn api_server:app --reload --port 8000
```

#### Access Frontend

Open a browser and visit:

```
http://127.0.0.1:8000
```

The frontend allows users to:

* submit research topics
* monitor pipeline progress
* view generated drafts

---

## **9. Project Structure**

```
Version3/
├── api_server.py
├── gap_miner.py
├── idea_generator.py
├── idea_evaluator.py
├── paper_analyzer.py
├── paper_draft_writer.py
├── paper_retriever.py
├── pdf_processor_fullscan.py
├── pdf_utils.py
├── llm_client.py
├── pipeline_service.py
├── static/
│   └── index.html
├── pdf_cache/               # runtime cache (ignored in Git)
├── pdf_structured_cache/    # runtime cache (ignored in Git)
└── tasks/                   # runtime outputs (ignored in Git)
```

---

## **10. Limitations and Future Work**

### Current Limitations

* Evaluation is LLM-based rather than empirical
* PDF extraction quality depends on document formatting
* System is designed for single-user execution

### Future Improvements

* Support for additional academic databases
* Hybrid retrieval and reranking strategies
* Multi-round idea refinement






