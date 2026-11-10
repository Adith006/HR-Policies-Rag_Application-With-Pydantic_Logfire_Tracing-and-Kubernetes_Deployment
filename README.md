# Enterprise HR Policies RAG Application

An end-to-end, production-oriented Retrieval-Augmented Generation (RAG) application designed to make organizational HR policies easier to search, understand, and access through natural-language conversations.

The platform combines document ingestion, vector search, semantic retrieval, cross-encoder reranking, LLM-powered answer generation, conversation memory, security guardrails, source citations, RAGAS evaluation, observability, and automated cloud deployment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [End-to-End RAG Workflow](#end-to-end-rag-workflow)
6. [Document Ingestion and Processing](#document-ingestion-and-processing)
7. [Retrieval and Answer Generation](#retrieval-and-answer-generation)
8. [Conversation Memory and Persistence](#conversation-memory-and-persistence)
9. [Security and Guardrails](#security-and-guardrails)
10. [RAGAS Evaluation](#ragas-evaluation)
11. [Observability with Pydantic Logfire](#observability-with-pydantic-logfire)
12. [Technology Stack](#technology-stack)
13. [Deployment Architecture](#deployment-architecture)
14. [CI/CD Pipeline](#cicd-pipeline)
15. [Environment Configuration](#environment-configuration)
16. [Local Development](#local-development)
17. [API and Application Access](#api-and-application-access)
18. [Security Considerations](#security-considerations)
19. [Future Improvements](#future-improvements)

---

## 1. Project Overview

Enterprise HR departments manage extensive policy documentation covering employee benefits, leave policies, compensation, compliance, workplace guidelines, and organizational procedures.

Traditional document-search systems often require employees to know exactly what they are looking for and where the information is located.

This project introduces an intelligent HR Policy Assistant powered by Retrieval-Augmented Generation.

Instead of searching manually through policy documents, users can ask questions in natural language and receive contextual responses grounded in retrieved policy information.

The platform follows a modular architecture that separates the user interface, backend orchestration, retrieval infrastructure, persistent storage, evaluation, and deployment.

## 2. Problem Statement

Traditional HR policy management presents several challenges:

- Employees spend time searching through multiple policy documents.
- Keyword-based searches often fail to identify contextually relevant information.
- Long documents make it difficult to locate specific clauses and exceptions.
- General-purpose LLMs can generate unsupported answers or hallucinate policy information.
- Conversation context is lost when users start new sessions.
- There is limited visibility into retrieval quality, answer grounding, and system performance.

### Proposed Solution

Build a production-oriented RAG application that:

- Ingests and indexes HR policy documents.
- Retrieves relevant information using semantic vector search.
- Improves retrieval quality through cross-encoder reranking.
- Generates contextual, source-grounded answers using LLMs.
- Maintains conversation history and memory.
- Applies guardrails to reduce prompt injection and unsupported responses.
- Evaluates retrieval and generation quality using RAGAS.
- Provides observability through Pydantic Logfire.
- Supports containerized deployment and automated CI/CD.

## 3. Key Features

| Feature | Description |
|---|---|
| Document ingestion | Processes HR policy content for indexing and retrieval |
| Semantic search | Retrieves relevant policy passages using vector similarity |
| Cross-encoder reranking | Improves the relevance of retrieved candidates |
| Context assembly | Constructs relevant context for LLM answer generation |
| Source citations | Provides supporting document and metadata information |
| Conversation history | Persists and retrieves user conversations |
| Short-term memory | Maintains recent conversational context using Redis |
| Long-term memory | Supports storing and retrieving conversation summaries |
| Answer caching | Reduces repeated processing for reusable answers |
| Security guardrails | Validates questions, retrieved content, and generated responses |
| RAGAS evaluation | Evaluates the quality of the RAG pipeline |
| Observability | Traces pipeline execution and performance using Logfire |
| Cloud deployment | Deploys containerized backend services and a cloud-hosted UI |
| Automated CI/CD | Automates testing, image building, publishing, and deployment |

## 4. System Architecture

The application separates the frontend, backend, databases, memory services, and observability infrastructure.

### High-Level Architecture

\`\`\`mermaid
flowchart TD
    USER[User] --> UI[Streamlit Community Cloud]

    UI -->|Authenticated HTTPS Request| CADDY[Caddy Reverse Proxy]

    CADDY --> API[FastAPI Backend]

    API --> VALIDATE[Question Validation and Intent Handling]

    VALIDATE --> MEMORY[Conversation Memory]
    VALIDATE --> CACHE[Answer Cache]
    VALIDATE --> RETRIEVAL[Semantic Retrieval]

    MEMORY <--> REDIS[(Upstash Redis)]
    MEMORY <--> DB[(Neon PostgreSQL)]

    RETRIEVAL --> VECTOR[pgvector Search]
    VECTOR --> DB

    VECTOR --> RERANK[Cross-Encoder Reranking]

    RERANK --> CONTEXT[Context Assembly]

    CONTEXT --> GUARD[Document Instruction Checks]

    GUARD --> LLM[LLM Answer Generation]

    LLM --> GROUND[Grounding Validation]

    GROUND --> CITATION[Source and Citation Construction]

    CITATION --> API

    API --> LOGFIRE[Pydantic Logfire]

    API --> UI

    EVAL[RAGAS Evaluation] --> API
\`\`\`

### Main Architectural Components

**Frontend**
- Streamlit application.
- Conversation interface and history.
- RAGAS evaluation dashboard.
- Authenticated communication with the backend API.

**Backend**
- FastAPI application.
- RAG pipeline orchestration.
- Question validation and intent handling.
- Retrieval, memory, caching, and guardrail integration.
- Evaluation API.

**Data Layer**
- Neon PostgreSQL for relational data and conversation persistence.
- pgvector for vector-based semantic retrieval.
- Upstash Redis for memory and cache operations.

**Observability**
- Pydantic Logfire for application tracing and execution visibility.

**Infrastructure**
- Docker containers.
- Caddy reverse proxy.
- GitHub Actions.
- GitHub Container Registry.
- Oracle Cloud Infrastructure.

## 5. End-to-End RAG Workflow

The application follows a multi-stage RAG pipeline.

\`\`\`mermaid
flowchart TD
    A[HR Policy Documents] --> B[Document Extraction]
    B --> C[Text Normalization]
    C --> D[Chunking and Metadata]
    D --> E[Embedding Generation]
    E --> F[(PostgreSQL with pgvector)]

    Q[User Question] --> V[Question Validation]
    V --> M[Memory Retrieval]
    M --> CACH[Answer Cache Lookup]

    CACH --> R[Semantic Retrieval]
    R --> P[Vector Search]
    F --> P

    P --> RE[Cross-Encoder Reranking]
    RE --> CT[Context Assembly]

    CT --> G[Document Instruction Checks]
    G --> L[LLM Answer Generation]
    L --> GV[Grounding Validation]
    GV --> S[Source and Citation Generation]

    S --> OUT[Response to User]
    OUT --> MEM[Memory and Cache Updates]
\`\`\`

### Pipeline Stages

1. Document ingestion and preparation.
2. Question validation and intent detection.
3. Short-term and long-term memory retrieval.
4. Answer-cache lookup.
5. Query embedding generation.
6. Semantic vector retrieval.
7. Cross-encoder reranking.
8. Context assembly.
9. Document instruction and security checks.
10. LLM answer generation.
11. Grounding validation.
12. Citation and source construction.
13. Conversation memory and cache updates.

---

## 6. Document Ingestion and Processing

The ingestion stage prepares organizational HR policy documents for semantic retrieval.

### Processing Lifecycle

**Step 1: Document Extraction**

Policy documents are processed to extract textual content and relevant document metadata.

**Step 2: Text Normalization**

Extracted content is normalized to improve consistency and remove unnecessary formatting artifacts.

**Step 3: Chunking**

Long documents are divided into smaller text segments.

Chunking helps ensure that retrieved information contains relevant context without requiring the entire document to be passed to the LLM.

Chunk size and overlap should be configured according to document structure and retrieval requirements.

**Step 4: Metadata Association**

Relevant metadata is retained with the content, including available source, page, and section information.

**Step 5: Embedding Generation**

Document chunks are converted into vector representations using the configured embedding model.

**Step 6: Vector Indexing**

The generated vectors, content, and associated metadata are stored in PostgreSQL using pgvector.

The resulting index supports semantic similarity search during question answering.

---

## 7. Retrieval and Answer Generation

The retrieval pipeline is responsible for identifying relevant policy passages and generating contextual responses.

### 7.1 Semantic Retrieval

The user question is converted into an embedding representation.

The embedding is used to retrieve semantically relevant document chunks from PostgreSQL using pgvector.

This allows the application to identify relevant information even when the wording of the question differs from the wording in the original document.

### 7.2 Cross-Encoder Reranking

Initial vector search retrieves candidate passages.

A cross-encoder reranker evaluates the relevance of those candidates against the original query and improves their ordering.

This provides a second relevance-filtering stage before context is constructed.

### 7.3 Context Assembly

The selected passages are combined into a context for the LLM.

Available metadata is retained to support source attribution and citations.

### 7.4 Answer Generation

The LLM receives:

- The user's question.
- Retrieved policy context.
- Relevant conversation context.
- Applicable instructions and response constraints.

The model generates a response using the retrieved material as its knowledge source.

### 7.5 Grounding and Citations

The generated response passes through grounding validation.

Source information is constructed from retrieved passages so users can identify the policy material supporting the response.

---

## 8. Conversation Memory and Persistence

The application supports conversation continuity through multiple memory and persistence mechanisms.

### Short-Term Memory

Redis-backed short-term memory maintains recent conversational turns and supports contextual follow-up questions.

### Long-Term Memory

The application supports saving and retrieving HR conversation summaries to provide relevant historical context.

### Conversation Persistence

Neon PostgreSQL stores conversation and message records.

Conversation operations include:

- Creating conversations.
- Saving user and assistant messages.
- Retrieving conversation history.
- Listing conversations.
- Enforcing user ownership during conversation access.

### Answer Caching

An answer-cache layer checks for reusable responses before performing unnecessary generation work.

The cache is updated according to the configured application logic.

### Evaluation Session Isolation

RAGAS evaluation creates synthetic session and user identifiers.

Persisted conversation-history restoration is disabled for these synthetic sessions because they do not represent previously created user conversations.

---

## 9. Security and Guardrails

Security and response reliability are important requirements for an application that handles organizational policy information.

The application includes multiple validation and guardrail stages.

### Question Validation

Incoming questions are checked before they enter the retrieval workflow.

### Document Instruction Checks

Retrieved document content is treated as untrusted input.

Instruction-like content inside documents should not be allowed to override the application's system-level instructions.

### Grounding Validation

Generated answers are checked against retrieved information to reduce unsupported responses.

### Conversation Ownership

Conversation retrieval and persistence operations enforce user and conversation ownership checks.

### API Authentication

The UI and API use configured authentication credentials for protected API operations.

### Evaluation Authentication

The evaluation endpoint supports its configured authentication mechanism.

### Security Objectives

- Reduce prompt-injection risks.
- Prevent cross-user conversation access.
- Keep API credentials out of source code.
- Avoid exposing database credentials and tokens.
- Reduce unsupported or ungrounded answers.
- Maintain visibility into application execution.

Guardrails reduce risk but do not guarantee that every generated answer is correct. Important HR decisions should be checked against authoritative policy documents and the responsible HR team.

---

## 10. RAGAS Evaluation

The project includes a RAGAS evaluation workflow to assess the RAG pipeline.

Evaluation is exposed through the backend API and integrated with the Streamlit evaluation dashboard.

### Evaluation Workflow

\`\`\`mermaid
flowchart TD
    A[Evaluation Question Set] --> B[Generate Synthetic Session]
    B --> C[Execute RAG Pipeline]
    C --> D[Capture Generated Answer]
    C --> E[Capture Retrieved Contexts]
    C --> F[Capture Source Information]

    D --> G[Construct Evaluation Samples]
    E --> G
    F --> G

    G --> H[RAGAS Evaluation]
    H --> I[Calculate Configured Metrics]
    I --> J[Generate Evaluation Report]
    J --> K[Streamlit Evaluation Dashboard]
\`\`\`

### Evaluation Stages

1. Load the evaluation question set.
2. Execute each question through the RAG pipeline.
3. Capture generated answers.
4. Capture retrieved contexts.
5. Capture source metadata.
6. Construct RAGAS evaluation samples.
7. Run the configured evaluation metrics.
8. Generate and return the evaluation report.

The configured metrics are used to inspect answer quality, retrieval relevance, and grounding performance.

Evaluation results depend on the selected dataset, metrics, models, and evaluator configuration.

RAGAS is used as an evaluation mechanism, not as a guarantee of production answer correctness.

---

## 11. Observability with Pydantic Logfire

Pydantic Logfire provides tracing and execution visibility across the application.

Instrumentation covers important stages of the RAG workflow, including:

- Question validation.
- Greeting and intent detection.
- Short-term memory retrieval.
- Long-term memory retrieval.
- Cache lookup and cache updates.
- Query embedding.
- pgvector retrieval.
- Cross-encoder reranking.
- Candidate diagnostics.
- Context assembly.
- Document instruction checks.
- LLM answer generation.
- Grounding validation.
- Citation construction.
- Conversation memory operations.
- RAGAS evaluation.

### Benefits