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

- Identify failures within individual pipeline stages.
- Understand request execution flow.
- Monitor operation latency.
- Inspect retrieval and generation behaviour.
- Diagnose API and evaluation failures.
- Improve application reliability.

Logfire traces should be configured to avoid exposing API keys, database credentials, and sensitive user information.

---

## 12. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| Relational Database | Neon PostgreSQL |
| Vector Database Extension | pgvector |
| Cache and Memory | Upstash Redis |
| Embedding Models | Sentence Transformers / configured embedding model |
| Reranking | Cross-Encoder |
| LLM Integration | Configured external LLM providers |
| Evaluation | RAGAS |
| Observability | Pydantic Logfire |
| API Communication | HTTPX |
| Containerization | Docker |
| Reverse Proxy | Caddy |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry |
| Backend Hosting | Oracle Cloud Infrastructure |
| Frontend Hosting | Streamlit Community Cloud |

---

## 13. Deployment Architecture

The current production application uses a distributed cloud deployment.

### Backend: Oracle Cloud Infrastructure

The Oracle Cloud VM hosts the containerized backend infrastructure.

The deployed services include:

**FastAPI API**
- Handles incoming requests.
- Executes RAG pipeline operations.
- Connects to database and memory services.
- Provides health and evaluation endpoints.

**Caddy Reverse Proxy**
- Acts as the public reverse proxy.
- Handles HTTPS and forwards traffic to the API service.

**Containerized Services**
- Docker manages the deployed application containers.
- Docker Compose manages the production service configuration.

### Frontend: Streamlit Community Cloud

The Streamlit application is deployed independently.

It provides:

- User authentication and application access.
- HR policy question answering.
- Conversation history.
- RAGAS evaluation dashboard.

The UI communicates with the Oracle-hosted FastAPI backend over HTTPS.

### Managed Infrastructure

The application uses:

- Neon PostgreSQL for persistent storage and vector retrieval.
- Upstash Redis for memory and caching.
- Pydantic Logfire for observability.

### Live Application

**Streamlit Application:**  
https://enterprise-hr-assistant.streamlit.app/

**FastAPI Backend:**  
https://hr-policies-api.duckdns.org/docs

---

## 14. CI/CD Pipeline

GitHub Actions automates application validation, container image creation, publishing, and deployment.

### Deployment Lifecycle

\`\`\`mermaid
flowchart TD
    A[Developer Pushes Code to GitHub] --> B[GitHub Actions Triggered]

    B --> C[Dependency Installation]
    C --> D[Automated Test Suite]

    D -->|Tests Pass| E[Build API Docker Image]
    D -->|Tests Pass| F[Build UI Docker Image]

    E --> G[Push Images to GHCR]
    F --> G

    G --> H[Deploy to Oracle Cloud]
    H --> I[Pull Versioned Images]
    I --> J[Update Docker Compose Deployment]
    J --> K[Restart / Reconcile Services]
    K --> L[Health Check]
    L --> M[Production Application]
\`\`\`

### CI/CD Stages

**1. Source Control**

Code changes are committed and pushed to the GitHub repository.

**2. Automated Validation**

GitHub Actions installs the configured dependencies and executes automated tests.

**3. Docker Image Build**

The API and UI images are built after validation succeeds.

**4. Image Publishing**

Images are published to GitHub Container Registry with commit-based version tags.

**5. Oracle Deployment**

The deployment workflow connects to the Oracle VM and updates the deployment configuration and container images.

**6. Health Verification**

The deployed API is checked using its health endpoint.

### Benefits

- Consistent deployment process.
- Automated regression testing.
- Version-controlled container images.
- Reproducible deployment revisions.
- Reduced manual deployment work.
- Easier rollback to a previous image revision.

The currently verified production hosting path is Docker-based on Oracle Cloud. Kubernetes-related deployment configuration, where present in the repository, should be treated separately from this live Docker deployment.

---

## 15. Environment Configuration

Application configuration is supplied through environment variables and platform secret management.

Depending on the enabled services, configuration includes:

| Configuration | Purpose |
|---|---|
| `DATABASE_URL` | Neon PostgreSQL connection |
| `REDIS_URL` | Upstash Redis connection |
| `STREAMLIT_AUTH_SECRET` | Shared authentication configuration |
| `EVALUATION_API_TOKEN` | Evaluation endpoint authentication |
| `PORTKEY_API_KEY` | Model gateway authentication, where configured |
| `GROQ_MODEL` | Configured Groq model identifier |
| `OPENAI_MODEL` | Configured OpenAI model identifier |
| Logfire configuration | Observability and tracing |
| Model credentials | Authentication for configured model providers |

Refer to the application's settings and deployment files for the exact required variables.

**Security rule:** Never commit real credentials, database URLs, API keys, tokens, or private SSH keys to GitHub.

Use local environment files for development and the hosting platforms' secret-management features for production.

Rotate any credential that has been exposed in logs, screenshots, terminal output, or source control.

---

## 16. Local Development

### Prerequisites

- Python version supported by the repository.
- Existing project dependency environment.
- Git.
- Docker and Docker Compose, when running the containerized stack.
- Access to the configured Neon PostgreSQL database.
- Access to the configured Upstash Redis instance.
- Valid credentials for configured LLM providers and observability services.

### Clone the Repository

\`\`\`bash
git clone https://github.com/Adith006/HR-Policies-Rag_Application-With-Pydantic_Logfire_Tracing-and-Kubernetes_Deployment.git

cd HR-Policies-Rag_Application-With-Pydantic_Logfire_Tracing-and_Kubernetes_Deployment
\`\`\`

### Configure Environment Variables

Configure the required variables using the repository's environment example and application settings.

Do not put production secrets into source control.

### Install Dependencies

Use the dependency manager and lock file provided by the repository to install the application and development dependencies.

### Start the Containerized Application

Where the repository's Compose configuration is intended for local execution:

\`\`\`bash
docker compose up -d --build
\`\`\`

Check service status:

\`\`\`bash
docker compose ps
\`\`\`

Follow logs:

\`\`\`bash
docker compose logs -f
\`\`\`

Stop the local stack:

\`\`\`bash
docker compose down
\`\`\`

Use the actual environment configuration and Compose file for the deployment target you are running. The production database and Redis services are externally managed.

---

## 17. API and Application Access

### Health Endpoint

\`\`\`http
GET /health
\`\`\`

Used to verify API availability.

### Conversation Operations

The application exposes conversation operations, including conversation listing and persisted message retrieval.

### RAG Question Answering

The question-answering API accepts requests from the Streamlit application and executes the RAG pipeline.

### RAGAS Evaluation

\`\`\`http
POST /evaluate
\`\`\`

Executes the configured RAGAS evaluation workflow.

The endpoint uses its configured authentication requirements.

For exact request schemas, parameters, authentication headers, and response structures, refer to the FastAPI route and schema definitions in the source code.

---

## 18. Security Considerations

This application works with organizational policy content and user conversation data. Production deployments should maintain the following controls:

- Store credentials in secret-management systems.
- Enforce authentication on protected API endpoints.
- Validate user ownership for conversation access.
- Use HTTPS for external requests.
- Treat retrieved document instructions as untrusted content.
- Validate generated answers against retrieved policy context.
- Protect database and Redis credentials.
- Avoid logging secrets or unnecessary personal information.
- Review access permissions for managed services.
- Maintain backups and recovery procedures.
- Rotate credentials after accidental exposure.

Guardrails, logging, and evaluation are complementary controls. They do not replace application security testing or human review of important HR decisions.

---

## 19. Future Improvements

Potential areas for continued development include:

- More comprehensive document-ingestion monitoring and error handling.
- Improved retrieval benchmarking across different HR policy categories.
- Expanded evaluation datasets for ambiguous and complex questions.
- More detailed citation correctness evaluation.
- Retrieval and generation latency optimization.
- Token usage and LLM cost monitoring.
- Automated cleanup of synthetic evaluation sessions and memory.
- More comprehensive rate limiting and abuse prevention.
- Improved operational alerting and service monitoring.
- Document versioning and re-indexing workflows.
- Additional authorization and governance controls.

These are potential improvements and should not be interpreted as features already completed.

---

## Project Status

The HR Policies RAG Application has been deployed with:

- A containerized FastAPI backend on Oracle Cloud.
- A Streamlit frontend on Streamlit Community Cloud.
- Neon PostgreSQL with pgvector.
- Upstash Redis for memory and cache operations.
- Semantic retrieval and cross-encoder reranking.
- LLM-based answer generation.
- Conversation history and persistence.
- Guardrails and source construction.
- RAGAS evaluation.
- Pydantic Logfire tracing.
- GitHub Actions CI/CD and GitHub Container Registry.

The production workflow has been exercised end to end, including document retrieval, answer generation, conversation operations, and successful RAGAS evaluation.

---

## Repository

**GitHub:**  
https://github.com/Adith006/HR-Policies-Rag_Application-With-Pydantic_Logfire_Tracing-and_Kubernetes_Deployment

**Live Application:**  
https://enterprise-hr-assistant.streamlit.app/

**Backend API:**  
https://hr-policies-api.duckdns.org

---

*Built as an end-to-end applied AI engineering project focused on enterprise RAG, retrieval quality, conversational memory, security, evaluation, observability, and production deployment.*
