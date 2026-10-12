# Agentic RAG Chatbot

> An agent-driven Retrieval-Augmented Generation (RAG) chatbot that
> plans how to answer a query, retrieves evidence from local documents
> and the web, evaluates evidence coverage, refines retrieval when
> necessary, and generates a source-grounded response.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Retrieval](https://img.shields.io/badge/Retrieval-Qdrant-orange)
![LLM](https://img.shields.io/badge/LLM-OpenRouter-purple)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## Overview

Traditional RAG often follows a fixed `retrieve → generate` pipeline.
This project introduces an agentic workflow: an LLM plans the retrieval
strategy, evidence is gathered from the selected sources, and an
evaluator checks whether the evidence is sufficient to answer the user's
requirements. If important information is missing, the agent refines its
search and retrieves more evidence before generating the response.

The prototype supports: - **Query planning** to classify the request and
select a source route. - **Local semantic search** over project
documents using Sentence Transformers and Qdrant. - **Web retrieval**
using Tavily. - **Evidence sufficiency evaluation** against planned
information requirements. - **Iterative retrieval and query refinement**
when evidence is incomplete. - **Grounded response generation** with
source references. - **FastAPI endpoints** and interactive Swagger
documentation.

> **Implementation note:** This README distinguishes the functionality
> implemented in the current prototype from advanced stages shown in the
> proposed workflow diagram. Multi-query rewriting, an LLM evidence
> judge, and a dedicated coherent reranking stage are architectural
> extensions unless explicitly implemented in the current code.

## System Workflow

The diagram below follows the supplied workflow concept. It includes the
current prototype's core path and marks advanced routing/ranking steps
as extensions.

``` mermaid
flowchart TD
    U([User Query]) --> P[Planner]

    P -->|Greeting / small talk| C[Direct conversational response]
    P -->|Local knowledge| LR[Local retrieval]
    P -->|External / current information| WR[Web retrieval]
    P -->|Both sources useful| HR[Hybrid retrieval]

    LR --> E[Evidence evaluator]
    WR --> E
    HR --> E

    E -->|Evidence sufficient| G[Generate grounded answer]
    E -->|Evidence incomplete| M[Identify missing requirements]
    M --> Q[Refine search query]
    Q --> R[Choose retrieval source]
    R --> LR
    R --> WR
    R --> HR

    E -. Proposed extension .-> J[LLM evidence judge]
    J -. Unclear evidence ranking .-> RR[Coherent reranking]
    RR -. Reordered evidence .-> G

    C --> O([Response])
    G --> S[Attach source references]
    S --> O

    classDef proposed stroke-dasharray: 5 5,stroke:#9CA3AF,color:#9CA3AF;
    class J,RR proposed;
```

### Workflow, step by step

1.  **Receive query** --- accept a user message.
2.  **Plan** --- classify the request, define information requirements,
    and choose an appropriate retrieval route.
3.  **Retrieve evidence** --- search local indexed documents, the web,
    or both.
4.  **Evaluate coverage** --- check whether the collected evidence
    covers the planned requirements.
5.  **Iterate when needed** --- identify missing information, refine the
    query, and retrieve additional evidence within the configured round
    limit.
6.  **Generate** --- produce a response grounded in the collected
    evidence and include available source references.
7.  **Return response** --- the API returns the answer, sources, status,
    and retrieval-round count.

## Architecture

``` text
agentic_rag/
├── app/
│   ├── agent.py          # Coordinates planning, retrieval, evaluation, and generation
│   ├── config.py         # Loads configuration and environment variables
│   ├── evaluator.py      # Evaluates evidence coverage and sufficiency
│   ├── generator.py      # Generates chat and evidence-grounded answers
│   ├── ingestion.py      # Ingests local documents into the vector store
│   ├── llm.py            # OpenRouter chat-completion client
│   ├── main.py           # FastAPI application and API endpoints
│   ├── planner.py        # Plans intent, requirements, and retrieval strategy
│   └── retriever.py      # Local Qdrant and Tavily web retrieval
├── data/
│   ├── documents/        # Source .txt and .md documents
│   └── qdrant/           # Local vector database (generated; ignored by Git)
├── .env                  # Local secrets; do not commit
├── .env.example          # Safe environment-variable template
├── .gitignore
├── requirements.txt
└── README.md
```

The exact contents of `data/`, configuration variables, and dependencies
may vary as development continues. The local Qdrant database is
generated data and should not be committed.

## Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Language                Python                  Application and
                                                  orchestration logic

  API                     FastAPI                 HTTP API for chatbot
                                                  requests

  API validation          Pydantic                Request/response
                                                  schemas and validation

  LLM access              OpenRouter API using    Query planning,
                          the OpenAI Python       evaluation, and answer
                          client                  generation

  Config                  `python-dotenv` /       Load API keys and
                          environment variables   runtime settings

  Embeddings              Sentence Transformers   Convert document text
                                                  and queries into
                                                  vectors

  Vector database         Qdrant local mode       Store and retrieve
                                                  semantically similar
                                                  document chunks

  Web search              Tavily API              Retrieve external web
                                                  evidence

  API testing             Swagger UI / OpenAPI    Test endpoints
                                                  interactively
  -----------------------------------------------------------------------

## Current Features

### Query planner

-   Classifies messages such as greetings and information-seeking
    questions.
-   Selects a retrieval route such as `none`, `local`, `web`, or `both`.
-   Produces information requirements and search queries for retrieval.
-   Includes a fallback plan if the LLM response cannot be parsed.

### Retrieval

-   **Local:** searches the Qdrant vector store using sentence
    embeddings.
-   **Web:** queries Tavily and returns titles, URLs, text, and
    relevance scores.
-   **Combined retrieval:** supports collecting local and web evidence
    when both are useful.
-   Deduplicates retrieved evidence where possible.
-   Applies a minimum relevance-score filter to web results.

### Evidence evaluation and iteration

-   Compares retrieved evidence with the planner's information
    requirements.
-   Returns covered and missing requirements with a sufficiency
    decision.
-   Refines queries and continues retrieval when evidence is
    insufficient.
-   Limits retrieval loops using a configurable maximum-round count.
-   Falls back conservatively when evaluation fails.

### Answer generation

-   Supports direct conversational responses without retrieval for
    simple greetings.
-   Generates answers from retrieved evidence for information-seeking
    queries.
-   Includes source references and returns source metadata separately.
-   Instructs the model not to invent unsupported facts.

### API

-   `GET /` --- basic API status.
-   `GET /health` --- health check.
-   `POST /chat` --- process a chatbot message.
-   `GET /docs` --- interactive Swagger UI provided by FastAPI.

## Getting Started

### 1. Requirements

-   Python 3.10 or newer is recommended.
-   Git.
-   An OpenRouter API key.
-   A Tavily API key for web retrieval.

Provider models and free-tier availability can change. The configured
model in `.env` must be available to your OpenRouter account.

### 2. Clone the repository

``` bash
git clone https://github.com/alobroke/agentic_rag.git
cd agentic_rag
```

### 3. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can invoke the environment's Python
directly with `.\venv\Scripts\python.exe`.

### 4. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy the example file:

``` powershell
Copy-Item .env.example .env
```

Add your own keys to `.env`:

``` dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
TAVILY_API_KEY=your_tavily_api_key
```

Never commit `.env` or paste real keys into issue reports, screenshots,
or public repositories. If a key is exposed, revoke it with the provider
and generate a replacement.

### 6. Add local documents

Place supported `.txt` or `.md` documents in `data/documents/`. Run the
ingestion script or command provided by the current `app/ingestion.py`
implementation to index them before testing local retrieval. The precise
command depends on the current ingestion entry point.

### 7. Start the API

``` powershell
python -m uvicorn app.main:app --reload
```

Open: - API root: <http://127.0.0.1:8000/> - Health check:
<http://127.0.0.1:8000/health> - Swagger UI:
<http://127.0.0.1:8000/docs>

## API Usage

### `POST /chat`

Example request:

``` json
{
  "message": "What is Agentic RAG?",
  "conversation_id": "test-001",
  "chat_history": []
}
```

Example response shape (the actual answer and sources will vary):

``` json
{
  "answer": "Agentic RAG uses an agent to plan and refine retrieval before generating a grounded response. [Source 1]",
  "sources": [
    {
      "id": 1,
      "title": "Example source title",
      "url": "https://example.com/article",
      "source_type": "web"
    }
  ],
  "status": "answered",
  "rounds": 1,
  "conversation_id": "test-001"
}
```

`conversation_id` is currently passed through the API response;
persistent conversation storage is not implied by this field. The
request's `chat_history` can be supplied by the caller, but durable
multi-session memory is not currently described as implemented.

## Testing

Test the API from Swagger at `/docs` or use PowerShell:

``` powershell
$body = @{
    message = "What is Agentic RAG?"
    conversation_id = "test-001"
    chat_history = @()
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Useful test cases: 1. **Greeting:** `Hi, how are you?` --- should take
the direct-chat path. 2. **Local retrieval:** ask a question covered by
files in `data/documents/`. 3. **Web retrieval:** ask a question
requiring external information. 4. **Combined retrieval:** ask a
question where local context and external evidence are both useful. 5.
**Insufficient evidence:** ask about a detail not present in the local
documents and inspect whether the agent searches again or communicates
limitations.

A `200 OK` response confirms the API handled the request, but it does
not alone prove that an answer is factually correct. Inspect source
relevance, citation support, and whether the evaluator stops too early.

## Current Development Status

The prototype has been exercised successfully for: - LLM connectivity
through OpenRouter. - Planner routing for greetings and
information-seeking queries. - Evidence evaluation. - Local Qdrant
retrieval. - Tavily web retrieval. - Agent execution with an answer and
source metadata. - FastAPI `/chat` returning HTTP `200 OK` through
Swagger.

The following are useful next improvements rather than claims of
completed functionality: - Persistent conversation memory. - Automated
unit and integration tests. - More robust citation-to-claim
validation. - Improved retrieval precision and chunk deduplication. -
Multi-query rewriting for comparison/aggregation questions. - Dedicated
reranking and an LLM-based evidence judge. - Better observability,
retries, rate-limit handling, and structured error reporting. -
Evaluation datasets and metrics for retrieval quality and answer
faithfulness.

## Security and Git Hygiene

The repository should exclude: - `.env` and all files containing
credentials. - `venv/`, `.venv/`, Python cache, and logs. -
`data/qdrant/` and other generated vector-store files.

Before pushing, verify the staged file list:

``` powershell
git status --short
git diff --cached --name-only
```

If a secret has ever been committed, removing it from the latest working
tree is not enough to erase it from Git history. Revoke/rotate the
exposed key and follow GitHub's guidance for cleaning repository history
if needed.

## Limitations

-   Answer quality depends on the LLM, retrieved evidence, prompts, and
    evaluator decisions.
-   Search results and provider quotas/availability can change.
-   Similarity scores are ranking signals, not calibrated probabilities
    of correctness.
-   Web content may be incomplete, outdated, or unreliable; source
    quality should be checked.
-   The current project is a prototype and should be evaluated before
    production use.

## Roadmap

-   [ ] Persist conversation state by conversation ID.
-   [ ] Add automated tests for planner, retriever, evaluator,
    generator, and API.
-   [ ] Validate citations against individual claims.
-   [ ] Add query decomposition and multi-query rewriting.
-   [ ] Add a reranker for retrieved passages.
-   [ ] Add evaluation metrics for retrieval recall, evidence coverage,
    and groundedness.
-   [ ] Add Docker deployment and CI.
-   [ ] Add structured logging, tracing, retries, and rate-limit
    handling.

## Contributing

Issues and pull requests are welcome. For bug reports, include the
relevant error message and reproduction steps, but remove API keys,
tokens, and private data first.

## License

No license has been specified yet. Add a `LICENSE` file before
presenting the repository as open source under a particular license.
