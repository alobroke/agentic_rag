
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent = Agent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic RAG API")
    yield

    if agent._retriever is not None:
        agent._retriever.close()

    logger.info("Agentic RAG API shut down")


app = FastAPI(
    title="Agentic RAG Chatbot",
    description="A chatbot with iterative retrieval, evidence evaluation, and web search.",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    chat_history: list[dict] = Field(default_factory=list)


class Source(BaseModel):
    id: int
    title: str
    url: str | None = None
    source_type: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    rounds: int
    conversation_id: str | None = None


@app.get("/")
def root():
    return {
        "message": "Agentic RAG Chatbot API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = agent.run(
            query=request.message.strip(),
            chat_history=request.chat_history[-12:],
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            status=result.get("status", "answered"),
            rounds=result.get("rounds", 0),
            conversation_id=request.conversation_id,
        )

    except Exception as error:
        logger.exception("Chat request failed: %s", error)

        raise HTTPException(
            status_code=500,
            detail="The chatbot could not process this request.",
        ) from error
