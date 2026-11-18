
import json
import logging

from app.llm import LLMClient

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self):
        self.llm = LLMClient()

    def plan(self, query: str, chat_history: list | None = None) -> dict:
        chat_history = chat_history or []

        history_text = ""
        for item in chat_history[-12:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role in {"user", "assistant"} and isinstance(content, str):
                history_text += f"{role}: {content}\n"

        messages = [
            {
                "role": "system",
                "content": """
You are the planning component of an Agentic RAG chatbot.

Use conversation history to understand follow-up questions.
Rewrite the current question as a standalone question.
Do not change its original meaning.

Return ONLY one valid JSON object with these keys:
standalone_query, intent, sources, requirements, search_queries.

Allowed values:
- intent: chat, knowledge, research
- sources: none, local, web, both

Classification rules:

1. CHAT:
Greetings, small talk, thanks, and simple conversational
messages that do not require factual information.
Use intent="chat", sources="none".

2. KNOWLEDGE:
Questions about information that may exist in the local
knowledge base. Use intent="knowledge", sources="local".

3. RESEARCH:
Current facts, recent developments, external research,
or questions requiring web sources.
Use intent="research", sources="web".

4. BOTH:
Use intent="research", sources="both" when local documents
and external evidence would both help.

Important:
- Never classify a factual question as chat.
- Use conversation history to resolve words like it, they,
  this, that, and similar references.
- If history does not help, use the current question as-is.
- For chat, requirements and search_queries must be empty.
- For retrieval, provide at least one requirement and
  one useful search query.
- Requirements must describe facts that evidence must support.
- Search queries must help retrieve evidence for those requirements.
- Return valid JSON only, without Markdown fences.

Example:
Current question: What are its limitations?
History: User asked about RAG.
standalone_query:
What are the limitations of Retrieval-Augmented Generation (RAG)?
""",
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n"
                    f"Current question:\n{query}"
                ),
            },
        ]

        try:
            response = self.llm.generate(
                messages=messages,
                max_tokens=3000,
                temperature=0.1,
            )

            response = response.strip()

            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0].strip()

            plan = json.loads(response)

            valid_sources = {"none", "local", "web", "both"}
            valid_intents = {"chat", "knowledge", "research"}

            if plan.get("sources") not in valid_sources:
                raise ValueError("Invalid source selection")

            if plan.get("intent") not in valid_intents:
                raise ValueError("Invalid intent")

            standalone_query = plan.get("standalone_query")
            if not isinstance(standalone_query, str) or not standalone_query.strip():
                raise ValueError("Invalid standalone query")

            requirements = plan.get("requirements")
            search_queries = plan.get("search_queries")

            if not isinstance(requirements, list):
                raise ValueError("Requirements must be a list")

            if not isinstance(search_queries, list):
                raise ValueError("Search queries must be a list")

            if plan["sources"] == "none":
                requirements = []
                search_queries = []

            elif not requirements or not search_queries:
                raise ValueError("Retrieval plan must include requirements and queries")

            if not all(isinstance(x, str) and x.strip() for x in requirements):
                raise ValueError("Invalid requirement")

            if not all(isinstance(x, str) and x.strip() for x in search_queries):
                raise ValueError("Invalid search query")

            plan["standalone_query"] = standalone_query.strip()
            plan["requirements"] = requirements
            plan["search_queries"] = search_queries

            logger.info(
                "Plan created: intent=%s, sources=%s",
                plan["intent"],
                plan["sources"],
            )

            return plan

        except Exception as error:
            logger.warning(
                "Planning failed; using safe fallback: %s", error
            )

            return {
                "standalone_query": query,
                "intent": "research",
                "sources": "both",
                "requirements": [
                    f"Find reliable evidence needed to answer: {query}"
                ],
                "search_queries": [query],
            }
