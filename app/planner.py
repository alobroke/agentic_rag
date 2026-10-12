
import json
import logging

from app.llm import LLMClient

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self):
        self.llm = LLMClient()

    def plan(self, query: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": """
You are the planning component of an Agentic RAG chatbot.

Return ONLY one valid JSON object with these keys:
intent, sources, requirements, search_queries.

Allowed values:
- intent: chat, knowledge, research
- sources: none, local, web, both

Classification rules (follow in order):

1. CHAT:
Use intent="chat" and sources="none" ONLY for greetings,
small talk, thanks, and simple conversational messages that
do not ask for factual information.

2. KNOWLEDGE:
Use intent="knowledge" and sources="local" when the user asks
about information that may be available in the local knowledge base.

3. RESEARCH:
Use intent="research" and sources="web" when the user asks
for current facts, recent developments, external research,
or information that requires web sources.

4. BOTH:
Use intent="research" and sources="both" when both local
documents and external evidence would help answer the question.

IMPORTANT:
- Questions starting with "what", "why", "how", "when", or
  "compare" are not automatically chat.
- "What is Agentic RAG?" is a factual question, NOT a greeting.
  For this example, choose intent="research", sources="web".
- When uncertain whether a message is factual or conversational,
  prefer retrieval rather than silently skipping evidence.
- For chat, requirements and search_queries must be empty.
- For retrieval, provide at least one requirement and one
  useful search query.

Example 1:
Input: Hi, how are you?
Output:
{"intent":"chat","sources":"none","requirements":[],"search_queries":[]}

Example 2:
Input: What is Agentic RAG?
Output:
{"intent":"research","sources":"web","requirements":["Find a reliable definition of Agentic RAG and explain how its iterative retrieval and evaluation work"],"search_queries":["Agentic RAG architecture iterative retrieval evaluation"]}

Return JSON only. Do not include Markdown fences or explanations.
""",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        try:
            response = self.llm.generate(
                messages=messages,
                max_tokens=3000,
                temperature=0.1,
            )

            # Handle JSON enclosed in Markdown fences.
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

            if not isinstance(plan.get("requirements"), list):
                raise ValueError("Requirements must be a list")

            if not isinstance(plan.get("search_queries"), list):
                raise ValueError("Search queries must be a list")

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

            # Avoid silently treating a failed plan as casual chat.
            return {
                "intent": "research",
                "sources": "both",
                "requirements": [
                    f"Find reliable evidence needed to answer: {query}"
                ],
                "search_queries": [query],
            }
