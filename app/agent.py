
import logging

from app.planner import Planner
from app.retriever import Retriever
from app.evaluator import Evaluator
from app.generator import Generator

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, max_rounds: int = 3):
        self.planner = Planner()
        self.evaluator = Evaluator()
        self.generator = Generator()
        self._retriever = None
        self.max_rounds = max(1, max_rounds)

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    @staticmethod
    def merge_evidence(existing: list, new_items: list) -> list:
        """Merge evidence while removing exact duplicate chunks."""
        merged = list(existing)
        seen = {
            (
                item.get("source", item.get("title", "")),
                " ".join(item.get("text", "").lower().split()),
            )
            for item in existing
        }

        for item in new_items:
            text = " ".join(item.get("text", "").lower().split())
            source = item.get("source", item.get("title", ""))
            key = (source, text)

            if text and key not in seen:
                merged.append(item)
                seen.add(key)

        return merged

    def run(
        self,
        query: str,
        chat_history: list | None = None,
    ) -> dict:
        """Run the Agentic RAG workflow."""
        query = query.strip()
        chat_history = chat_history or []

        if not query:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "status": "invalid_input",
                "rounds": 0,
            }

        # 1. Plan
        plan = self.planner.plan(query)

        source = plan["sources"]
        requirements = plan["requirements"]
        search_queries = plan["search_queries"] or [query]

        logger.info(
            "Intent: %s | Initial route: %s",
            plan["intent"],
            source,
        )

        # 2. Direct chat skips retrieval and evaluation.
        if source == "none":
            result = self.generator.generate(
                query=query,
                evidence=[],
                chat_history=chat_history,
            )
            return {
                **result,
                "status": "answered",
                "rounds": 0,
            }

        evidence = []
        evaluation = None
        rounds_used = 0

        # 3. Retrieve -> evaluate -> refine.
        for round_number in range(1, self.max_rounds + 1):
            rounds_used = round_number

            logger.info(
                "Retrieval round %s/%s using route %s",
                round_number,
                self.max_rounds,
                source,
            )

            for search_query in search_queries:
                try:
                    new_evidence = self.retriever.retrieve(
                        query=search_query,
                        source=source,
                        top_k=5,
                    )
                    evidence = self.merge_evidence(
                        evidence,
                        new_evidence,
                    )
                except Exception:
                    logger.exception(
                        "Retrieval failed for query: %s",
                        search_query,
                    )

            logger.info("Evidence items collected: %s", len(evidence))

            # All retrieval routes converge on evaluation.
            evaluation = self.evaluator.evaluate(
                requirements=requirements,
                evidence=evidence,
            )

            logger.info(
                "Evidence sufficient: %s",
                evaluation["sufficient"],
            )

            if evaluation["sufficient"]:
                break

            missing = evaluation["missing"]
            if not missing or round_number == self.max_rounds:
                break

            # Target the next retrieval round at missing requirements.
            search_queries = [
                f"{requirement}. Original question: {query}"
                for requirement in missing
            ]

            # If local evidence is insufficient, try web search.
            if source == "local":
                source = "web"

        # 4. Generate the final answer.
        result = self.generator.generate(
            query=query,
            evidence=evidence,
            chat_history=chat_history,
        )

        sufficient = bool(
            evaluation and evaluation["sufficient"]
        )

        if not sufficient:
            if evidence:
                warning = (
                    "\n\n**Evidence limitation:** The retrieved "
                    "evidence did not sufficiently cover all "
                    "requirements. Some details may remain unverified."
                )
            else:
                warning = (
                    "\n\n**Evidence limitation:** No usable evidence "
                    "was retrieved. This answer could not be verified "
                    "against the selected sources."
                )

            result["answer"] += warning

        return {
            **result,
            "status": (
                "answered" if sufficient
                else "insufficient_evidence"
            ),
            "rounds": rounds_used,
        }


if __name__ == "__main__":
    agent = Agent()
    history = []

    print("Agentic RAG Chatbot")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            try:
                response = agent.run(
                    query=question,
                    chat_history=history,
                )

                print(f"\nAssistant: {response['answer']}")

                if response["sources"]:
                    print("\nSources:")
                    for item in response["sources"]:
                        print(
                            f"- {item['title']}"
                            + (
                                f": {item['url']}"
                                if item.get("url")
                                else ""
                            )
                        )

                print(f"\nStatus: {response['status']}")
                print(f"Rounds: {response['rounds']}\n")

                history.extend([
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": response["answer"],
                    },
                ])
                history = history[-12:]

            except Exception:
                logger.exception("Agent request failed")
                print("Sorry, the chatbot encountered an error.\n")

    finally:
        if agent._retriever is not None:
            agent._retriever.close()
