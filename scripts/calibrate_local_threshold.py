"""Print top local similarity scores to help choose LOCAL_MIN_SCORE."""
from app.retriever import Retriever

# Questions your documents DO answer (replace with real ones).
IN_DOCUMENT = [
    "replace with a question your documents answer",
]

# Questions your documents clearly do NOT answer.
OFF_TOPIC = [
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "Who won the last football world cup?",
]


def top_scores(retriever, query):
    hits = retriever.retrieve_local(query, top_k=3, min_score=0.0)
    return [round(hit["score"], 3) for hit in hits]


if __name__ == "__main__":
    with Retriever() as retriever:
        for label, questions in (
            ("IN-DOCUMENT", IN_DOCUMENT),
            ("OFF-TOPIC", OFF_TOPIC),
        ):
            print(f"\n== {label} ==")
            for question in questions:
                print(f"{top_scores(retriever, question)}  {question}")