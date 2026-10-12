
from openai import OpenAI
from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
)


class LLMClient:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Check your .env file."
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = OPENROUTER_MODEL

    def generate(self, messages, max_tokens=2000, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort="low",
        )

        choice = response.choices[0]
        answer = choice.message.content

        if not answer or not answer.strip():
            raise RuntimeError(
            "Model returned no final answer. "
            f"Finish reason: {choice.finish_reason}. "
            f"Usage: {response.usage}"
        )

        if choice.finish_reason == "length":
            raise RuntimeError("Model output reached the token limit.")

        return answer.strip()
