
import json
import logging

from app.llm import LLMClient

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.llm = LLMClient()

    def evaluate(self, requirements: list, evidence: list) -> dict:
        if not requirements:
            return {
                "sufficient": True,
                "covered": [],
                "missing": [],
                "reason": "No evidence requirements were specified.",
            }

        if not evidence:
            return {
                "sufficient": False,
                "covered": [],
                "missing": requirements,
                "reason": "No evidence was retrieved.",
            }

        evidence_text = "\n\n".join(
            f"Source: {item.get('title', item.get('source', 'Unknown'))}\n"
            f"Content: {item.get('text', '')}"
            for item in evidence
        )

        prompt = f"""
Evaluate whether the supplied evidence adequately covers each
information requirement.

Requirements:
{json.dumps(requirements, ensure_ascii=False)}

Evidence:
{evidence_text}

Return ONLY valid JSON with this structure:
{{
  "sufficient": false,
  "covered": ["requirements adequately supported by evidence"],
  "missing": ["requirements not adequately supported"],
  "reason": "brief explanation"
}}

Rules:
- A requirement is covered only when the evidence supports it.
- Do not assume missing facts.
- Mark a requirement missing when evidence is irrelevant,
  contradictory, or too weak.
- "sufficient" must be true only when every requirement is covered.
- Use the exact requirement text in covered and missing.
"""

        try:
            response = self.llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evidence evaluator. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3000,
                temperature=0.1,
            )

            # Support responses wrapped in Markdown fences.
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0].strip()

            result = json.loads(response)

            covered = result.get("covered", [])
            missing = result.get("missing", [])
            sufficient = result.get("sufficient")

            if (
                not isinstance(covered, list)
                or not isinstance(missing, list)
                or not isinstance(sufficient, bool)
            ):
                raise ValueError("Invalid evaluator response structure")

            # Fail closed: every requirement must be explicitly covered.
            valid_covered = [
                requirement
                for requirement in requirements
                if requirement in covered
            ]
            valid_missing = [
                requirement
                for requirement in requirements
                if requirement not in valid_covered
            ]

            sufficient = sufficient and not valid_missing

            return {
                "sufficient": sufficient,
                "covered": valid_covered,
                "missing": valid_missing,
                "reason": str(result.get("reason", "")),
            }

        except Exception as error:
            logger.warning("Evidence evaluation failed: %s", error)

            # Do not claim evidence is sufficient when evaluation fails.
            return {
                "sufficient": False,
                "covered": [],
                "missing": requirements,
                "reason": "Evaluation failed; evidence sufficiency is unverified.",
            }
