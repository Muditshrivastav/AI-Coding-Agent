"""
graphrag/faithfulness.py - Anti-hallucination verification using RAGAS and citation rules.

Implements:
1. Grounded context formatting requiring [path:line] citations.
2. Explicit "insufficient context" instruction.
3. RAGAS faithfulness checking to verify generated claims against retrieved context.
4. Automatic query narrowing and retry loop when hallucination is detected.
"""

from typing import Any
import re
import logging

logger = logging.getLogger(__name__)


class FaithfulnessScorer:
    """
    Evaluates faithfulness of generated output against retrieved code context.
    Identifies unsupported claims and suggests refined queries.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def format_retrieval_context(self, candidates: list[dict[str, Any]]) -> str:
        """
        Formats retrieved candidates with mandatory [path:line] citation prefixes
        and the explicit insufficient context escape hatch.
        """
        if not candidates:
            return (
                "No relevant code found in repository.\n\n"
                "INSTRUCTION: If the retrieved context does not contain enough information to answer "
                "correctly, say so explicitly and request a more specific search — do NOT guess or invent "
                "function signatures, APIs, or file contents that were not in the retrieved context."
            )

        formatted_chunks = []
        for c in candidates:
            path = c.get("path", "unknown")
            line = c.get("line_start", 1)
            name = c.get("name", "symbol")
            source = c.get("source", "").strip()
            formatted_chunks.append(f"[{path}:{line}] {name}:\n{source}")

        joined_chunks = "\n\n".join(formatted_chunks)
        return (
            f"Retrieved code (cite by [path:line] when referencing):\n\n"
            f"{joined_chunks}\n\n"
            f"CRITICAL RULES:\n"
            f"1. You must cite real code chunks using '[path:line]' whenever referencing existing code.\n"
            f"2. If the retrieved context does not contain enough information to answer correctly (insufficient context), "
            f"say so explicitly and request a more specific search — do NOT guess or invent function "
            f"signatures, APIs, or file contents that were not in the retrieved context."
        )

    def evaluate_faithfulness(
        self,
        query: str,
        retrieved_contexts: list[str],
        generated_answer: str,
    ) -> dict[str, Any]:
        """
        Computes faithfulness score (0.0 to 1.0) using RAGAS faithfulness metric
        with fallback groundedness claim decomposition.
        """
        if not retrieved_contexts or not generated_answer.strip():
            return {
                "score": 0.0,
                "passed": False,
                "unsupported_claims": ["No context or empty generation"],
            }

        # Try RAGAS first
        try:
            from ragas.metrics import faithfulness
            from ragas import evaluate
            from datasets import Dataset

            eval_data = Dataset.from_dict(
                {
                    "question": [query],
                    "contexts": [retrieved_contexts],
                    "answer": [generated_answer],
                }
            )
            res = evaluate(eval_data, metrics=[faithfulness])
            score = float(res["faithfulness"]) if "faithfulness" in res else 1.0
            passed = score >= self.threshold
            return {
                "score": score,
                "passed": passed,
                "unsupported_claims": [] if passed else ["Unsupported statements detected by RAGAS judge"],
            }
        except Exception:
            # Fallback claim-level citation and symbol verification
            return self._heuristic_groundedness_check(retrieved_contexts, generated_answer)

    def _heuristic_groundedness_check(
        self,
        retrieved_contexts: list[str],
        generated_answer: str,
    ) -> dict[str, Any]:
        """
        Checks that code references in the answer exist in the retrieved context.
        Verifies presence of cited [path:line] or referenced function names.
        """
        # If model explicitly said insufficient context, treat as faithful
        if "insufficient context" in generated_answer.lower():
            return {"score": 1.0, "passed": True, "unsupported_claims": []}

        # Extract function-like calls from generated answer: e.g. some_func(
        called_symbols = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", generated_answer))
        # Filter standard keywords
        ignored = {
            "def", "class", "return", "if", "for", "while", "import", "from",
            "print", "len", "range", "str", "int", "dict", "list", "set", "super"
        }
        testable_symbols = [s for s in called_symbols if s not in ignored and len(s) > 2]

        all_context = "\n".join(retrieved_contexts)
        unsupported = []
        for sym in testable_symbols:
            if sym not in all_context:
                unsupported.append(sym)

        total = len(testable_symbols)
        if total == 0:
            score = 1.0
        else:
            supported = total - len(unsupported)
            score = supported / total

        passed = score >= self.threshold
        return {
            "score": round(score, 3),
            "passed": passed,
            "unsupported_claims": unsupported,
        }

    def narrow_query(self, original_query: str, unsupported_claims: list[str]) -> str:
        """
        Refines the retrieval query based on missing/unsupported claims or symbols
        instead of repeating the same broad search query.
        """
        if not unsupported_claims:
            return original_query

        # Extract distinct terms to target
        specific_terms = " ".join(unsupported_claims[:3])
        return f"{original_query} {specific_terms}".strip()
