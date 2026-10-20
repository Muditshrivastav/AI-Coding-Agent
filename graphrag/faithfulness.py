"""
graphrag/faithfulness.py - Anti-hallucination verification using RAGAS and citation rules.

Implements:
1. Grounded context formatting requiring [path:line] citations.
2. Explicit "insufficient context" instruction.
3. RAGAS faithfulness checking to verify generated claims against retrieved context.
4. Automatic query narrowing and retry loop when hallucination is detected.

Fixes (B9):
- RAGAS exceptions are now logged at WARNING level (not silently swallowed).
- "insufficient context" in the answer no longer short-circuits to score=1.0; the
  heuristic falls through so any *additional* invented symbols are still caught.
- Heuristic uses word-boundary (\\b) matching, not substring, so `validate_token`
  cannot pass because the context contains `validate_token_expiry`.
- _ast_symbol_check() provides a deterministic fallback: parses the generated code
  with `ast` and verifies each called/imported name exists in the retrieved context
  symbol set, which is far more accurate than substring matching.
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
            # B9: return 0.0 if key missing, not 1.0 (was a silent pass-through)
            score = float(res["faithfulness"]) if "faithfulness" in res else 0.0
            passed = score >= self.threshold
            return {
                "score": score,
                "passed": passed,
                "unsupported_claims": [] if passed else ["Unsupported statements detected by RAGAS judge"],
            }
        except Exception as ragas_exc:
            # B9: log at WARNING so RAGAS mis-configs surface visibly
            logger.warning(
                f"[faithfulness] RAGAS evaluation failed ({ragas_exc}). "
                "Falling back to deterministic heuristic check."
            )
            return self._heuristic_groundedness_check(retrieved_contexts, generated_answer)

    def _heuristic_groundedness_check(
        self,
        retrieved_contexts: list[str],
        generated_answer: str,
    ) -> dict[str, Any]:
        """
        Checks that code references in the answer exist in the retrieved context.

        B9 fixes:
        - Does NOT short-circuit to score=1.0 on "insufficient context" alone.
          The model might say that AND still invent function names.
        - Uses word-boundary regex (\\b) instead of `in` substring check so
          `validate_token` cannot match `validate_token_expiry`.
        - Delegates to _ast_symbol_check() first when the answer looks like Python code.
        """
        all_context = "\n".join(retrieved_contexts)

        # If the model explicitly flagged insufficient context AND invented nothing extra,
        # we score that as a pass (model behaved correctly).
        declared_insufficient = "insufficient context" in generated_answer.lower()

        # Try the deterministic AST-based check first (more accurate than regex).
        ast_result = self._ast_symbol_check(generated_answer, all_context)
        if ast_result is not None:
            if declared_insufficient and not ast_result["unsupported_claims"]:
                # Model correctly flagged gap and invented nothing.
                return {"score": 1.0, "passed": True, "unsupported_claims": []}
            return ast_result

        # Fallback regex heuristic
        # B9: use word-boundary matching to avoid substring false-positives.
        called_symbols = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", generated_answer))
        ignored = {
            "def", "class", "return", "if", "for", "while", "import", "from",
            "print", "len", "range", "str", "int", "dict", "list", "set", "super",
            "isinstance", "hasattr", "getattr", "setattr", "type", "repr", "bool",
        }
        testable_symbols = [s for s in called_symbols if s not in ignored and len(s) > 2]

        unsupported = []
        for sym in testable_symbols:
            # B9: word-boundary match — 'run' must not match 'runner'
            if not re.search(rf"\b{re.escape(sym)}\b", all_context):
                unsupported.append(sym)

        if declared_insufficient and not unsupported:
            return {"score": 1.0, "passed": True, "unsupported_claims": []}

        total = len(testable_symbols)
        if total == 0:
            # No verifiable symbols: give benefit of the doubt only if model flagged gap
            score = 1.0 if declared_insufficient else 0.5
        else:
            supported = total - len(unsupported)
            score = supported / total

        passed = score >= self.threshold
        return {
            "score": round(score, 3),
            "passed": passed,
            "unsupported_claims": unsupported,
        }

    def _ast_symbol_check(
        self,
        generated_answer: str,
        all_context: str,
    ) -> dict[str, Any] | None:
        """
        Deterministic faithfulness check: parse the generated answer with ast,
        extract all called and imported names, and verify each one exists (as a
        word-boundary match) in the concatenated retrieved context.

        Returns None if the answer does not look like valid Python code, so the
        caller can fall back to the regex heuristic.
        """
        # Only attempt to parse if the answer looks like Python code
        code_block = generated_answer
        # Strip markdown fences if present
        fence_match = re.search(r"```(?:python)?\n?(.*?)```", generated_answer, re.DOTALL)
        if fence_match:
            code_block = fence_match.group(1)

        try:
            import ast as _ast
            tree = _ast.parse(code_block)
        except SyntaxError:
            return None  # Not parseable Python; fall back to regex heuristic

        # Collect called function names
        called: set[str] = set()
        imported: set[str] = set()
        ignored = {
            "def", "class", "return", "if", "for", "while", "import", "from",
            "print", "len", "range", "str", "int", "dict", "list", "set", "super",
            "isinstance", "hasattr", "getattr", "setattr", "type", "repr", "bool",
        }

        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name) and node.func.id not in ignored:
                    called.add(node.func.id)
                elif isinstance(node.func, _ast.Attribute):
                    called.add(node.func.attr)
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, _ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
                for alias in node.names:
                    imported.add(alias.name)

        testable = [s for s in (called | imported) if len(s) > 2]
        unsupported = [
            s for s in testable
            if not re.search(rf"\b{re.escape(s)}\b", all_context)
        ]

        total = len(testable)
        if total == 0:
            return {"score": 1.0, "passed": True, "unsupported_claims": []}

        score = (total - len(unsupported)) / total
        return {
            "score": round(score, 3),
            "passed": score >= self.threshold,
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
