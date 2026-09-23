"""
tests/test_graphrag.py - Unit tests for GraphRAG package components.
Tests:
- AST code-aware chunking (boundaries, classes, functions, calls, imports, line ranges)
- LlamaIndex GitHub reader integration and local repository ingestion
- Neo4j Graph + Vector store (graph relationships, vectors, Layer 3 metadata filters)
- CrossEncoder reranking and relevance ordering
- Faithfulness scoring, [path:line] grounded citation formatting, and query narrowing
- Full GraphRAG pipeline and LangChain retriever tool
"""

import pytest
from unittest.mock import MagicMock, patch

from graphrag.ast_parser import ASTCodeParser, CodeChunk
from graphrag.config import GraphRAGConfig
from graphrag.embedder import DefaultEmbedder
from graphrag.faithfulness import FaithfulnessScorer
from graphrag.github_reader import GitHubRepoIngester
from graphrag.pipeline import GraphRAGPipeline
from graphrag.reranker import CrossEncoderReranker
from graphrag.store import Neo4jGraphStore
from graphrag.tool import create_graphrag_retriever_tool


SAMPLE_PYTHON_CODE = '''import os
from math import sqrt

class PaymentProcessor:
    """Handles payments for customers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def charge_customer(self, customer_id: str, amount: float) -> bool:
        if amount <= 0:
            return False
        return self._execute_charge(customer_id, amount)

    def _execute_charge(self, customer_id: str, amount: float) -> bool:
        print(f"Charging {customer_id} ${amount}")
        return True

def refund_customer(customer_id: str, amount: float) -> bool:
    """Issues a refund."""
    return PaymentProcessor("test").charge_customer(customer_id, -amount)
'''


def test_ast_code_aware_chunking():
    parser = ASTCodeParser(project_id="test_proj", commit_sha="abc1234")
    chunks = parser.parse_python(SAMPLE_PYTHON_CODE, "billing/payments.py")

    assert len(chunks) >= 4  # Class, 3 methods/functions

    # Find the Class chunk
    class_chunks = [c for c in chunks if c.chunk_type == "Class" and c.name == "PaymentProcessor"]
    assert len(class_chunks) == 1
    c_chunk = class_chunks[0]
    assert c_chunk.path == "billing/payments.py"
    assert c_chunk.project_id == "test_proj"
    assert "class PaymentProcessor:" in c_chunk.source
    assert "math.sqrt" in c_chunk.imports or "os" in c_chunk.imports

    # Find charge_customer function
    func_chunks = [c for c in chunks if "charge_customer" in c.name and c.chunk_type == "Function"]
    assert len(func_chunks) >= 1
    charge_func = func_chunks[0]
    assert charge_func.line_start < charge_func.line_end
    # Code aware boundary: the function is not cut in half
    assert "def charge_customer" in charge_func.source
    assert "return self._execute_charge" in charge_func.source
    assert "_execute_charge" in charge_func.calls

    # Standalone refund_customer function
    refund_chunks = [c for c in chunks if c.name == "refund_customer"]
    assert len(refund_chunks) == 1
    assert "PaymentProcessor" in refund_chunks[0].calls or "charge_customer" in refund_chunks[0].calls


def test_embedder_deterministic_and_dimensions():
    embedder = DefaultEmbedder(dimension=128)
    vec1 = embedder.embed_text("def charge_customer(id, amount):")
    vec2 = embedder.embed_text("def charge_customer(id, amount):")
    vec3 = embedder.embed_text("import unrelated_module")

    assert len(vec1) == embedder.dimension
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Different tokens yield different vector


def test_store_metadata_filtering_and_similarity():
    store = Neo4jGraphStore()  # Uses in-memory simulation if Neo4j is not locally running
    embedder = DefaultEmbedder(dimension=64)

    chunk1 = CodeChunk(
        name="charge_customer",
        chunk_type="Function",
        path="billing/payments.py",
        source="def charge_customer(customer_id, amount): ...",
        line_start=10,
        line_end=20,
        project_id="billing_proj",
        language="python",
        calls=["_execute_charge"],
    )
    chunk2 = CodeChunk(
        name="handle_login",
        chunk_type="Function",
        path="auth/login.py",
        source="def handle_login(user, pass): ...",
        line_start=5,
        line_end=15,
        project_id="auth_proj",
        language="python",
    )

    embs = embedder.embed_batch([chunk1.source, chunk2.source])
    store.store_chunks([chunk1, chunk2], embs)

    # Search with project_id metadata filter
    query_emb = embedder.embed_text("charge customer amount")
    res_billing = store.query_similar_nodes(
        query_embedding=query_emb,
        project_id="billing_proj",
        top_k=5,
    )
    assert len(res_billing) == 1
    assert res_billing[0]["name"] == "charge_customer"

    # Search with different project_id
    res_auth = store.query_similar_nodes(
        query_embedding=query_emb,
        project_id="auth_proj",
        top_k=5,
    )
    assert len(res_auth) == 1
    assert res_auth[0]["name"] == "handle_login"

    # Search with module_scope prefix filter
    res_scope = store.query_similar_nodes(
        query_embedding=query_emb,
        module_scope="billing/",
        top_k=5,
    )
    assert len(res_scope) == 1
    assert res_scope[0]["path"] == "billing/payments.py"


def test_reranker():
    reranker = CrossEncoderReranker()
    candidates = [
        {"name": "unrelated_func", "source": "def unrelated_task(): pass", "score": 0.8},
        {"name": "charge_customer", "source": "def charge_customer(customer, amount): process_payment()", "score": 0.75},
    ]
    query = "charge customer payment"
    ranked = reranker.rerank(query, candidates, top_k=2)

    assert len(ranked) == 2
    # charge_customer should be boosted/ranked #1 for query matching
    assert ranked[0]["name"] == "charge_customer"
    assert "rerank_score" in ranked[0]


def test_faithfulness_and_citation_context():
    scorer = FaithfulnessScorer(threshold=0.8)
    candidates = [
        {
            "name": "charge_customer",
            "path": "billing/payments.py",
            "line_start": 42,
            "source": "def charge_customer(customer_id, amount):\n    return True",
        }
    ]

    context = scorer.format_retrieval_context(candidates)
    assert "[billing/payments.py:42] charge_customer:" in context
    assert "CRITICAL RULES:" in context
    assert "insufficient context" in context

    # Test grounded output
    faithful_answer = "According to [billing/payments.py:42], charge_customer takes customer_id and amount."
    res = scorer.evaluate_faithfulness("How to charge customer?", [candidates[0]["source"]], faithful_answer)
    assert res["passed"] is True

    # Test query narrowing on hallucinated claim
    narrowed = scorer.narrow_query("How to charge customer?", ["non_existent_validator", "fake_token_parser"])
    assert "non_existent_validator" in narrowed


def test_github_reader_local_directory(tmp_path):
    # Test local repository ingestion fallback
    test_file = tmp_path / "service.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    ingester = GitHubRepoIngester()
    chunks = ingester.ingest_from_local_directory(str(tmp_path), project_id="test_local")

    assert len(chunks) > 0
    assert any(c.name == "refund_customer" for c in chunks)


def test_graphrag_pipeline_and_tool(tmp_path):
    test_file = tmp_path / "payments.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    pipeline = GraphRAGPipeline()
    stored_count = pipeline.ingest_local_repo(str(tmp_path), project_id="repo_123")
    assert stored_count > 0

    tool = create_graphrag_retriever_tool(pipeline)
    assert tool.name == "retrieve_graphrag_code"

    result_context = tool.invoke({"query": "charge_customer", "project_id": "repo_123"})
    assert "charge_customer" in result_context
    assert "Retrieved code" in result_context
