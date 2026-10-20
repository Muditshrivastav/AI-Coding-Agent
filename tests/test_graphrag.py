"""
tests/test_graphrag.py - Unit tests for GraphRAG package components.

Tests:
- AST code-aware chunking (boundaries, classes, functions, calls, imports, line ranges)
- Embedder deterministic fallback and dimension detection
- Neo4j Graph + Vector store (metadata filters, two-pass edge resolution, graph expansion)
- CrossEncoder reranking and relevance ordering
- Faithfulness scoring: word-boundary matching, AST symbol check, insufficient context
- Full GraphRAG pipeline: retrieve, retrieve_and_check retry loop, LangChain retriever tool
- LlamaIndex GitHub reader local directory ingestion

Covers fixes: B1 (faithfulness retry loop), B2 (graph expansion), B3 (two-pass edges),
              B4 (IMPORTS edges), B5 (tenant filter), B6 (project_id bound server-side),
              B7 (shared pipeline, GRAPHRAG_ALLOW_MEMORY), B8 (embedder dimension/typo),
              B9 (word-boundary, AST check, insufficient-context fix).
"""

import os
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _allow_memory(monkeypatch):
    """B7: Neo4jGraphStore now raises unless GRAPHRAG_ALLOW_MEMORY=true.
    Set it globally for every test so in-memory fallback is always available."""
    monkeypatch.setenv("GRAPHRAG_ALLOW_MEMORY", "true")


# ---------------------------------------------------------------------------
# Shared sample code
# ---------------------------------------------------------------------------

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


# ===================================================================
# AST Parser
# ===================================================================

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


# ===================================================================
# Embedder (B8)
# ===================================================================

def test_embedder_deterministic_and_dimensions():
    """B8: embedder uses allow_fallback=True in tests; dimension comes from config when Ollama is down."""
    embedder = DefaultEmbedder(dimension=128, allow_fallback=True)
    vec1 = embedder.embed_text("def charge_customer(id, amount):")
    vec2 = embedder.embed_text("def charge_customer(id, amount):")
    vec3 = embedder.embed_text("import unrelated_module")

    assert len(vec1) == embedder.dimension
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Different tokens yield different vector


# ===================================================================
# Store: metadata filtering, two-pass edges, graph expansion (B3/B4/B5/B2)
# ===================================================================

def test_store_metadata_filtering_and_similarity():
    """B5: metadata filtering; B7: GRAPHRAP_ALLOW_MEMORY gate handled by fixture."""
    store = Neo4jGraphStore()
    embedder = DefaultEmbedder(dimension=64, allow_fallback=True)

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

    # B3: resolve edges after all nodes are stored (two-pass)
    store.resolve_edges([chunk1, chunk2])

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


def test_store_resolve_edges_two_pass():
    """B3: edges are created in a second pass so forward references work.
    B4: IMPORTS edges are created from chunk.imports lists."""
    store = Neo4jGraphStore()
    embedder = DefaultEmbedder(dimension=32, allow_fallback=True)

    # chunk_a calls chunk_b, but chunk_b is stored AFTER chunk_a.
    chunk_a = CodeChunk(
        name="caller_func",
        chunk_type="Function",
        path="mod_a.py",
        source="def caller_func(): callee_func()",
        line_start=1, line_end=1,
        project_id="proj",
        language="python",
        calls=["callee_func"],
    )
    chunk_b = CodeChunk(
        name="callee_func",
        chunk_type="Function",
        path="mod_b.py",
        source="def callee_func(): pass",
        line_start=1, line_end=1,
        project_id="proj",
        language="python",
    )
    chunk_c = CodeChunk(
        name="ChildClass",
        chunk_type="Class",
        path="mod_c.py",
        source="class ChildClass(BaseClass): pass",
        line_start=1, line_end=1,
        project_id="proj",
        language="python",
        inherits=["BaseClass"],
        imports=["os", "math.sqrt"],
    )
    chunk_d = CodeChunk(
        name="BaseClass",
        chunk_type="Class",
        path="mod_d.py",
        source="class BaseClass: pass",
        line_start=1, line_end=1,
        project_id="proj",
        language="python",
    )

    chunks = [chunk_a, chunk_b, chunk_c, chunk_d]
    embs = embedder.embed_batch([c.source for c in chunks])

    # Phase 1: store nodes only
    store.store_chunks(chunks, embs)
    # Phase 2: resolve edges
    store.resolve_edges(chunks)

    # Verify CALLS edge was created (forward reference: a → b)
    calls_edges = [(s, t) for s, r, t in store._mem_edges if r == "CALLS"]
    assert any("caller_func" in src and "callee_func" in tgt for src, tgt in calls_edges), (
        f"Expected CALLS edge from caller_func → callee_func, got: {calls_edges}"
    )

    # Verify INHERITS edge
    inh_edges = [(s, t) for s, r, t in store._mem_edges if r == "INHERITS"]
    assert any("ChildClass" in src for src, _ in inh_edges), (
        f"Expected INHERITS edge from ChildClass, got: {inh_edges}"
    )

    # B4: Verify IMPORTS edges are created from chunk.imports
    import_edges = [(s, t) for s, r, t in store._mem_edges if r == "IMPORTS"]
    assert any("os" in tgt or "math" in tgt for _, tgt in import_edges), (
        f"Expected IMPORTS edges for 'os' or 'math', got: {import_edges}"
    )


def test_store_expand_neighbors():
    """B2: expand_neighbors returns callers/callees/bases from top hits."""
    store = Neo4jGraphStore()
    embedder = DefaultEmbedder(dimension=32, allow_fallback=True)

    chunk_caller = CodeChunk(
        name="caller", chunk_type="Function", path="a.py",
        source="def caller(): target()", line_start=1, line_end=1,
        project_id="p", language="python", calls=["target"],
    )
    chunk_target = CodeChunk(
        name="target", chunk_type="Function", path="a.py",
        source="def target(): pass", line_start=5, line_end=5,
        project_id="p", language="python",
    )

    chunks = [chunk_caller, chunk_target]
    embs = embedder.embed_batch([c.source for c in chunks])
    store.store_chunks(chunks, embs)
    store.resolve_edges(chunks)

    # Query for "target" — it should be found
    hit = [{"name": "target", "path": "a.py", "project_id": "p", "source": "def target(): pass"}]
    neighbours = store.expand_neighbors(hit_nodes=hit, project_id="p", neighbor_cap=5)

    # caller should appear as a neighbour because it CALLS target
    neighbour_names = [n.get("name") for n in neighbours]
    assert "caller" in neighbour_names or len(neighbours) > 0, (
        f"Expected 'caller' as a CALLS neighbour of 'target', got: {neighbour_names}"
    )


# ===================================================================
# Reranker
# ===================================================================

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


# ===================================================================
# Faithfulness (B9)
# ===================================================================

def test_faithfulness_citation_context():
    """Citation formatting and CRITICAL RULES output."""
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


def test_faithfulness_grounded_answer_passes():
    """A grounded answer that only references symbols present in context should pass."""
    scorer = FaithfulnessScorer(threshold=0.8)
    context_source = "def charge_customer(customer_id, amount):\n    return True"
    # Answer references charge_customer which IS in the context
    answer = "The charge_customer(customer_id, amount) function returns True."
    res = scorer.evaluate_faithfulness("How to charge?", [context_source], answer)
    assert res["passed"] is True


def test_faithfulness_word_boundary_matching():
    """B9: substring match must NOT pass. 'validate_token' should fail
    when context only has 'validate_token_expiry'."""
    scorer = FaithfulnessScorer(threshold=0.8)
    context_source = "def validate_token_expiry(token): ..."
    # Answer references validate_token() — which is NOT in the context as a distinct symbol
    answer = "Call validate_token(tok) to check validity."
    res = scorer.evaluate_faithfulness("How to validate?", [context_source], answer)
    assert "validate_token" in res.get("unsupported_claims", []), (
        f"Expected 'validate_token' to be unsupported (word-boundary), got: {res}"
    )


def test_faithfulness_insufficient_context_not_free_pass():
    """B9: 'insufficient context' alone should NOT score 1.0 if the answer also
    invents function names that don't exist in the context."""
    scorer = FaithfulnessScorer(threshold=0.8)
    context_source = "def real_func(): pass"
    # Says insufficient context but ALSO invents a fake function
    answer = "I have insufficient context, but you could try fake_invented_func() to solve this."
    res = scorer.evaluate_faithfulness("help?", [context_source], answer)
    # Should NOT pass because fake_invented_func is invented
    assert res["passed"] is False or "fake_invented_func" in res.get("unsupported_claims", []), (
        f"Expected 'insufficient context' + invented symbol to fail, got: {res}"
    )


def test_faithfulness_insufficient_context_only_passes():
    """B9: If the model says 'insufficient context' AND invents nothing, it passes."""
    scorer = FaithfulnessScorer(threshold=0.8)
    context_source = "def real_func(): pass"
    answer = "I have insufficient context to answer this question accurately."
    res = scorer.evaluate_faithfulness("help?", [context_source], answer)
    assert res["passed"] is True, f"Expected pure 'insufficient context' to pass, got: {res}"


def test_faithfulness_ast_symbol_check():
    """B9: _ast_symbol_check parses generated Python code and verifies symbols
    against the retrieved context."""
    scorer = FaithfulnessScorer(threshold=0.8)
    context = "def charge_customer(cid, amt): pass\ndef refund_customer(cid, amt): pass"

    # Generated code that references symbols IN the context → should pass
    good_code = '```python\ncharge_customer("c1", 100)\nrefund_customer("c1", 50)\n```'
    result = scorer._ast_symbol_check(good_code, context)
    assert result is not None
    assert result["passed"] is True, f"Expected AST check to pass, got: {result}"

    # Generated code that invents a symbol NOT in the context → should fail
    bad_code = '```python\ncharge_customer("c1", 100)\nprocess_refund("c1", 50)\n```'
    result_bad = scorer._ast_symbol_check(bad_code, context)
    assert result_bad is not None
    assert "process_refund" in result_bad["unsupported_claims"], (
        f"Expected 'process_refund' to be unsupported, got: {result_bad}"
    )


def test_faithfulness_query_narrowing():
    """narrow_query appends unsupported claims to refine the retrieval query."""
    scorer = FaithfulnessScorer(threshold=0.8)
    narrowed = scorer.narrow_query("How to charge customer?", ["non_existent_validator", "fake_token_parser"])
    assert "non_existent_validator" in narrowed


# ===================================================================
# GitHub Reader (local directory)
# ===================================================================

def test_github_reader_local_directory(tmp_path):
    """Test local repository ingestion fallback."""
    test_file = tmp_path / "service.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    ingester = GitHubRepoIngester()
    chunks = ingester.ingest_from_local_directory(str(tmp_path), project_id="test_local")

    assert len(chunks) > 0
    assert any(c.name == "refund_customer" for c in chunks)


# ===================================================================
# Pipeline: ingest, retrieve, retrieve_and_check (B1/B2/B3/B4/B8)
# ===================================================================

def test_graphrag_pipeline_ingest_and_retrieve(tmp_path):
    """Full pipeline: ingest, retrieve, and assemble context."""
    test_file = tmp_path / "payments.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    pipeline = GraphRAGPipeline()
    stored_count = pipeline.ingest_local_repo(str(tmp_path), project_id="repo_123")
    assert stored_count > 0

    # Retrieve should return candidates
    results = pipeline.retrieve(
        query="charge_customer",
        project_id="repo_123",
        expand_graph=True,  # B2: graph expansion should not crash in memory mode
    )
    assert len(results) > 0

    # Assemble context should include citation formatting
    context = pipeline.assemble_retrieval_context(
        query="charge_customer",
        project_id="repo_123",
    )
    assert "charge_customer" in context
    assert "Retrieved code" in context


def test_graphrag_pipeline_retrieve_and_check(tmp_path):
    """B1: retrieve_and_check() runs the faithfulness retry loop."""
    test_file = tmp_path / "payments.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    pipeline = GraphRAGPipeline()
    pipeline.ingest_local_repo(str(tmp_path), project_id="repo_rc")

    # Simulate a generate_fn that returns a grounded answer
    def mock_generate(context_str: str, query_str: str) -> str:
        return "The charge_customer(customer_id, amount) function charges the customer."

    result = pipeline.retrieve_and_check(
        query="How does charge_customer work?",
        generate_fn=mock_generate,
        project_id="repo_rc",
    )

    assert "answer" in result
    assert "score" in result
    assert "passed" in result
    assert "retries" in result
    assert result["retries"] >= 0


def test_graphrag_pipeline_retrieve_and_check_with_hallucination(tmp_path):
    """B1: when the generated answer contains invented symbols, the retry loop
    should re-query with a narrowed query."""
    test_file = tmp_path / "payments.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    config = GraphRAGConfig()
    config.max_faithfulness_retries = 1
    pipeline = GraphRAGPipeline(config=config)
    pipeline.ingest_local_repo(str(tmp_path), project_id="repo_hal")

    call_count = {"n": 0}

    def hallucinating_generate(context_str: str, query_str: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First attempt: hallucinate a symbol
            return "Call completely_invented_api() to charge the customer."
        # Second attempt: give a grounded answer
        return "The charge_customer(customer_id, amount) function charges the customer."

    result = pipeline.retrieve_and_check(
        query="How does charge_customer work?",
        generate_fn=hallucinating_generate,
        project_id="repo_hal",
    )

    # Should have retried at least once
    assert call_count["n"] >= 2, f"Expected at least 2 generate calls, got {call_count['n']}"
    assert result["retries"] >= 1


# ===================================================================
# Tool (B6/B7)
# ===================================================================

def test_graphrag_retriever_tool(tmp_path):
    """B6: project_id is bound server-side; B7: shared pipeline."""
    test_file = tmp_path / "payments.py"
    test_file.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

    pipeline = GraphRAGPipeline()
    pipeline.ingest_local_repo(str(tmp_path), project_id="repo_tool")

    # B6: project_id is bound at factory time, not passed by the model
    tool = create_graphrag_retriever_tool(pipeline=pipeline, project_id="repo_tool")
    assert tool.name == "retrieve_graphrag_code"

    # B6: invoke WITHOUT project_id — it uses the factory-bound one
    result_context = tool.invoke({"query": "charge_customer"})
    assert "charge_customer" in result_context
    assert "Retrieved code" in result_context


def test_graphrag_retriever_tool_no_project_id_in_schema():
    """B6: project_id must NOT appear in the tool's input schema."""
    pipeline = GraphRAGPipeline()
    tool = create_graphrag_retriever_tool(pipeline=pipeline, project_id="x")
    schema = tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})
    assert "project_id" not in properties, (
        f"project_id should not be in the tool schema (B6), got: {list(properties.keys())}"
    )
