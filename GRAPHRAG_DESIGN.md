# GraphRAG Design — Coding Agent Harness

**Purpose**: Defines how a coding agent retrieves relevant code from a GitHub repository before generating new code, using a graph database plus vector search, with explicit measures to prevent hallucinated (invented, non-existent) code references.

---

## 0. System architecture overview

The system is two independent pipelines sharing one store. The **ingestion pipeline** builds and maintains the graph; the **retrieval and generation pipeline** queries it per task. Neither runs the other — they only meet at the Neo4j store.

```
┌─────────────────────────── INGESTION PIPELINE ───────────────────────────┐
│                                                                            │
│  [1] GitHub repository        Source, per connected project              │
│           ↓                                                               │
│  [2] Clone / incremental sync Full checkout (build) or                   │
│                                GithubRepositoryReader (ingestion-only)    │
│           ↓                                                               │
│  [3] AST parser                Finds exact code boundaries                │
│           ↓                                                               │
│  [4] Graph + embedding builder Builds nodes, edges, vectors               │
│           ↓                                                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                     ↓
                    ┌────────────────────────────────┐
                    │  [5] NEO4J STORE                │
                    │      Graph + vectors + metadata │
                    │      (the shared hub)            │
                    └────────────────────────────────┘
                                     ↓
┌───────────────────────────── RETRIEVAL + GENERATION PIPELINE ────────────┐
│                                                                            │
│  [6] Task / todo query          Current todo item's need                 │
│           ↓                                                               │
│  [7] Metadata filter            Scopes by project and language           │
│           ↓                                                               │
│  [8] Vector search               Top ~20 similar candidates              │
│           ↑                       ↓                                       │
│           │  (loop on fail)  [9] Reranker            Top ~5, relevant    │
│           │                       ↓                                       │
│           │                  [10] Context assembly    Citations required │
│           │                       ↓                                       │
│           │                  [11] LLM generation       Cites or flags gap│
│           │                       ↓                                       │
│           └──────────────── [12] Faithfulness check   RAGAS groundedness │
│                                    ↓ passes                               │
│                               [13] Verification         Lint, test, ship │
└────────────────────────────────────────────────────────────────────────┘
```

**Component-to-section map**:

| # | Component | Detailed in |
|---|---|---|
| 1–2 | Repository acquisition (clone / incremental sync) | §3 |
| 3 | AST parser | §2 |
| 4 | Graph + embedding builder | §3 |
| 5 | Neo4j store | §3 |
| 6 | Task / todo query | §7 |
| 7 | Metadata filter | §4 |
| 8 | Vector search | §4, §7 |
| 9 | Reranker | §5 |
| 10 | Context assembly (citations) | §6a |
| 11 | LLM generation (cite-or-flag-gap rule) | §6b |
| 12 | Faithfulness check (RAGAS) | §6c |
| 13 | Verification (final backstop) | §6d |

The one loop in the system: a failed faithfulness check (12) returns to vector search (8) with a narrowed query — never straight back to generation with the same candidates, and never forward to verification with unfaithful output.

---

## 1. Why not chunk code like a text document

Standard RAG splits text every N characters or tokens. For code, this is actively harmful — it can cut a function in half, separate a function from the class it belongs to, or split an import from the code that uses it. A retrieved chunk that's incomplete forces the model to guess at the missing part, which is a direct source of hallucination.

**The fix — code-aware chunking**: a chunk is always a complete, meaningful unit — one whole function, one whole class, one whole module-level block. Never a partial one.

---

## 2. AST — finding the correct chunk boundaries

**AST = Abstract Syntax Tree.** It's what you get by parsing code as code, rather than treating it as plain text. Every function, class, import, and function call becomes a labeled node in a tree structure.

```python
import ast

source = open("payments.py").read()
tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        print(node.name, node.lineno)  # exact function name and its start line
```

This is what lets the system say "retrieve exactly the `charge_customer` function, nothing more, nothing less" — the AST gives exact start/end boundaries, which become the chunk boundaries.

- **Python** → the built-in `ast` module.
- **JavaScript, TypeScript, Go, Java, etc.** → `tree-sitter`, which does the equivalent job across many languages through one consistent interface.

Either way, the output is the same: a tree of exact, labeled code boundaries.

---

## 3. Storing the repository — three layers, one node

Each meaningful code unit (a function, a class) is stored as a single graph node carrying three layers of information at once:

```
Layer 1 — THE GRAPH (structure)
  Nodes: File, Function, Class, Module
  Edges: CONTAINS, CALLS, IMPORTS, INHERITS
  → answers "what depends on what"

Layer 2 — THE VECTOR EMBEDDING (meaning)
  Each Function/Class node also carries an embedding of its own source
  → answers "what's semantically similar to this description"

Layer 3 — METADATA (facts about the chunk)
  project_id, file path, language, commit SHA, node type, line range
  → answers "narrow this down to only what's relevant right now"
```

Example node, in Neo4j:

```cypher
CREATE (f:Function {
  name: "charge_customer",
  path: "billing/payments.py",
  source: "def charge_customer(...): ...",
  embedding: [0.021, -0.045, ...],
  project_id: "proj_123",
  language: "python",
  commit_sha: "def456",
  line_start: 42, line_end: 58
})
```

**Ingestion pipeline**:
1. Acquire repository content — two options, used for different purposes:
   - **`git clone`** — a full local checkout. Required whenever the agent needs a real writable working directory (the build/sandbox lifecycle, where the agent edits files and later runs `git push`).
   - **`llama-index-readers-github`** — pulls file contents directly through the GitHub API, no local checkout. Better fit for the **ingestion-only** path (populating the graph), especially for incremental re-ingestion after a webhook, where only a handful of changed files need to be read and parsed, not a full working directory.

```python
from llama_index.readers.github import GithubRepositoryReader, GithubClient

github_client = GithubClient(github_token=installation_token, verbose=False)

reader = GithubRepositoryReader(
    github_client=github_client,
    owner=owner, repo=repo_name,
    use_parser=False,          # skip LlamaIndex's own chunking — AST does that instead
    filter_file_extensions=(
        [".py", ".js", ".ts", ".md"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_directories=(
        ["node_modules", "dist", ".git"],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)

documents = reader.load_data(branch=branch)  # each Document carries file_path in metadata
```

`use_parser=False` is important: LlamaIndex's own text splitter would chunk by token count — exactly what Section 1 rejects. Keeping each `Document` as one whole file lets the AST parser run on `doc.text` per file, exactly as it would on a locally cloned file, extracting Function/Class nodes from it. The reader only changes *how file content gets onto the server* — it does not touch chunking logic.

2. Walk every source file (from either acquisition method).
3. Parse each file with AST (or tree-sitter for non-Python languages).
4. For every function/class found, create a graph node with the three layers above.
5. Create edges for every call, import, and inheritance relationship discovered.
6. Generate the embedding from the function/class's own source text.

**Keeping it current**: re-ingest incrementally on every push (via a webhook), rather than re-parsing the whole repository on every agent run. Track the last-ingested commit SHA per project so a freshness check can catch anything a missed webhook would have skipped.

---

## 4. Metadata filtering — narrowing the search before ranking

Without this, a vector search across the whole graph could surface a similar-looking function from an unrelated project, or from the wrong language entirely. Metadata filtering restricts which nodes are even considered, before similarity ranking happens:

```cypher
CALL db.index.vector.queryNodes('function_embeddings', 20, $query_embedding)
YIELD node, score
WHERE node.project_id = $pid              // only this project
  AND node.language = $target_language    // only the relevant language
  AND node.path STARTS WITH $module_scope // only the relevant part of the repo
RETURN node, score
```

This is cheap, deterministic, and eliminates a large class of wrong-project or wrong-language hallucination risk before the LLM ever sees a candidate.

---

## 5. Reranking — a second, smarter pass after the first

Vector similarity search is fast but approximate — good at "roughly in the right neighborhood," not great at "actually the most relevant one." A **reranker** is a smaller model that examines the query and each top candidate together (not just comparing embedding distance) and re-scores them for genuine relevance.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, c["source"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, score in ranked[:top_k]]
```

**Why this reduces hallucination**: vector search alone can rank a superficially similar function above the actually relevant one (similar variable names, different logic). A reranker catches this because it reads the actual content of the query and candidate together. Feeding the model 5 well-reranked results is more reliable than feeding it 20 roughly-similar ones — fewer, better candidates means less chance of grounding an answer in the wrong one.

---

## 6. Anti-hallucination design

Good retrieval alone does not prevent hallucination — the model also needs to be constrained in *how* it uses what's retrieved. Four concrete mechanisms:

### a) Grounded citations
Every claim about existing code must point to a real retrieved node.

```python
retrieval_context = f"""
Retrieved code (cite by [path:line] when referencing):
[{node['path']}:{node['line_start']}] {node['name']}:
{node['source']}
"""
```
The generation prompt requires citing `[path:line]` when referencing existing code — if the model cannot cite a real retrieved chunk, it should not assert the referenced code exists.

### b) An explicit "insufficient context" escape hatch
```
If the retrieved context does not contain enough information to answer
correctly, say so explicitly and request a more specific search — do
NOT guess or invent function signatures, APIs, or file contents that
were not in the retrieved context.
```
This single instruction is one of the highest-leverage anti-hallucination levers available. Most hallucination happens because a model feels pressure to answer confidently even without sufficient material. Explicitly permitting "insufficient context" as a valid response removes that pressure.

### c) RAGAS faithfulness scoring — automated, after-the-fact measurement

**How it's computed**: RAGAS does not compare the answer to the context as one blob. It breaks the generated answer into individual factual claims first, then checks each one separately.

```
1. Decompose the generated answer into atomic claims
   "This function validates the token, then queries the user table,
    then returns a 401 if the query fails."
   →  claim 1: "validates the token"
   →  claim 2: "queries the user table"
   →  claim 3: "returns 401 if the query fails"

2. For each claim, an LLM judge asks: "Can this claim be inferred
   from the retrieved context alone?" — yes or no.

3. Faithfulness score = (claims verified as supported) / (total claims)
```

A score of 1.0 means every claim in the generated output traces back to something actually present in the retrieved chunks. A score of 0.6 means 40% of what was asserted was not backed by anything actually retrieved — that is the numeric signature of hallucination.

```python
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": [todo_item_description],
    "contexts": [[c["source"] for c in reranked_candidates]],  # top-k from the reranker
    "answer": [generated_code_explanation_or_reasoning],
})

result = evaluate(eval_data, metrics=[faithfulness])
score = result["faithfulness"]
```

**What happens on failure**:

```python
FAITHFULNESS_THRESHOLD = 0.8

if score < FAITHFULNESS_THRESHOLD:
    # Do not proceed to verification with unfaithful output.
    # Return to vector search, but with a narrowed query — not a repeat of the same one.
    refined_query = narrow_query(original_query, unsupported_claims)
    candidates = vector_search(refined_query, project_id)
    # retry generation with the new candidates
else:
    proceed_to_verification(generated_output)
```

The retry is informed by *which* claims failed, not a blind re-ask. If the model claimed a function exists that was never actually retrieved, the refined query searches specifically for that function name or concept — giving the next retrieval pass a real chance at finding the right context, rather than repeating the same search and getting the same weak result.

**Why this sits before verification, not instead of it**: faithfulness catches a different failure mode than tests do. Code can be syntactically valid, pass a linter, even pass tests in isolation, while still being built on a hallucinated premise — e.g., the model invents a helper function that doesn't exist and writes a test that mocks it instead of catching that it's missing. Faithfulness scoring catches "this reasoning wasn't grounded in real code" before that ungrounded reasoning even reaches the point of being tested; verification then catches whatever faithfulness-checking cannot (actual runtime correctness).

**Tuning note**: the threshold above is not universal — the faithfulness judge is itself an LLM call, so it carries some variance. Calibrate the cutoff empirically against a first batch of real runs rather than trusting a number chosen in the abstract.

### d) Verification is the final backstop
Even with perfect retrieval, reranking, and faithfulness scoring, code hallucination's real-world damage shows up as "this doesn't compile" or "this test fails" — and that is caught by an independent verification step (lint, type-check, test suite), regardless of how the code was generated.

**RAG quality reduces how often bad code gets generated. Verification is what catches it when it happens anyway. Neither replaces the other.**

---

## 7. The full pipeline

```
Todo item / task description
        ↓
Metadata-filtered vector search
  (project_id, language, path scope)
        ↓  top ~20 candidates
Reranker
  (cross-encoder scores query + candidate together)
        ↓  top ~5, genuinely relevant
Assembled context, with [path:line] citations required
        ↓
Generate code
  — model must cite sources, or explicitly say "insufficient context"
        ↓
RAGAS faithfulness check
        ↓ below threshold           ↓ passes
   retry with a narrower       proceed to verification
   retrieval query             (lint / type-check / tests)
```

---

## 8. Summary of components and their role

| Component | Role |
|---|---|
| AST / tree-sitter | Finds exact, complete code boundaries — defines what a "chunk" is |
| Graph (nodes + edges) | Captures structural relationships — what calls/imports/inherits what |
| Vector embeddings | Captures semantic meaning — what's conceptually similar to a query |
| Metadata (project_id, language, path) | Filters candidates before ranking — eliminates wrong-project/wrong-language noise |
| Reranker (cross-encoder) | Re-scores top candidates for genuine relevance, not just embedding distance |
| Citation requirement | Forces generated claims about code to be traceable to real retrieved content |
| "Insufficient context" instruction | Removes pressure to guess when retrieval doesn't provide enough |
| RAGAS faithfulness | Automated, numeric check for whether output is actually grounded in retrieval |
| Independent verification (lint/test) | Final backstop — catches incorrect code regardless of why it was generated incorrectly |
