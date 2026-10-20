# GraphRAG Audit: What Is Broken or Missing

**Repo:** `Muditshrivastav/AI-Coding-Agent` @ `beb9ddb` (Oct 6, 2026, "updated automatic restarting looped api")
**Scope:** everything in `graphrag/` (11 files, ~1,560 lines), how `agent/core.py` wires it in, `tests/test_graphrag.py`, and a comparison against `GRAPHRAG_DESIGN.md`.

**How I checked:** I read every file in `graphrag/` and traced its callers. I also ran an isolated script (`python3 -I`, no Neo4j, no Ollama) against `ast_parser.py`, `store.py` (in-memory mode) and `faithfulness.py` to confirm the items marked **[verified]**. I did **not** run your pytest suite, a live Neo4j, Ollama, or RAGAS. Items marked **[from code]** come from reading, not running.

---

## Verdict

It is **not crashing-broken**. Ingest, AST chunking, vector search, metadata filters, reranking and citation formatting all work in the in-memory tests.

But the design promises *GraphRAG with an anti-hallucination loop*, and today the code is **vector RAG with a graph-shaped store**:

1. The graph is never traversed at query time.
2. The faithfulness retry loop (design step 12) is never called.
3. Graph edges are built incorrectly or incompletely.
4. Multi-tenant isolation is not enforced.

---

## 1. Design vs. code, component by component

| # | Design component | Status | Where |
|---|---|---|---|
| 1-2 | Clone / incremental sync | **Partial.** Full ingest only. No incremental sync, no webhook, no stale-node deletion | `github_reader.py`, `pipeline.py` |
| 3 | AST parser (`ast` + tree-sitter) | **Partial.** Python only. tree-sitter is not implemented | `ast_parser.py:237-262` |
| 4 | Graph + embedding builder | **Partial.** `CONTAINS`, `CALLS`, `INHERITS` only; `IMPORTS` never stored. `CALLS`/`INHERITS` are order-dependent | `store.py:123-191` |
| 5 | Neo4j store | **Works**, but silently falls back to in-memory | `store.py:35-53` |
| 6 | Task / todo query | **Missing.** Nothing feeds the todo item into retrieval; the LLM calls the tool ad hoc | `agent/core.py`, `nodes/` |
| 7 | Metadata filter | **Works but post-filters** (see B5) | `store.py:248-264` |
| 8 | Vector search | **Works** | `store.py` |
| 9 | Reranker | **Works** (falls back to word overlap if the model fails to load) | `reranker.py` |
| 10 | Context assembly + citations | **Works** | `faithfulness.py:27-57` |
| 11 | Generation with cite-or-flag-gap | **Prompt text only; not enforced** | `faithfulness.py:49-57` |
| 12 | Faithfulness check + retry loop | **Not wired** (see B1) | `faithfulness.py:59-153` |
| 13 | Verification | Outside `graphrag/` | `tools/verification_tools.py` |

---

## 2. Bugs and critical gaps (fix first)

### B1. The faithfulness loop is never executed
`evaluate_faithfulness()` and `narrow_query()` are called **only from `tests/test_graphrag.py`**. `config.max_faithfulness_retries = 2` is never read. No node in `nodes/` imports anything from `graphrag/`, and `pipeline.py` has no method that does generate → score → narrow → retry.
Also, `frameworks/ragas_evaluator.py` is a stub that returns `1.0` whenever any context exists.

**Fix:** add `GraphRAGPipeline.retrieve_and_check(query, generate_fn)` that runs retrieve → generate → evaluate → `narrow_query` → retry up to `max_faithfulness_retries`, then returns an "insufficient context" result. Call it from the build subagent per todo item.

### B2. The graph is never used for retrieval **[from code]**
`pipeline.retrieve()` (`pipeline.py:100-134`) does embed → vector search → rerank. The Cypher in `_query_neo4j_vector` contains no `MATCH ... -[:CALLS|INHERITS|CONTAINS]-`. The `CALLS`/`INHERITS` edges are write-only.

**Fix:** after reranking, expand 1-2 hops from the top hits and trim to a token budget:
```cypher
MATCH (c:CodeChunk {project_id: $pid})
WHERE c.path = $path AND c.name = $name
OPTIONAL MATCH (c)-[:CALLS]->(callee:CodeChunk {project_id: $pid})
OPTIONAL MATCH (caller:CodeChunk {project_id: $pid})-[:CALLS]->(c)
OPTIONAL MATCH (c)-[:INHERITS]->(base:CodeChunk {project_id: $pid})
RETURN c, collect(DISTINCT callee)[..5] AS callees,
          collect(DISTINCT caller)[..5] AS callers,
          collect(DISTINCT base)[..2]  AS bases
```

### B3. `CALLS` / `INHERITS` edges are order-dependent and name-matched **[from code + verified]**
`store.py:158-191` creates edges at the moment each chunk is stored, using `MATCH` on targets that **already exist**.
- A function that calls something stored *later* in the loop (or in a later file) **never gets the edge**. Which edges exist depends on file iteration order.
- Matching is by bare name across the **whole project**: `target.name = $t OR target.name ENDS WITH ('.' + $t)`. A call to `run()` or `get()` links to *every* method with that name, in any file or class.
- `ast_parser.py:56-62` records only the callee's bare name (`foo`, or `attr` for `obj.attr()`), with no module or receiver, so it cannot be resolved correctly later. In the in-memory mode, `CALLS` targets are stored as raw strings (`'b'`, `'sqrt'`), not node ids **[verified]**.

**Fix:** store all nodes first, then run a second resolution pass. Resolve through each file's import map (`from x import y`) and the enclosing class (`self.m()` → `Class.m`). For better accuracy use Jedi, pyright or SCIP. Drop unresolved calls instead of guessing. Write edges with `UNWIND` batches.

### B4. `IMPORTS` edges are never created **[verified]**
`CodeChunk.imports` is populated (`['os', 'math.sqrt']`), but neither the Neo4j path nor the memory path writes it. The only relationship types ever stored are `CONTAINS` and `CALLS` (and `INHERITS` when present). The docstring in `store.py:4` and the README claim `IMPORTS` exists.

**Fix:** add `(:File)-[:IMPORTS]->(:File|:Module)` (internal imports) and `(:File)-[:DEPENDS_ON]->(:Package)` (external).

### B5. Metadata filtering happens *after* the vector top-k (multi-tenant correctness) **[from code]**
`store.py:249-253` calls `db.index.vector.queryNodes(..., max(top_k*2, 50), ...)` and **then** applies `WHERE node.project_id = $project_id`. With many projects in one index, the 50 nearest nodes can all belong to other tenants, so the filter leaves **zero results** even though the project has relevant code.

**Fix:** use a filtered vector search (Neo4j 5.x+ supports filtering inside the vector query), or over-fetch heavily, or keep one index per project / tenant.

### B6. Tenant isolation is optional and decided by the LLM **[from code]**
`tool.py:19-22` declares `project_id` as `Optional` with default `None`, and the Cypher treats `None` as "no filter" (`$project_id IS NULL OR ...`). If the model omits it, the search spans **every project in the database**. `language` also defaults to `"python"` (`tool.py:23-26`), so a JS/TS project silently returns nothing.

**Fix:** bind `project_id` in the tool factory from the authenticated session. Never accept it from the model, and make the store raise if it is `None`. Default `language` to `None`.

### B7. Silent in-memory fallback gives false success **[from code]**
If Neo4j is unreachable, `store.py:48-53` logs a warning and switches to in-memory. Then `ingest_tool.py:102-106` still returns *"Successfully ingested ... into Neo4j"*. Data is lost on restart.
Also `agent/core.py:169-170` creates the ingest tool (module singleton `_get_pipeline()`) and the retriever tool (`create_graphrag_retriever_tool()` → its **own** `GraphRAGPipeline()`) separately, so in fallback mode ingest and retrieve **don't share data**. You also load two reranker models and two embedder connections.

**Fix:** in production, fail loudly when Neo4j is down (`GRAPHRAG_ALLOW_MEMORY=false`). Build one pipeline in `CodingAgentHarness.__init__` and pass it to both tool factories. Report `store.is_connected()` in the ingest message.

### B8. Embedding configuration can break retrieval silently **[from code]**
- `config.embedding_dimension = 4096` is hard-coded and used to create the vector index (`store.py:84`), but `DefaultEmbedder` auto-detects its real dimension (`embedder.py:62-63`). If they differ (I believe `qwen3-embedding:4b` outputs 2560, **please verify** with `len(embed_query("x"))`), the index is created with the wrong size and inserts or queries fail or return nothing.
- `embedder.py:80` has a typo: `self.ollama_embeddings = None` (missing leading underscore). The real attribute `_ollama_embeddings` stays set to the broken Ollama object, so each embed call retries a failing network request before falling back.
- When Ollama is down, every embedding becomes a **hash-of-tokens pseudo-vector** (`embedder.py:96-98`, `114-134`) with no log line. Ingest and retrieval "work" but return meaningless matches. Index vectors made with one embedder (Ollama) are also incompatible with queries embedded by the other (hash).

**Fix:** read the dimension from the embedder when creating the index; raise (don't fall back) outside tests; log which embedder is active; store `embedder_name` on nodes.

### B9. The "faithfulness" check is much weaker than the design says
`faithfulness.py`:
- `:90` returns `1.0` if the key `"faithfulness"` is missing from the result.
- `:97` `except Exception:` swallows every RAGAS failure (missing LLM key, version change) and drops to the heuristic without logging. RAGAS also needs a judge LLM passed in explicitly; none is configured.
- `:125-127` heuristic uses **substring** matching: a claim about `validate_token(x)` passes if the context contains `validate_token_expiry` **[verified: score 1.0, passed]**.
- `:110-112` the phrase "insufficient context" anywhere in the answer scores 1.0, even alongside invented calls **[verified: passed]**.
- It only checks function-call names. It never checks imports, attributes, argument counts, or whether the symbol exists in the repo graph.

**Fix:** replace the live-loop check with a deterministic one: parse the generated code with `ast`, extract called/imported names, and verify each against (graph symbols ∪ stdlib ∪ installed packages). Keep RAGAS for offline evaluation, with an explicit judge LLM and visible failures.

---

## 3. Missing features (design promised, or needed for correctness)

### Ingestion
- **Incremental update + deletion.** No webhook handler (`/webhooks/github` is a stub in `CODING_AGENT_IMPLEMENTATION.md`, absent from `interfaces/api_server.py`). Re-ingest only `MERGE`s, so deleted or renamed functions/files stay in the graph forever and get retrieved as if real. Need: per-file delete-then-insert for changed files; delete removed files.
- **Commit SHA tracking.** `ingest_tool.py:91-96` never passes `commit_sha`, so every node has `""`. There's no "last ingested SHA per project" and no freshness check.
- **tree-sitter / other languages.** `parse_file` returns one whole-file `Module` chunk for non-Python files **[verified: a 100,000-character `.tsx` became one chunk, labelled `javascript`]**. That ruins the "whole function, never a partial" principle and blows past embedder/reranker input limits. `.ts`/`.tsx` are labelled `javascript` (`ast_parser.py:249`).
- **Chunk gaps in Python** **[verified]**: nested functions, nested classes (`A.B`) and module-level constants/globals are not captured; `@x.setter` methods collide on the `(project_id, path, name)` unique key, so the second overwrites the first; each Class chunk contains all its methods **and** the methods are also separate chunks, so retrieval can return the same code twice and waste tokens.
- **Size limits.** No max-chunk-size handling for huge functions/files.
- **Batching.** One Cypher session per chunk plus one per call target (`store.py:155-191`): slow for real repos. Use `UNWIND` batches.
- **Per-tenant GitHub auth.** `GitHubRepoIngester` uses a global `GITHUB_TOKEN` from env (`github_reader.py:31`). Your multi-tenant plan needs GitHub App installation tokens or OAuth per project (`tools/github_oauth.py` exists but graphrag doesn't use it). Default branch is hard-coded to `"main"`.
- **Dependency index.** Only your repo is indexed. Hallucinated *library* APIs (the most common kind) can't be caught.

### Retrieval
- **Hybrid search.** Pure embeddings are weak on identifiers. Add BM25 / full-text on `name`, `path`, `docstring`, and merge with reciprocal rank fusion.
- **Exact symbol lookup tool** (`lookup_symbol(name)`, `get_callers(name)`, `get_file(path)`), so the agent can fetch deterministically.
- **Better embedding text.** Only raw `source` is embedded (`pipeline.py:71`). Prepend path, qualified name, signature and docstring (`docstring` is stored but never embedded).
- **Token budgeting.** No cap on assembled context. Add a budget, dedupe overlapping class/method chunks, and trim the expanded subgraph by score.
- **Reranker edge cases.** `bge-reranker-base` truncates long chunks; the fallback reranker's scores are on a different scale than the model's.

### Agent integration
- **The agent is never told to use it.** The build subagent receives the tools (via `agent/core.py:168-170` → `self._tools`), but the system prompt in `nodes/build_subagent.py` never mentions `ingest_github_repo` or `retrieve_graphrag_code`, and neither does `harness/AGENTS.md`. The planning subagent only receives `write_plan_tool` (plus an optional `extra_tools` list that `agent/core.py` doesn't use for GraphRAG), and the design subagent only gets draw.io tools, so neither can query the repo graph.
- **Citations are not checked.** Nothing verifies that `[path:line]` references in generated output match retrieved chunks.

### Testing and evaluation
- `tests/test_graphrag.py` runs against the in-memory store (no Neo4j), and its faithfulness test configures no judge LLM, so it most likely exercises the heuristic fallback rather than RAGAS (`ragas>=0.4.3` is a dependency, but I didn't run it). Missing: live-Neo4j integration test, edge correctness (order independence, false-positive links), `project_id=None` isolation, multi-project vector-search starvation (B5), incremental update / deletion, and the retry loop.
- No golden retrieval set. Build 20-30 tasks with the symbols that should come back; track recall@5 and the rate of non-existent references. The `0.8` threshold and `top_k=20→5` are guesses until you measure.
- No observability: log retrieved node ids, scores, which embedder/reranker was active, and retry counts into LangSmith.

---

## 4. Side notes (outside `graphrag/` but worth fixing)

- **`node_modules/` is committed:** `git ls-files | grep -c node_modules` returns **9,435** tracked files (`ui/` and `vscode-extension/`), and `.gitignore` has no `node_modules` entry. Remove them from the index and add the ignore rule.
- `graphrag/tool.py:67-98` `create_neo4j_graphrag_mcp_tools` defaults to `read_only=False` and password `"password"`. It is unused today, but if enabled it gives the model raw Cypher write access that bypasses `project_id` isolation. Default it to read-only and don't give it to tenant-facing agents.
- `config.py:21-24` defaults the Neo4j password to `"password"`; fail if unset outside dev.

---

## 5. Suggested fix order

| Priority | Item | Why |
|---|---|---|
| 1 | B6 + B5 (enforce `project_id`, filter inside the vector query) | Data leakage / empty results in multi-tenant use |
| 2 | B7 + B8 (no silent fallbacks; one shared pipeline; dimension from embedder) | Currently "works" while returning junk |
| 3 | B3 + B4 (two-pass edge resolution, `IMPORTS` edges) | The graph is wrong, so everything built on it is wrong |
| 4 | B2 (1-2 hop expansion + token budget) | This is what makes it GraphRAG |
| 5 | B9 + B1 (deterministic symbol check, wire the retry loop) | Delivers the anti-hallucination promise |
| 6 | Incremental ingest, deletion, commit SHA, webhook | Keeps the graph from going stale |
| 7 | Hybrid search, `lookup_symbol` tools, better embedding text | Retrieval quality |
| 8 | tree-sitter for JS/TS, dependency index | Coverage |
| 9 | Golden set, integration tests, LangSmith traces | Tuning on evidence instead of guesses |

---

## 6. Not verified

- Behavior against a live Neo4j (vector index creation, `queryNodes` filtering semantics on your Neo4j version).
- RAGAS runtime behavior (version-specific return type, judge LLM configuration).
- The real output dimension of `qwen3-embedding:4b` in your Ollama.
- Whether your pytest suite currently passes; I did not install your dependencies.
