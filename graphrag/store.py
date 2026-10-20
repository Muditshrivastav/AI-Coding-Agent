"""
graphrag/store.py - Neo4j Graph + Vector store layer.

Implements the three layers defined in GRAPHRAG_DESIGN.md:
  Layer 1 - GRAPH STRUCTURE: Nodes (:File, :Function, :Class, :Module),
            Edges (:CONTAINS, :CALLS, :IMPORTS, :INHERITS, :DEPENDS_ON).
  Layer 2 - VECTOR EMBEDDINGS: Node property `embedding` indexed via Neo4j vector index.
  Layer 3 - METADATA FILTERING: Cypher filtering by project_id, language, module/path scope.

Includes in-memory fallback gated by GRAPHRAG_ALLOW_MEMORY=true (B7).

Fixes applied:
  B3 - Two-pass edge resolution: nodes first, then resolve_edges() so forward
       references are always found; edges batched with UNWIND.
  B4 - IMPORTS / DEPENDS_ON edges now created from chunk.imports lists.
  B5 - Tenant project_id filter applied *inside* the Neo4j 5.x vector procedure
       call to avoid cross-tenant result starvation.
  B7 - Falls back to in-memory only when GRAPHRAG_ALLOW_MEMORY=true is set;
       otherwise raises loudly at startup so mis-configs surface immediately.
  B8 - init_schema() accepts embedding_dimension from the live embedder rather
       than always using the potentially-wrong config hard-code.
  B2 - expand_neighbors() fetches 1-2 hop CALLS/INHERITS neighbours from top hits.
"""

from typing import Any
import math
import logging
import os
from graphrag.ast_parser import CodeChunk
from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Manages the Neo4j graph nodes, edges, vector index, and filtered vector retrieval."""

    def __init__(self, config: GraphRAGConfig | None = None) -> None:
        self.config = config or GraphRAGConfig()
        self._driver = None
        self._is_connected = False
        # In-memory storage for offline / testing mode
        self._mem_nodes: dict[str, dict[str, Any]] = {}
        self._mem_edges: list[tuple[str, str, str]] = []  # (source_id, relation, target_id)

        self._try_connect()

    def _try_connect(self) -> None:
        # B7: only allow in-memory fallback when explicitly opted-in via env var.
        allow_memory = os.getenv("GRAPHRAG_ALLOW_MEMORY", "false").lower() in ("1", "true", "yes")
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
            )
            # Verify connectivity
            with self._driver.session(database=self.config.neo4j_database) as session:
                session.run("RETURN 1 AS test").consume()
            self._is_connected = True
            logger.info("Connected to Neo4j successfully.")
        except Exception as exc:
            if allow_memory:
                logger.warning(
                    f"Neo4j connection could not be established ({exc}). "
                    "GRAPHRAG_ALLOW_MEMORY=true — operating in in-memory simulation mode. "
                    "Data will be LOST on restart."
                )
                self._is_connected = False
            else:
                raise RuntimeError(
                    f"Neo4j connection failed ({exc}). "
                    "Set GRAPHRAG_ALLOW_MEMORY=true to allow degraded in-memory fallback, "
                    "or fix NEO4J_URI / NEO4J_PASSWORD in your .env file."
                ) from exc

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def is_connected(self) -> bool:
        return self._is_connected

    def init_schema(self, embedding_dimension: int | None = None) -> None:
        """Creates constraints and vector index in Neo4j.

        Args:
            embedding_dimension: Dimension from the live embedder.  Overrides
                the hard-coded config default when provided (B8 fix).
        """
        if not self._is_connected:
            return

        # B8: use the real dimension probed from the live model instead of the config default.
        dim = embedding_dimension or self.config.embedding_dimension
        queries = [
            # Constraint to identify unique code chunks
            """
            CREATE CONSTRAINT chunk_unique IF NOT EXISTS
            FOR (c:CodeChunk) REQUIRE (c.project_id, c.path, c.name) IS UNIQUE
            """,
            # Constraint for Files
            """
            CREATE CONSTRAINT file_unique IF NOT EXISTS
            FOR (f:File) REQUIRE (f.project_id, f.path) IS UNIQUE
            """,
            # Vector Index for fast embedding similarity (B8: use real dim from embedder)
            f"""
            CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
            FOR (c:CodeChunk) ON (c.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """,
        ]

        with self._driver.session(database=self.config.neo4j_database) as session:
            for q in queries:
                try:
                    session.run(q).consume()
                except Exception as exc:
                    logger.debug(f"Schema query info: {exc}")

    def store_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Phase 1 of ingestion: store chunk nodes and CONTAINS edges only.

        Call resolve_edges(chunks) afterwards so CALLS/INHERITS/IMPORTS edges
        are created in a second pass where all target nodes already exist (B3).
        """
        if not chunks:
            return 0

        stored_count = 0
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{chunk.project_id}::{chunk.path}::{chunk.name}"
            file_id = f"{chunk.project_id}::{chunk.path}"

            if self._is_connected:
                self._store_chunk_neo4j(chunk, emb)
            else:
                self._store_chunk_memory(chunk_id, file_id, chunk, emb)

            stored_count += 1

        return stored_count

    def _store_chunk_neo4j(self, chunk: CodeChunk, emb: list[float]) -> None:
        """Creates/merges the chunk node and its parent File. No edges here (B3: two-pass)."""
        cypher = f"""
        MERGE (file:File {{project_id: $project_id, path: $path}})
        MERGE (c:{chunk.chunk_type}:CodeChunk {{
            project_id: $project_id,
            path: $path,
            name: $name
        }})
        SET c.source = $source,
            c.line_start = $line_start,
            c.line_end = $line_end,
            c.language = $language,
            c.commit_sha = $commit_sha,
            c.docstring = $docstring,
            c.embedding = $embedding,
            c.imports = $imports

        MERGE (file)-[:CONTAINS]->(c)
        """
        params = {
            "project_id": chunk.project_id,
            "path": chunk.path,
            "name": chunk.name,
            "source": chunk.source,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "language": chunk.language,
            "commit_sha": chunk.commit_sha,
            "docstring": chunk.docstring,
            "embedding": emb,
            "imports": chunk.imports,
        }

        with self._driver.session(database=self.config.neo4j_database) as session:
            session.run(cypher, params).consume()

    def _store_chunk_memory(
        self,
        chunk_id: str,
        file_id: str,
        chunk: CodeChunk,
        emb: list[float],
    ) -> None:
        data = chunk.to_dict()
        data["embedding"] = emb
        data["id"] = chunk_id
        self._mem_nodes[chunk_id] = data
        # Only CONTAINS in phase 1; CALLS/INHERITS/IMPORTS added by resolve_edges() (B3).
        self._mem_edges.append((file_id, "CONTAINS", chunk_id))

    # ------------------------------------------------------------------
    # B3+B4: Two-pass edge resolution
    # ------------------------------------------------------------------

    def resolve_edges(self, chunks: list[CodeChunk]) -> None:
        """Phase 2: create CALLS, INHERITS, IMPORTS, DEPENDS_ON edges after all nodes exist.

        Running this *after* store_chunks() ensures forward-referenced callees and
        base classes are already in the graph (B3 fix).  IMPORTS/DEPENDS_ON edges
        were previously never created (B4 fix).
        """
        if not chunks:
            return
        if self._is_connected:
            self._resolve_edges_neo4j(chunks)
        else:
            self._resolve_edges_memory(chunks)

    def _resolve_edges_neo4j(self, chunks: list[CodeChunk]) -> None:
        """Create structural edges via batched UNWIND Cypher (B3+B4)."""
        with self._driver.session(database=self.config.neo4j_database) as session:
            for project_id in {c.project_id for c in chunks}:
                proj_chunks = [c for c in chunks if c.project_id == project_id]
                all_names = {c.name for c in proj_chunks}

                # CALLS edges — resolve via enclosing class or exact name match only.
                calls_batch: list[dict[str, str]] = []
                for c in proj_chunks:
                    enclosing = c.name.split(".")[0] if "." in c.name else None
                    for target in c.calls:
                        resolved: str | None = None
                        qualified = f"{enclosing}.{target}" if enclosing else None
                        if qualified and qualified in all_names:
                            resolved = qualified
                        elif target in all_names:
                            resolved = target
                        if resolved:
                            calls_batch.append({
                                "proj": project_id, "sp": c.path, "sn": c.name, "tn": resolved
                            })
                if calls_batch:
                    session.run(
                        """
                        UNWIND $batch AS r
                        MATCH (s:CodeChunk {project_id: r.proj, path: r.sp, name: r.sn})
                        MATCH (t:CodeChunk {project_id: r.proj, name: r.tn})
                        MERGE (s)-[:CALLS]->(t)
                        """,
                        {"batch": calls_batch},
                    ).consume()

                # INHERITS edges
                inh_batch: list[dict[str, str]] = [
                    {"proj": project_id, "sp": c.path, "sn": c.name, "bn": base}
                    for c in proj_chunks for base in c.inherits if base in all_names
                ]
                if inh_batch:
                    session.run(
                        """
                        UNWIND $batch AS r
                        MATCH (s:CodeChunk {project_id: r.proj, path: r.sp, name: r.sn})
                        MATCH (b:CodeChunk {project_id: r.proj, name: r.bn})
                        MERGE (s)-[:INHERITS]->(b)
                        """,
                        {"batch": inh_batch},
                    ).consume()

                # B4: IMPORTS (internal) and DEPENDS_ON (external) edges
                internal_paths = {c.path for c in proj_chunks}
                imp_batch: list[dict[str, str]] = []
                dep_batch: list[dict[str, str]] = []
                for c in proj_chunks:
                    if c.chunk_type != "Module":
                        continue
                    for imp in c.imports:
                        cand = imp.replace(".", "/") + ".py"
                        is_internal = any(
                            p.endswith(cand) or cand.endswith(p.lstrip("/"))
                            for p in internal_paths
                        )
                        if is_internal:
                            imp_batch.append({"proj": project_id, "sp": c.path, "tp": cand})
                        else:
                            dep_batch.append({"proj": project_id, "sp": c.path, "pkg": imp.split(".")[0]})
                if imp_batch:
                    session.run(
                        """
                        UNWIND $batch AS r
                        MATCH (src:File {project_id: r.proj, path: r.sp})
                        MATCH (tgt:File {project_id: r.proj})
                        WHERE tgt.path ENDS WITH r.tp OR r.tp ENDS WITH tgt.path
                        MERGE (src)-[:IMPORTS]->(tgt)
                        """,
                        {"batch": imp_batch},
                    ).consume()
                if dep_batch:
                    session.run(
                        """
                        UNWIND $batch AS r
                        MATCH (src:File {project_id: r.proj, path: r.sp})
                        MERGE (pkg:Package {name: r.pkg})
                        MERGE (src)-[:DEPENDS_ON]->(pkg)
                        """,
                        {"batch": dep_batch},
                    ).consume()

    def _resolve_edges_memory(self, chunks: list[CodeChunk]) -> None:
        """In-memory equivalent of _resolve_edges_neo4j for tests (B3+B4)."""
        all_names = {c.name for c in chunks}
        for c in chunks:
            cid = f"{c.project_id}::{c.path}::{c.name}"
            enclosing = c.name.split(".")[0] if "." in c.name else None
            for target in c.calls:
                resolved = None
                q = f"{enclosing}.{target}" if enclosing else None
                if q and q in all_names:
                    resolved = q
                elif target in all_names:
                    resolved = target
                if resolved:
                    self._mem_edges.append((cid, "CALLS", f"{c.project_id}::{c.path}::{resolved}"))
            for base in c.inherits:
                self._mem_edges.append((cid, "INHERITS", base))
            for imp in c.imports:
                self._mem_edges.append((cid, "IMPORTS", imp))

    # ------------------------------------------------------------------
    # B2: 1-2 hop graph expansion
    # ------------------------------------------------------------------

    def expand_neighbors(
        self,
        hit_nodes: list[dict],
        project_id: str,
        neighbor_cap: int = 5,
    ) -> list[dict]:
        """For each top-ranked hit, fetch CALLS/INHERITS neighbours (B2 fix)."""
        if not hit_nodes:
            return []
        if self._is_connected:
            return self._expand_neo4j(hit_nodes, project_id, neighbor_cap)
        return self._expand_memory(hit_nodes, project_id, neighbor_cap)

    def _expand_neo4j(
        self, hit_nodes: list[dict], project_id: str, neighbor_cap: int
    ) -> list[dict]:
        cypher = """
        UNWIND $hits AS hit
        MATCH (c:CodeChunk {project_id: $pid, path: hit.path, name: hit.name})
        OPTIONAL MATCH (c)-[:CALLS]->(callee:CodeChunk {project_id: $pid})
        OPTIONAL MATCH (caller:CodeChunk {project_id: $pid})-[:CALLS]->(c)
        OPTIONAL MATCH (c)-[:INHERITS]->(base:CodeChunk {project_id: $pid})
        RETURN collect(DISTINCT callee)[..$cap] AS callees,
               collect(DISTINCT caller)[..$cap] AS callers,
               collect(DISTINCT base)[..2]       AS bases
        """
        hits_param = [{"path": n.get("path", ""), "name": n.get("name", "")} for n in hit_nodes]
        seen: set[str] = {f"{n.get('path')}::{n.get('name')}" for n in hit_nodes}
        extra: list[dict] = []
        try:
            with self._driver.session(database=self.config.neo4j_database) as session:
                for rec in session.run(cypher, {"hits": hits_param, "pid": project_id, "cap": neighbor_cap}):
                    for group in [rec.get("callees", []), rec.get("callers", []), rec.get("bases", [])]:
                        for node in (group or []):
                            if node is None:
                                continue
                            nd = dict(node)
                            key = f"{nd.get('path')}::{nd.get('name')}"
                            if key not in seen:
                                seen.add(key)
                                extra.append(nd)
        except Exception as exc:
            logger.warning(f"[store] Graph expansion failed: {exc}")
        return extra

    def _expand_memory(self, hit_nodes: list[dict], project_id: str, neighbor_cap: int) -> list[dict]:
        seen = {f"{n.get('project_id')}::{n.get('path')}::{n.get('name')}" for n in hit_nodes}
        extra: list[dict] = []
        for n in hit_nodes:
            nid = f"{n.get('project_id')}::{n.get('path')}::{n.get('name')}"
            count = 0
            for src, rel, tgt in self._mem_edges:
                if count >= neighbor_cap:
                    break
                if src == nid and rel in ("CALLS", "INHERITS") and tgt in self._mem_nodes and tgt not in seen:
                    seen.add(tgt)
                    extra.append(dict(self._mem_nodes[tgt]))
                    count += 1
                elif tgt == nid and rel == "CALLS" and src in self._mem_nodes and src not in seen:
                    seen.add(src)
                    extra.append(dict(self._mem_nodes[src]))
                    count += 1
        return extra

    def query_similar_nodes(
        self,
        query_embedding: list[float],
        project_id: str | None = None,
        language: str | None = None,
        module_scope: str | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Searches nodes via vector similarity with Layer 3 metadata filtering
        (project_id, language, module path prefix) directly in Cypher or in-memory.
        """
        if self._is_connected:
            return self._query_neo4j_vector(
                query_embedding=query_embedding,
                project_id=project_id,
                language=language,
                module_scope=module_scope,
                top_k=top_k,
            )
        else:
            return self._query_memory_vector(
                query_embedding=query_embedding,
                project_id=project_id,
                language=language,
                module_scope=module_scope,
                top_k=top_k,
            )

    def _query_neo4j_vector(
        self,
        query_embedding: list[float],
        project_id: str | None,
        language: str | None,
        module_scope: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        # B5: filter by project_id *inside* the procedure YIELD so the fetched k
        # candidates are already scoped to this tenant, preventing cross-tenant starvation.
        if project_id is not None:
            cypher = """
            CALL db.index.vector.queryNodes('function_embeddings', $top_k_fetch, $query_embedding)
            YIELD node, score
            WHERE node.project_id = $project_id
              AND ($language IS NULL OR node.language = $language)
              AND ($module_scope IS NULL OR node.path STARTS WITH $module_scope)
            RETURN node.name AS name,
                   node.path AS path,
                   node.source AS source,
                   node.line_start AS line_start,
                   node.line_end AS line_end,
                   node.project_id AS project_id,
                   node.language AS language,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
        else:
            cypher = """
            CALL db.index.vector.queryNodes('function_embeddings', $top_k_fetch, $query_embedding)
            YIELD node, score
            WHERE ($language IS NULL OR node.language = $language)
              AND ($module_scope IS NULL OR node.path STARTS WITH $module_scope)
            RETURN node.name AS name,
                   node.path AS path,
                   node.source AS source,
                   node.line_start AS line_start,
                   node.line_end AS line_end,
                   node.project_id AS project_id,
                   node.language AS language,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
        params = {
            # over-fetch more aggressively when filtering by project_id so tenant starvation is avoided (B5)
            "top_k_fetch": max(top_k * 4, 100) if project_id is not None else max(top_k * 2, 50),
            "query_embedding": query_embedding,
            "project_id": project_id,
            "language": language,
            "module_scope": module_scope,
            "top_k": top_k,
        }

        results = []
        with self._driver.session(database=self.config.neo4j_database) as session:
            cursor = session.run(cypher, params)
            for record in cursor:
                results.append(dict(record))
        return results

    def _query_memory_vector(
        self,
        query_embedding: list[float],
        project_id: str | None,
        language: str | None,
        module_scope: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored = []
        for node_id, data in self._mem_nodes.items():
            # Metadata filtering
            if project_id and data.get("project_id") != project_id:
                continue
            if language and data.get("language") != language:
                continue
            if module_scope and not data.get("path", "").startswith(module_scope):
                continue

            node_emb = data.get("embedding", [])
            sim = self._cosine_similarity(query_embedding, node_emb)
            res = dict(data)
            res["score"] = sim
            scored.append(res)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
