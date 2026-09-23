"""
graphrag/store.py - Neo4j Graph + Vector store layer.

Implements the three layers defined in GRAPHRAG_DESIGN.md:
  Layer 1 - GRAPH STRUCTURE: Nodes (:File, :Function, :Class, :Module),
            Edges (:CONTAINS, :CALLS, :IMPORTS, :INHERITS).
  Layer 2 - VECTOR EMBEDDINGS: Node property `embedding` indexed via Neo4j vector index.
  Layer 3 - METADATA FILTERING: Cypher filtering by project_id, language, module/path scope.

Includes in-memory fallback for environments without an active Neo4j server.
"""

from typing import Any
import math
import logging
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
            self.init_schema()
            logger.info("Connected to Neo4j successfully.")
        except Exception as exc:
            logger.warning(
                f"Neo4j connection could not be established ({exc}). "
                "Operating in in-memory simulation mode."
            )
            self._is_connected = False

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def is_connected(self) -> bool:
        return self._is_connected

    def init_schema(self) -> None:
        """Creates constraints and vector index in Neo4j."""
        if not self._is_connected:
            return

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
            # Vector Index for fast embedding similarity
            f"""
            CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
            FOR (c:CodeChunk) ON (c.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {self.config.embedding_dimension},
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
        Stores code chunks as nodes with structural edges (:CONTAINS, :CALLS, :IMPORTS, :INHERITS).
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
        """Executes Cypher statements to create nodes and relationships."""
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
            c.embedding = $embedding

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
        }

        with self._driver.session(database=self.config.neo4j_database) as session:
            session.run(cypher, params).consume()

            # Create CALLS edges
            for call_target in chunk.calls:
                call_cypher = """
                MATCH (c:CodeChunk {project_id: $project_id, path: $path, name: $name})
                MATCH (target:CodeChunk {project_id: $project_id})
                WHERE target.name = $target_name OR target.name ENDS WITH ('.' + $target_name)
                MERGE (c)-[:CALLS]->(target)
                """
                session.run(
                    call_cypher,
                    {
                        "project_id": chunk.project_id,
                        "path": chunk.path,
                        "name": chunk.name,
                        "target_name": call_target,
                    },
                ).consume()

            # Create INHERITS edges
            for base_class in chunk.inherits:
                inherit_cypher = """
                MATCH (c:CodeChunk {project_id: $project_id, path: $path, name: $name})
                MATCH (base:CodeChunk {project_id: $project_id, name: $base_name})
                MERGE (c)-[:INHERITS]->(base)
                """
                session.run(
                    inherit_cypher,
                    {
                        "project_id": chunk.project_id,
                        "path": chunk.path,
                        "name": chunk.name,
                        "base_name": base_class,
                    },
                ).consume()

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

        self._mem_edges.append((file_id, "CONTAINS", chunk_id))
        for call in chunk.calls:
            self._mem_edges.append((chunk_id, "CALLS", call))
        for inh in chunk.inherits:
            self._mem_edges.append((chunk_id, "INHERITS", inh))

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
        cypher = """
        CALL db.index.vector.queryNodes('function_embeddings', $top_k_fetch, $query_embedding)
        YIELD node, score
        WHERE ($project_id IS NULL OR node.project_id = $project_id)
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
        params = {
            "top_k_fetch": max(top_k * 2, 50),
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
