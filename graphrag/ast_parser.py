"""
graphrag/ast_parser.py - AST code-aware chunking and symbol boundary extraction.

Treats code as code: parses Python AST to identify exact function, class, and module
boundaries, tracking calls, imports, and inheritance without cutting code arbitrarily.
"""

from dataclasses import dataclass, field
from typing import Any
import ast
import os


@dataclass
class CodeChunk:
    """Represents a coherent, code-aware AST unit (function, class, or module)."""

    name: str
    chunk_type: str  # "Function", "Class", "Module"
    path: str
    source: str
    line_start: int
    line_end: int
    project_id: str = "default"
    language: str = "python"
    commit_sha: str = ""
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    docstring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chunk_type": self.chunk_type,
            "path": self.path,
            "source": self.source,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "project_id": self.project_id,
            "language": self.language,
            "commit_sha": self.commit_sha,
            "calls": self.calls,
            "imports": self.imports,
            "inherits": self.inherits,
            "docstring": self.docstring,
        }


class _CallVisitor(ast.NodeVisitor):
    """Visits an AST subtree and gathers function/method call names."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


class ASTCodeParser:
    """Parses code files using the AST module to extract complete semantic code units."""

    def __init__(self, project_id: str = "default", commit_sha: str = "") -> None:
        self.project_id = project_id
        self.commit_sha = commit_sha

    def parse_python(
        self,
        source: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Parses Python code and extracts functions, classes, and top-level module blocks."""
        chunks: list[CodeChunk] = []
        if not source.strip():
            return chunks

        lines = source.splitlines()

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            # Fallback for unparseable or snippet files: emit entire file as Module chunk
            return [
                CodeChunk(
                    name=os.path.basename(file_path),
                    chunk_type="Module",
                    path=file_path,
                    source=source,
                    line_start=1,
                    line_end=len(lines),
                    project_id=self.project_id,
                    language="python",
                    commit_sha=self.commit_sha,
                )
            ]

        # Gather top-level module imports
        module_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    module_imports.append(f"{mod}.{alias.name}" if mod else alias.name)

        # Inspect classes and functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    self._extract_function_chunk(
                        node=node,
                        lines=lines,
                        file_path=file_path,
                        parent_class=None,
                        module_imports=module_imports,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                # Extract Class chunk
                class_chunk = self._extract_class_chunk(
                    node=node,
                    lines=lines,
                    file_path=file_path,
                    module_imports=module_imports,
                )
                chunks.append(class_chunk)

                # Also extract inner methods as distinct Function chunks
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = self._extract_function_chunk(
                            node=item,
                            lines=lines,
                            file_path=file_path,
                            parent_class=node.name,
                            module_imports=module_imports,
                        )
                        chunks.append(method_chunk)

        # If no functions or classes were found (e.g. config or script), store file as Module
        if not chunks:
            chunks.append(
                CodeChunk(
                    name=os.path.basename(file_path),
                    chunk_type="Module",
                    path=file_path,
                    source=source,
                    line_start=1,
                    line_end=len(lines),
                    project_id=self.project_id,
                    language="python",
                    commit_sha=self.commit_sha,
                    imports=module_imports,
                )
            )

        return chunks

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        file_path: str,
        parent_class: str | None,
        module_imports: list[str],
    ) -> CodeChunk:
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        source_segment = "\n".join(lines[start - 1 : end])

        visitor = _CallVisitor()
        visitor.visit(node)

        docstring = ast.get_docstring(node) or ""
        qualified_name = f"{parent_class}.{node.name}" if parent_class else node.name

        return CodeChunk(
            name=qualified_name,
            chunk_type="Function",
            path=file_path,
            source=source_segment,
            line_start=start,
            line_end=end,
            project_id=self.project_id,
            language="python",
            commit_sha=self.commit_sha,
            calls=list(set(visitor.calls)),
            imports=module_imports,
            docstring=docstring,
        )

    def _extract_class_chunk(
        self,
        node: ast.ClassDef,
        lines: list[str],
        file_path: str,
        module_imports: list[str],
    ) -> CodeChunk:
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        source_segment = "\n".join(lines[start - 1 : end])

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{getattr(base.value, 'id', '')}.{base.attr}")

        visitor = _CallVisitor()
        visitor.visit(node)
        docstring = ast.get_docstring(node) or ""

        return CodeChunk(
            name=node.name,
            chunk_type="Class",
            path=file_path,
            source=source_segment,
            line_start=start,
            line_end=end,
            project_id=self.project_id,
            language="python",
            commit_sha=self.commit_sha,
            calls=list(set(visitor.calls)),
            imports=module_imports,
            inherits=bases,
            docstring=docstring,
        )

    def parse_file(self, file_path: str, content: str | None = None) -> list[CodeChunk]:
        """Convenience method to parse a local file or provided content."""
        if content is None:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".py":
            return self.parse_python(content, file_path)

        # Basic fallback for non-python files (Markdown, JSON, etc.)
        lines = content.splitlines()
        lang = "javascript" if ext in [".js", ".ts", ".jsx", ".tsx"] else ext.lstrip(".")
        return [
            CodeChunk(
                name=os.path.basename(file_path),
                chunk_type="Module",
                path=file_path,
                source=content,
                line_start=1,
                line_end=max(len(lines), 1),
                project_id=self.project_id,
                language=lang or "text",
                commit_sha=self.commit_sha,
            )
        ]
