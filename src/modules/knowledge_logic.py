import json

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# ══════════════════════════════════════════════════════════════════════
#  Knowledge Logic (Local RAG)
# ════════════════════════════════════════════════─═════════════════════

class KnowledgeLogic:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._model = None # Lazy loaded

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    @staticmethod
    def _chunk_text(content: str, chunk_size: int = 1000, stride: int = 800) -> List[str]:
        if not content:
            return []
        return [content[i:i + chunk_size] for i in range(0, len(content), stride)]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def load(self) -> bool:
        try:
            import faiss
        except ImportError:
            return False
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            # Safe JSON deserialization instead of pickle
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        return False

    def save(self):
        try:
            import faiss
        except ImportError:
            return
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
            # Safe JSON serialization instead of pickle
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)

    def index_files(self, root_dir: Path, extensions: List[str] = [".py", ".md", ".txt"]):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Vector indexing requires the optional knowledge extras "
                "(`faiss-cpu` and `sentence-transformers`)."
            ) from exc

        documents = []
        metas = []
        
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                try:
                    content = p.read_text(errors="ignore")
                    chunks = self._chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metas.append({"path": str(p), "chunk": i, "text": chunk})
                except Exception:
                    continue
        
        if not documents:
            return 0
            
        embeddings = self.model.encode(documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = metas
        self.save()
        return len(documents)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            import numpy as np
        except ImportError:
            return []
        if not self.index:
            if not self.load():
                return []
        
        try:
            query_vec = self.model.encode([query_text])
        except ImportError:
            return []
        distances, indices = self.index.search(np.array(query_vec).astype("float32"), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                meta = self.metadata[idx]
                results.append({
                    "path": meta["path"],
                    "score": float(distances[0][i]),
                    "chunk_id": meta["chunk"],
                    "text": meta.get("text", ""),
                })
        return results

    def query_repo(self, query_text: str, root_dir: Path, k: int = 5) -> List[Dict[str, Any]]:
        query = query_text.strip().lower()
        if not query:
            return []

        query_tokens = self._tokenize(query)
        ignore_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        matches: List[Dict[str, Any]] = []

        for path in root_dir.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".txt"}:
                continue
            if any(part in ignore_dirs for part in path.parts):
                continue
            try:
                content = path.read_text(errors="ignore")
            except Exception:
                continue
            for i, chunk in enumerate(self._chunk_text(content)):
                lowered = chunk.lower()
                phrase_hits = lowered.count(query)
                token_hits = len(query_tokens & self._tokenize(chunk))
                if phrase_hits == 0 and token_hits == 0:
                    continue
                matches.append(
                    {
                        "path": str(path),
                        "score": float(phrase_hits * 100 + token_hits),
                        "chunk_id": i,
                        "text": chunk,
                    }
                )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:k]

def get_logic() -> KnowledgeLogic:
    return KnowledgeLogic(Path.home() / ".elfweave_knowledge")
