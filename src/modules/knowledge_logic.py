import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  Knowledge Logic (Local RAG)
# ════════════════════════════════════════════════─═════════════════════

class KnowledgeLogic:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.pkl"
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._model = None # Lazy loaded

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def load(self) -> bool:
        import faiss
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def save(self):
        import faiss
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def index_files(self, root_dir: Path, extensions: List[str] = [".py", ".md", ".txt"]):
        import faiss
        documents = []
        metas = []
        
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                try:
                    content = p.read_text(errors="ignore")
                    # Chunking: Simple fixed-size for prototype
                    chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
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
        import faiss
        if not self.index:
            if not self.load():
                return []
        
        query_vec = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
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

def get_logic() -> KnowledgeLogic:
    return KnowledgeLogic(Path.home() / ".elfweave_knowledge")
