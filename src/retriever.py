"""Semantic retriever over the KB chunks."""

import random
import numpy as np

from src.config import EMBED_MODEL_ID


class Retriever:
    def __init__(self, chunks):
        from sentence_transformers import SentenceTransformer
        print("   Loading embedding model...")
        self.model      = SentenceTransformer(EMBED_MODEL_ID, local_files_only=True)
        self.chunks     = chunks
        texts           = [c.full_text[:512] for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"   ✅ Embedded {len(chunks)} chunks")

    def retrieve(self, query, domain, top_k=6):
        top_k *= 2
        idx = [i for i, c in enumerate(self.chunks) if c.domain == domain]
        if not idx:
            idx = list(range(len(self.chunks)))
        q_emb  = self.model.encode([query])
        subset = self.embeddings[idx]
        q_n    = q_emb  / (np.linalg.norm(q_emb,  axis=1, keepdims=True) + 1e-10)
        s_n    = subset / (np.linalg.norm(subset, axis=1, keepdims=True) + 1e-10)
        scores = (q_n @ s_n.T).flatten()
        top    = [idx[i] for i in np.argsort(scores)[::-1][:top_k]]
        result = [self.chunks[i] for i in top]
        result = random.sample(result, len(result) // 2)
        print(f"   ✅ Retrieved: {[c.chunk_id for c in result]}")
        return result
