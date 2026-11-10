"""Retrieval engine implementing configurations A, B, and C with transparent indexing."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import Citation
from app.document_processor import DocumentProcessor

logger = logging.getLogger("relay.retrieval")


class RetrievalEngine:
    """Retrieval Engine providing configurable chunking, TF-IDF vector retrieval,

    and optional hybrid term-density reranking.
    """

    CONFIG_PARAMS = {
        "A": {"chunk_size": 300, "chunk_overlap": 50, "top_k": 3, "rerank": False},
        "B": {"chunk_size": 600, "chunk_overlap": 100, "top_k": 5, "rerank": False},
        "C": {"chunk_size": 600, "chunk_overlap": 100, "top_k": 5, "rerank": True},
    }

    def __init__(self, papers_dir: Path, processed_dir: Path):
        self.papers_dir = papers_dir
        self.processed_dir = processed_dir
        self.doc_processor = DocumentProcessor(papers_dir, processed_dir)

        # In-memory indices per configuration
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.matrices: Dict[str, Any] = {}
        self.is_initialized = False

    def initialize(self, force_rebuild: bool = False):
        """Build or load chunk sets and vector indices for settings A, B, and C."""
        for cfg_name, params in self.CONFIG_PARAMS.items():
            self._load_or_build_config(
                cfg_name,
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
                force_rebuild=force_rebuild,
            )
        self.is_initialized = True
        logger.info("RetrievalEngine successfully initialized for configs A, B, and C.")

    def _load_or_build_config(
        self,
        cfg_name: str,
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False,
    ):
        """Loads or builds chunks and scikit-learn TF-IDF matrix for a configuration."""
        chunks = self.doc_processor.process_all_papers(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_rebuild=force_rebuild,
            config_name=cfg_name,
        )
        self.chunks[cfg_name] = chunks

        matrix_path = self.processed_dir / f"tfidf_matrix_{cfg_name}.pkl"
        vectorizer_path = self.processed_dir / f"vectorizer_{cfg_name}.pkl"

        if not force_rebuild and matrix_path.exists() and vectorizer_path.exists():
            try:
                with open(vectorizer_path, "rb") as vf:
                    vectorizer = pickle.load(vf)
                with open(matrix_path, "rb") as mf:
                    matrix = pickle.load(mf)
                self.vectorizers[cfg_name] = vectorizer
                self.matrices[cfg_name] = matrix
                logger.info(f"Loaded existing TF-IDF index for config {cfg_name}")
                return
            except Exception as e:
                logger.warning(f"Error loading cached index {cfg_name}: {e}. Rebuilding...")

        # Build TF-IDF vectorizer
        corpus = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(corpus)

        with open(vectorizer_path, "wb") as vf:
            pickle.dump(vectorizer, vf)
        with open(matrix_path, "wb") as mf:
            pickle.dump(matrix, mf)

        self.vectorizers[cfg_name] = vectorizer
        self.matrices[cfg_name] = matrix
        logger.info(f"Built TF-IDF index for config {cfg_name} over {len(corpus)} passages.")

    def _rerank_candidates(
        self, query: str, candidate_indices: List[int], base_scores: List[float], cfg_name: str
    ) -> List[Tuple[int, float]]:
        """Setting C reranker: Re-scores candidates using hybrid lexical overlap,

        exact match density, and TF-IDF cosine score.
        """
        query_terms = set(query.lower().split())
        query_terms = {t for t in query_terms if len(t) > 2}

        scored_candidates = []
        for idx, base_score in zip(candidate_indices, base_scores):
            chunk = self.chunks[cfg_name][idx]
            text_lower = chunk["text"].lower()

            # Exact query terms matching density
            matches = sum(1 for term in query_terms if term in text_lower)
            term_overlap_ratio = matches / max(1, len(query_terms))

            # Query exact phrase bonus
            phrase_bonus = 0.2 if query.lower()[:30] in text_lower else 0.0

            # Composite rerank score
            rerank_score = float(0.6 * base_score + 0.3 * term_overlap_ratio + 0.1 * phrase_bonus)
            scored_candidates.append((idx, rerank_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def retrieve(
        self,
        query: str,
        retrieval_setting: str = "B",
        custom_top_k: Optional[int] = None,
    ) -> Tuple[List[Citation], float]:
        """Retrieve top passages for the given query using the selected setting."""
        if not self.is_initialized:
            self.initialize()

        cfg = retrieval_setting if retrieval_setting in self.CONFIG_PARAMS else "B"
        params = self.CONFIG_PARAMS[cfg]
        top_k = custom_top_k or params["top_k"]

        vectorizer = self.vectorizers[cfg]
        matrix = self.matrices[cfg]
        chunks = self.chunks[cfg]

        # Vectorize query
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()

        # Handle zero-match edge case
        if sims.max() == 0.0:
            return [], 0.0

        max_score = float(sims.max())

        if params["rerank"]:
            # Retrieve wider pool of top_k * 2 candidates and rerank
            candidate_pool_size = min(len(sims), top_k * 3)
            pool_indices = np.argsort(sims)[::-1][:candidate_pool_size].tolist()
            pool_scores = [float(sims[i]) for i in pool_indices]
            ranked = self._rerank_candidates(query, pool_indices, pool_scores, cfg)[:top_k]
        else:
            top_indices = np.argsort(sims)[::-1][:top_k].tolist()
            ranked = [(i, float(sims[i])) for i in top_indices]

        citations: List[Citation] = []
        for idx, score in ranked:
            c = chunks[idx]
            citations.append(
                Citation(
                    paper_title=c["paper_title"],
                    arxiv_id=c["arxiv_id"],
                    version=c["version"],
                    page=c["page"],
                    section=c.get("section"),
                    score=round(score, 4),
                    text_snippet=c["text"][:350] + ("..." if len(c["text"]) > 350 else ""),
                )
            )

        return citations, max_score

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Return corpus and passage statistics for observability."""
        return {
            "total_documents": 3,
            "config_A_passages": len(self.chunks.get("A", [])),
            "config_B_passages": len(self.chunks.get("B", [])),
            "config_C_passages": len(self.chunks.get("C", [])),
            "retrieval_engine": "scikit-learn TF-IDF + hybrid term-density reranking",
        }

