"""Three retrieval methods for Day 7: Lexical (TF-IDF), Dense (Day 6 pgvector), and Hybrid (RRF)."""
import json
from typing import Any
import psycopg # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from baseline import TfidfIndex

SECTION_KEYWORDS = {
    "Executive Summary": ["executive summary", "overview", "summary"],
    "Energy": ["energy", "oil", "brent", "natural gas", "lng", "coal", "diesel", "refining"],
    "Agriculture": ["agriculture", "food", "grain", "wheat", "maize", "rice", "palm oil", "soybean", "coffee", "arabica", "cocoa", "tea", "beverage"],
    "Fertilizers": ["fertilizer", "fertilizers", "urea", "potash", "mop", "dap", "phosphate", "nitrogen"],
    "Metals and minerals": ["metals", "minerals", "copper", "aluminum", "gold", "silver", "platinum", "cobalt", "lithium", "iron ore", "rare earth"],
    "Special Focus Commodity Price Connections: How Does Volatility Spread?": ["special focus", "connectedness", "volatility spread", "diebold-yilmaz", "net transmitter"],
}


def infer_section_filter(query: str) -> dict[str, Any] | None:
    """Rule-based section inference from explicit section mentions in query (never uses gold)."""
    q = query.lower()
    if "special focus" in q:
        return {"section": "Special Focus Commodity Price Connections: How Does Volatility Spread?"}
    for sec in ("Fertilizers", "Agriculture", "Metals and minerals", "Executive Summary"):
        if sec.lower() in q:
            return {"section": sec}
    return None


def _matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for k, v in filters.items():
        if meta.get(k) != v:
            return False
    return True


class LexicalRetriever:
    """Method 1: Offline TF-IDF lexical retrieval."""
    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]], use_filter: bool = False):
        self.index = TfidfIndex(chunks)
        self.use_filter = use_filter

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        active_filter = filters if self.use_filter else None
        raw = self.index.search(query, top_k=max(top_k * 5, 25))
        filtered = [c for c in raw if _matches_filter(c, active_filter)][:top_k]
        return [{**c, "strategy": self.name} for c in filtered]


class DensePgVectorRetriever:
    """Method 2: Day 6 pgvector dense retrieval with sentence-transformers/all-MiniLM-L6-v2."""
    name = "dense"

    def __init__(
        self,
        db_url: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config_name: str = "config_c",
        use_filter: bool = True,
        auto_infer_filter: bool = False,
    ):
        self.db_url = db_url
        self.model = SentenceTransformer(model_name)
        self.config_name = config_name
        self.use_filter = use_filter
        self.auto_infer_filter = auto_infer_filter

    def _encode(self, query: str) -> str:
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        active_filter = filters if self.use_filter else None
        if active_filter is None and self.auto_infer_filter:
            active_filter = infer_section_filter(query)

        qvec_str = self._encode(query)
        where_clauses = ["strategy = %s"]
        params: list[Any] = [qvec_str, self.config_name]

        if active_filter:
            for k, v in active_filter.items():
                if k == "section":
                    where_clauses.append("section = %s")
                    params.append(str(v))
                else:
                    where_clauses.append("metadata ->> %s = %s")
                    params.extend([k, str(v)])

        params.extend([qvec_str, top_k])
        sql = f"""
            SELECT chunk_id, document_id, source, page, section, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
        """
        out: list[dict[str, Any]] = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for chunk_id, doc_id, source, page, section, text, meta, score in cur.fetchall():
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    meta_dict = dict(meta or {})
                    if section and "section" not in meta_dict:
                        meta_dict["section"] = section
                    out.append({
                        "id": chunk_id,
                        "source_id": doc_id,
                        "source": source,
                        "page": page,
                        "text": text,
                        "metadata": meta_dict,
                        "score": float(score),
                        "strategy": self.name,
                    })
        return out


class HybridRRFRetriever:
    """Method 3: Reciprocal Rank Fusion combining Dense (pgvector) and Lexical (TF-IDF)."""
    name = "hybrid_rrf"

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: LexicalRetriever,
        rrf_k: int = 60,
        candidate_pool: int = 12,
        auto_infer_filter: bool = True,
    ):
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.auto_infer_filter = auto_infer_filter

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        active_filter = filters or (infer_section_filter(query) if self.auto_infer_filter else None)
        dense_hits = self.dense.search(query, filters=active_filter, top_k=self.candidate_pool)
        lex_hits = self.lexical.search(query, filters=active_filter, top_k=self.candidate_pool)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{**by_id[cid], "score": rrf_score, "strategy": self.name} for cid, rrf_score in ranked]