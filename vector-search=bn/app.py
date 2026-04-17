import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

st.set_page_config(page_title="Hybrid Search Demo", page_icon="🔎", layout="wide")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


HOST_ES = get_env("HOST_ES")
ES_USERNAME = get_env("ES_USERNAME")
ES_PASSWORD = get_env("ES_PASSWORD")
ES_INDEX = get_env("ES_INDEX", "test-embeddings")
EMBEDDING_API_URL = get_env("EMBEDDING_API_URL", "http://localhost:9001/qwen3")
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY", "test")
REQUEST_TIMEOUT = int(get_env("REQUEST_TIMEOUT", "30"))
DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "10"))
DEFAULT_VECTOR_WEIGHT = float(get_env("DEFAULT_VECTOR_WEIGHT", "0.7"))
DEFAULT_LEXICAL_WEIGHT = float(get_env("DEFAULT_LEXICAL_WEIGHT", "0.3"))
DEFAULT_VECTOR_THRESHOLD = float(get_env("DEFAULT_VECTOR_THRESHOLD", "0.3"))
DEFAULT_FINAL_THRESHOLD = float(get_env("DEFAULT_FINAL_THRESHOLD", "0.4"))


@st.cache_resource
def get_es_client() -> Elasticsearch:
    if not HOST_ES:
        raise ValueError("Missing HOST_ES in .env")

    return Elasticsearch(
        hosts=HOST_ES,
        http_auth=(ES_USERNAME, ES_PASSWORD),
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )


def get_small_embedding_api(text: str, emb_type: str = "query") -> List[float]:
    payload = {"inputs": f"{emb_type}: {text[:30000]}"}
    response = requests.post(
        EMBEDDING_API_URL,
        json=payload,
        headers={"x-api-key": EMBEDDING_API_KEY},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    embs = response.json()
    if not embs:
        raise ValueError("Embedding API returned empty response")

    if isinstance(embs[0], list):
        return [round(float(elem), 4) for elem in embs[0]]
    return [round(float(elem), 4) for elem in embs]


def normalize_scores(hits: List[Dict[str, Any]], score_key: str = "_score") -> List[Dict[str, Any]]:
    if not hits:
        return hits

    scores = [float(hit.get(score_key, 0.0) or 0.0) for hit in hits]
    min_score = min(scores)
    max_score = max(scores)

    for hit in hits:
        raw_score = float(hit.get(score_key, 0.0) or 0.0)
        if max_score == min_score:
            hit["_normalized_score"] = 1.0 if max_score > 0 else 0.0
        else:
            hit["_normalized_score"] = (raw_score - min_score) / (max_score - min_score)

    return hits


def search_vector(es: Elasticsearch, query: str, size: int, vector_threshold: float) -> List[Dict[str, Any]]:
    query_vector = get_small_embedding_api(query)

    response = es.search(
        index=ES_INDEX,
        body={
            "size": size,
            "_source": ["text", "metadata"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": size,
                    }
                }
            },
        },
    )

    hits = response.get("hits", {}).get("hits", [])
    if vector_threshold > 0:
        hits = [hit for hit in hits if float(hit.get("_score", 0.0) or 0.0) >= vector_threshold]
    return hits


def search_lexical(es: Elasticsearch, query: str, size: int) -> List[Dict[str, Any]]:
    response = es.search(
        index=ES_INDEX,
        body={
            "size": size,
            "_source": ["text", "metadata"],
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "text^2",
                        "metadata.title^3",
                        "metadata.content",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
            "highlight": {
                "fields": {
                    "text": {},
                    "metadata.content": {},
                }
            },
        },
    )
    return response.get("hits", {}).get("hits", [])


def search_hybrid(
    es: Elasticsearch,
    query: str,
    size: int,
    vector_threshold: float,
    final_threshold: float,
    vector_weight: float,
    lexical_weight: float,
) -> List[Dict[str, Any]]:
    vector_hits = normalize_scores(search_vector(es, query, size * 3, vector_threshold))
    lexical_hits = normalize_scores(search_lexical(es, query, size * 3))

    merged: Dict[str, Dict[str, Any]] = {}

    for hit in vector_hits:
        doc_id = hit.get("_id")
        merged[doc_id] = {
            **hit,
            "vector_score": float(hit.get("_score", 0.0) or 0.0),
            "vector_norm": float(hit.get("_normalized_score", 0.0) or 0.0),
            "lexical_score": 0.0,
            "lexical_norm": 0.0,
        }

    for hit in lexical_hits:
        doc_id = hit.get("_id")
        if doc_id not in merged:
            merged[doc_id] = {
                **hit,
                "vector_score": 0.0,
                "vector_norm": 0.0,
                "lexical_score": float(hit.get("_score", 0.0) or 0.0),
                "lexical_norm": float(hit.get("_normalized_score", 0.0) or 0.0),
            }
        else:
            merged[doc_id]["lexical_score"] = float(hit.get("_score", 0.0) or 0.0)
            merged[doc_id]["lexical_norm"] = float(hit.get("_normalized_score", 0.0) or 0.0)
            merged[doc_id]["highlight"] = hit.get("highlight")

    results = []
    for item in merged.values():
        final_score = (item.get("vector_norm", 0.0) * vector_weight) + (
            item.get("lexical_norm", 0.0) * lexical_weight
        )
        item["final_score"] = round(final_score, 4)

        if item["final_score"] >= final_threshold:
            results.append(item)

    results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return results[:size]


def clean_preview(source: Dict[str, Any], limit: int = 450) -> str:
    metadata = source.get("metadata", {}) or {}
    text = source.get("text") or metadata.get("content") or ""
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def render_result(hit: Dict[str, Any], rank: int, mode: str) -> None:
    source = hit.get("_source", {}) or {}
    metadata = source.get("metadata", {}) or {}
    title = metadata.get("title") or f"Result {rank}"
    link_id = metadata.get("link_id") or hit.get("_id", "-")

    with st.container(border=True):
        col1, col2 = st.columns([5, 2])

        with col1:
            st.subheader(f"{rank}. {title}")
            st.caption(f"ID: {link_id}")

        with col2:
            if mode == "Hybrid":
                st.metric("Final score", f"{hit.get('final_score', 0.0):.4f}")
            else:
                st.metric("Score", f"{float(hit.get('_score', 0.0) or 0.0):.4f}")

        if mode == "Hybrid":
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Vector raw", f"{hit.get('vector_score', 0.0):.4f}")
            s2.metric("Vector norm", f"{hit.get('vector_norm', 0.0):.4f}")
            s3.metric("Lexical raw", f"{hit.get('lexical_score', 0.0):.4f}")
            s4.metric("Lexical norm", f"{hit.get('lexical_norm', 0.0):.4f}")

        st.write(clean_preview(source))

        with st.expander("Metadata"):
            st.json(metadata)


def main() -> None:
    st.title("🔎 OpenSearch Search UI")
    st.write("Choose search type from the left side. Only required filters will appear.")

    with st.sidebar:
        st.header("Search settings")

        search_mode = st.selectbox(
            "Search type",
            ["Hybrid", "Vector", "Lexical"],
            help="Hybrid = meaning + exact words, Vector = meaning only, Lexical = exact words only.",
        )

        st.divider()

        if search_mode == "Hybrid":
            st.info("Hybrid = meaning search + exact word search. Best for most cases.")

            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=50,
                value=DEFAULT_TOP_K,
                help="How many results to show. Min: 1, Max: 50, Recommended: 5 to 15.",
            )

            vector_threshold = st.number_input(
                "Vector threshold",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_VECTOR_THRESHOLD,
                step=0.05,
                help="Minimum vector score to keep a result. Min: 0.0, Max: 2.0, Recommended: 0.2 to 1.0.",
            )

            final_threshold = st.number_input(
                "Final threshold",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_FINAL_THRESHOLD,
                step=0.05,
                help="Minimum final hybrid score to keep a result. Min: 0.0, Max: 1.0, Recommended: 0.3 to 0.8.",
            )

            vector_weight = st.slider(
                "Vector weight",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_VECTOR_WEIGHT,
                step=0.05,
                help="How much meaning matters in hybrid. Min: 0.0, Max: 1.0, Recommended: 0.6 to 0.8.",
            )

            lexical_weight = st.slider(
                "Lexical weight",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_LEXICAL_WEIGHT,
                step=0.05,
                help="How much exact words matter in hybrid. Min: 0.0, Max: 1.0, Recommended: 0.2 to 0.4.",
            )

            total_weight = vector_weight + lexical_weight
            if abs(total_weight - 1.0) > 0.001:
                st.warning("Keep Vector weight + Lexical weight = 1.0")
            else:
                st.success("Good: weights add up to 1.0")

            st.markdown(
                """
**Meaning**
- **Top K** = number of results
- **Vector threshold** = remove weak meaning matches first
- **Final threshold** = remove weak final hybrid results
- **Vector weight** = importance of meaning
- **Lexical weight** = importance of exact words
                """
            )

        elif search_mode == "Vector":
            st.info("Vector = meaning search only.")

            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=50,
                value=DEFAULT_TOP_K,
                help="How many results to show. Min: 1, Max: 50, Recommended: 5 to 15.",
            )

            vector_threshold = st.number_input(
                "Vector threshold",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_VECTOR_THRESHOLD,
                step=0.05,
                help="Minimum vector score to keep a result. Min: 0.0, Max: 2.0, Recommended: 0.2 to 1.0.",
            )

            final_threshold = 0.0
            vector_weight = 1.0
            lexical_weight = 0.0

            st.markdown(
                """
**Meaning**
- **Top K** = number of results
- **Vector threshold** = remove weak meaning matches
                """
            )

        else:
            st.info("Lexical = exact word search only.")

            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=50,
                value=DEFAULT_TOP_K,
                help="How many results to show. Min: 1, Max: 50, Recommended: 5 to 15.",
            )

            vector_threshold = 0.0
            final_threshold = 0.0
            vector_weight = 0.0
            lexical_weight = 1.0

            st.markdown(
                """
**Meaning**
- **Top K** = number of results
- No threshold or weight needed for lexical search
                """
            )

        st.divider()
        st.caption(f"Index: {ES_INDEX}")
        st.caption(f"Embedding endpoint: {EMBEDDING_API_URL}")

    with st.expander("What each search type means"):
        st.markdown(
            """
### Hybrid
Uses **meaning + exact words**. Best default.

### Vector
Uses **meaning only**. Good for natural language search.

### Lexical
Uses **exact words only**. Good for names, ids, tags, and strict keywords.
            """
        )

    query = st.text_input("Enter your query", placeholder="e.g. Law meetings in USA")
    run_search = st.button("Search", type="primary", use_container_width=True)

    if not query:
        st.info("Enter a query to start.")
        return

    if run_search:
        try:
            es = get_es_client()

            with st.spinner("Searching..."):
                if search_mode == "Vector":
                    results = search_vector(es, query, top_k, vector_threshold)
                elif search_mode == "Lexical":
                    results = search_lexical(es, query, top_k)
                else:
                    results = search_hybrid(
                        es=es,
                        query=query,
                        size=top_k,
                        vector_threshold=vector_threshold,
                        final_threshold=final_threshold,
                        vector_weight=vector_weight,
                        lexical_weight=lexical_weight,
                    )

            st.success(f"Found {len(results)} result(s)")

            if not results:
                st.warning("No results matched your filters.")
                return

            for idx, hit in enumerate(results, start=1):
                render_result(hit, idx, search_mode)

            with st.expander("Raw response preview"):
                st.json(results)

        except Exception as exc:
            st.error(f"Search failed: {exc}")


if __name__ == "__main__":
    main()