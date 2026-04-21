import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

st.set_page_config(page_title="Index Comparison Search UI", page_icon="🔎", layout="wide")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


# -----------------------------
# Config
# -----------------------------
HOST_ES = get_env("HOST_ES")
ES_USERNAME = get_env("ES_USERNAME")
ES_PASSWORD = get_env("ES_PASSWORD")
EMBEDDING_API_URL = get_env("EMBEDDING_API_URL", "http://localhost:9001/qwen3")
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY", "test")
REQUEST_TIMEOUT = int(get_env("REQUEST_TIMEOUT", "30"))

DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "10"))
DEFAULT_VECTOR_WEIGHT = float(get_env("DEFAULT_VECTOR_WEIGHT", "0.7"))
DEFAULT_LEXICAL_WEIGHT = float(get_env("DEFAULT_LEXICAL_WEIGHT", "0.3"))
DEFAULT_VECTOR_THRESHOLD = float(get_env("DEFAULT_VECTOR_THRESHOLD", "0.3"))
DEFAULT_FINAL_THRESHOLD = float(get_env("DEFAULT_FINAL_THRESHOLD", "0.4"))

INDEX_CONFIGS = {
    "qwen-embeddings-v2": {
        "label": "Full Markdown Index",
        "index_name": "qwen-embeddings-v2",
        "embedding_field": "embedding",
        "text_field": "markdown",
        "title_field": "meta_data_without_llm.title",
        "content_field": "meta_data_without_llm.content",
        "link_id_field": "link_id",
        "url_field": "url",
        "tags_field": "meta_data_with_llm.tags",
        "entities_field": "meta_data_with_llm.entities",
        "content_type_field": "meta_data_with_llm.content_type",
        "chunk_index_field": None,
        "chunk_total_field": None,
    },
    "test-embeddings-chunks": {
        "label": "Chunked Markdown Index",
        "index_name": "test-embeddings-chunks",
        "embedding_field": "embedding",
        "text_field": "markdown",
        "title_field": "meta_data_without_llm.title",
        "content_field": "meta_data_without_llm.content",
        "link_id_field": "link_id",
        "url_field": "url",
        "tags_field": "meta_data_with_llm.tags",
        "entities_field": "meta_data_with_llm.entities",
        "content_type_field": "meta_data_with_llm.content_type",
        "chunk_index_field": "chunk_index",
        "chunk_total_field": "chunk_total"
    },
}


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


def get_nested_value(data: Dict[str, Any], dotted_key: Optional[str], default: Any = None) -> Any:
    if not dotted_key:
        return default

    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def build_source_fields(index_cfg: Dict[str, Any]) -> List[str]:
    fields = [
        index_cfg["text_field"],
        index_cfg["title_field"],
        index_cfg["content_field"],
        index_cfg["link_id_field"],
        index_cfg["url_field"],
        index_cfg["tags_field"],
        index_cfg["entities_field"],
        index_cfg["content_type_field"],
    ]
    if index_cfg.get("chunk_index_field"):
        fields.append(index_cfg["chunk_index_field"])
    if index_cfg.get("chunk_total_field"):
        fields.append(index_cfg["chunk_total_field"])
    return fields

def search_vector(
    es: Elasticsearch,
    index_cfg: Dict[str, Any],
    query: str,
    size: int,
    vector_threshold: float,
) -> List[Dict[str, Any]]:
    query_vector = get_small_embedding_api(query)

    response = es.search(
        index=index_cfg["index_name"],
        body={
            "size": size,
            "_source": build_source_fields(index_cfg),
            "query": {
                "knn": {
                    index_cfg["embedding_field"]: {
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


def search_lexical(
    es: Elasticsearch,
    index_cfg: Dict[str, Any],
    query: str,
    size: int,
) -> List[Dict[str, Any]]:
    response = es.search(
        index=index_cfg["index_name"],
        body={
            "size": size,
            "_source": build_source_fields(index_cfg),
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        f"{index_cfg['text_field']}^2",
                        f"{index_cfg['title_field']}^3",
                        index_cfg["content_field"],
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
            "highlight": {
                "fields": {
                    index_cfg["text_field"]: {},
                    index_cfg["content_field"]: {},
                }
            },
        },
    )
    return response.get("hits", {}).get("hits", [])


def search_hybrid(
    es: Elasticsearch,
    index_cfg: Dict[str, Any],
    query: str,
    size: int,
    vector_threshold: float,
    final_threshold: float,
    vector_weight: float,
    lexical_weight: float,
) -> List[Dict[str, Any]]:
    vector_hits = normalize_scores(search_vector(es, index_cfg, query, size * 3, vector_threshold))
    lexical_hits = normalize_scores(search_lexical(es, index_cfg, query, size * 3))

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


def clean_preview(text: str, limit: int = 450) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def extract_result_fields(hit: Dict[str, Any], index_cfg: Dict[str, Any]) -> Dict[str, Any]:
    source = hit.get("_source", {}) or {}

    title = get_nested_value(source, index_cfg["title_field"], "Untitled")
    content = get_nested_value(source, index_cfg["content_field"], "")
    markdown = get_nested_value(source, index_cfg["text_field"], "")
    link_id = get_nested_value(source, index_cfg["link_id_field"], hit.get("_id", "-"))
    url = get_nested_value(source, index_cfg["url_field"], "")
    tags = get_nested_value(source, index_cfg["tags_field"], []) or []
    entities = get_nested_value(source, index_cfg["entities_field"], []) or []
    content_type = get_nested_value(source, index_cfg["content_type_field"], "")
    chunk_index = get_nested_value(source, index_cfg.get("chunk_index_field"))
    chunk_total = get_nested_value(source, index_cfg.get("chunk_total_field"))

    preview = content or markdown or title

    raw_view = {
        "title": title,
        "content": content,
        "tags": tags,
        "entities": entities,
        "content_type": content_type,
    }

    return {
        "title": title,
        "content": content,
        "markdown": markdown,
        "link_id": link_id,
        "url": url,
        "tags": tags,
        "entities": entities,
        "content_type": content_type,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "preview": clean_preview(preview),
        "raw_view": raw_view,
        "source": source,
    }

def render_result(hit: Dict[str, Any], rank: int, mode: str, index_cfg: Dict[str, Any]) -> None:
    parsed = extract_result_fields(hit, index_cfg)

    with st.container(border=True):
        col1, col2 = st.columns([5, 2])

        with col1:
            st.subheader(f"{rank}. {parsed['title']}")
            st.caption(f"Link ID: {parsed['link_id']}")

            if parsed["url"]:
                st.caption(parsed["url"])

            if parsed["chunk_index"] is not None and parsed["chunk_total"] is not None:
                st.caption(f"Chunk: {parsed['chunk_index'] + 1} / {parsed['chunk_total']}")

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

        st.write(parsed["preview"])

        with st.expander("Raw fields"):
            st.json(parsed["raw_view"])


def run_search_for_index(
    es: Elasticsearch,
    index_cfg: Dict[str, Any],
    search_mode: str,
    query: str,
    top_k: int,
    vector_threshold: float,
    final_threshold: float,
    vector_weight: float,
    lexical_weight: float,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        if search_mode == "Vector":
            results = search_vector(es, index_cfg, query, top_k, vector_threshold)
        elif search_mode == "Lexical":
            results = search_lexical(es, index_cfg, query, top_k)
        else:
            results = search_hybrid(
                es=es,
                index_cfg=index_cfg,
                query=query,
                size=top_k,
                vector_threshold=vector_threshold,
                final_threshold=final_threshold,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight,
            )
        return results, None
    except Exception as exc:
        return [], str(exc)


def main() -> None:
    st.title("🔎 Index Comparison Search UI")
    st.write("Compare results from full-markdown index and chunked-markdown index side by side.")

    with st.sidebar:
        st.header("Search settings")

        compare_mode = st.checkbox("Compare both indexes", value=True)

        if not compare_mode:
            selected_index_name = st.selectbox(
                "Choose index",
                list(INDEX_CONFIGS.keys()),
                format_func=lambda x: f"{INDEX_CONFIGS[x]['label']} ({x})",
            )
        else:
            selected_index_name = None

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
                help="How many results to show. Min: 1, Max: 50.",
            )

            vector_threshold = st.number_input(
                "Vector threshold",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_VECTOR_THRESHOLD,
                step=0.05,
                help="Remove weak meaning matches before combining scores.",
            )

            final_threshold = st.number_input(
                "Final threshold",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_FINAL_THRESHOLD,
                step=0.05,
                help="Remove weak final hybrid results after combining vector and lexical scores.",
            )

            vector_weight = st.slider(
                "Vector weight",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_VECTOR_WEIGHT,
                step=0.05,
                help="How much meaning matters in hybrid.",
            )

            lexical_weight = st.slider(
                "Lexical weight",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_LEXICAL_WEIGHT,
                step=0.05,
                help="How much exact words matter in hybrid.",
            )

            total_weight = vector_weight + lexical_weight
            if abs(total_weight - 1.0) > 0.001:
                st.warning("Keep Vector weight + Lexical weight = 1.0")
            else:
                st.success("Good: weights add up to 1.0")

        elif search_mode == "Vector":
            st.info("Vector = meaning search only.")

            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=50,
                value=DEFAULT_TOP_K,
                help="How many results to show. Min: 1, Max: 50.",
            )

            vector_threshold = st.number_input(
                "Vector threshold",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_VECTOR_THRESHOLD,
                step=0.05,
                help="Remove weak meaning matches.",
            )

            final_threshold = 0.0
            vector_weight = 1.0
            lexical_weight = 0.0

        else:
            st.info("Lexical = exact word search only.")

            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=50,
                value=DEFAULT_TOP_K,
                help="How many results to show. Min: 1, Max: 50.",
            )

            vector_threshold = 0.0
            final_threshold = 0.0
            vector_weight = 0.0
            lexical_weight = 1.0

        st.divider()
        st.markdown(
            """
**What the filters mean**
- **Top K** = how many results to show
- **Vector threshold** = remove weak meaning matches
- **Final threshold** = remove weak final hybrid results
- **Vector weight** = importance of meaning
- **Lexical weight** = importance of exact words
            """
        )

    with st.expander("What each search type means"):
        st.markdown(
            """
### Hybrid
Uses **meaning + exact words**. Best default.

### Vector
Uses **meaning only**. Good for natural language search.

### Lexical
Uses **exact words only**. Good for names, tags, ids, and strict keyword matching.
            """
        )

    query = st.text_input("Enter your query", placeholder="e.g. agenda meetings")
    run_search = st.button("Search", type="primary", use_container_width=True)

    if not query:
        st.info("Enter a query to start.")
        return

    if run_search:
        try:
            es = get_es_client()

            if compare_mode:
                left_index = INDEX_CONFIGS["qwen-embeddings-v2"]
                right_index = INDEX_CONFIGS["test-embeddings-chunks"]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"{left_index['label']}")
                    st.caption(left_index["index_name"])

                    results, error = run_search_for_index(
                        es=es,
                        index_cfg=left_index,
                        search_mode=search_mode,
                        query=query,
                        top_k=top_k,
                        vector_threshold=vector_threshold,
                        final_threshold=final_threshold,
                        vector_weight=vector_weight,
                        lexical_weight=lexical_weight,
                    )

                    if error:
                        st.error(f"Search failed on {left_index['index_name']}: {error}")
                    else:
                        st.success(f"Found {len(results)} result(s)")
                        if not results:
                            st.warning("No results found.")
                        for idx, hit in enumerate(results, start=1):
                            render_result(hit, idx, search_mode, left_index)

                with col2:
                    st.subheader(f"{right_index['label']}")
                    st.caption(right_index["index_name"])

                    results, error = run_search_for_index(
                        es=es,
                        index_cfg=right_index,
                        search_mode=search_mode,
                        query=query,
                        top_k=top_k,
                        vector_threshold=vector_threshold,
                        final_threshold=final_threshold,
                        vector_weight=vector_weight,
                        lexical_weight=lexical_weight,
                    )

                    if error:
                        st.error(f"Search failed on {right_index['index_name']}: {error}")
                    else:
                        st.success(f"Found {len(results)} result(s)")
                        if not results:
                            st.warning("No results found.")
                        for idx, hit in enumerate(results, start=1):
                            render_result(hit, idx, search_mode, right_index)

            else:
                index_cfg = INDEX_CONFIGS[selected_index_name]

                st.subheader(index_cfg["label"])
                st.caption(index_cfg["index_name"])

                results, error = run_search_for_index(
                    es=es,
                    index_cfg=index_cfg,
                    search_mode=search_mode,
                    query=query,
                    top_k=top_k,
                    vector_threshold=vector_threshold,
                    final_threshold=final_threshold,
                    vector_weight=vector_weight,
                    lexical_weight=lexical_weight,
                )

                if error:
                    st.error(f"Search failed on {index_cfg['index_name']}: {error}")
                    return

                st.success(f"Found {len(results)} result(s)")
                if not results:
                    st.warning("No results found.")
                    return

                for idx, hit in enumerate(results, start=1):
                    render_result(hit, idx, search_mode, index_cfg)

        except Exception as exc:
            st.error(f"Search failed: {exc}")


if __name__ == "__main__":
    main()