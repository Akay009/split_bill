"""
Beyond News MCP — one page, tabs: Search · Charts · Index fields · Advanced.

Env / secrets: BN_MCP_URL, BN_API_TOKEN; OpenAI or Anthropic keys for the Search planner.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from bn_client import BNClient
from openai_planner import plan_mcp_call
from result_views import dataframe_for_st, render_aggregations, render_list_fields_clean, render_mcp_result
from ui_shared import hero, inject_theme, sidebar_connection

MANUAL_TOOLS = (
    "check_health",
    "explore",
    "list_fields",
    "links_aggregations",
    "search_links",
    "search_links_post",
    "search_documents",
    "search_documents_post",
)

MANUAL_DEFAULTS: dict[str, dict] = {
    "check_health": {},
    "explore": {},
    "list_fields": {"index": "links"},
    "links_aggregations": {"facet": "content_type", "agg_size": 50},
    "search_links": {"q": "*", "page": 1, "page_size": 10, "sort_by": "date", "sort_order": "desc"},
    "search_links_post": {"q": "*", "page": 1, "page_size": 10, "sort_by": "date", "sort_order": "desc"},
    "search_documents": {"page": 1, "size": 10, "keyword_operator": "and"},
    "search_documents_post": {"page": 1, "size": 10, "keyword_operator": "and"},
}


def fetch_content_type_values(client: BNClient) -> list[str]:
    try:
        agg = client.call_tool("links_aggregations", {"facet": "content_type", "agg_size": 200})
    except Exception:
        return []
    if not isinstance(agg, dict):
        return []
    block = agg.get("content_type")
    if not isinstance(block, dict):
        return []
    out: list[str] = []
    for o in block.get("options") or []:
        if isinstance(o, dict) and o.get("value") is not None:
            v = str(o["value"]).strip()
            if v:
                out.append(v)
    return out


def build_llm_context(client: BNClient) -> str:
    """
    Order matters: short cheat sheet first (like a tool description), then live explore.
    Cursor/Desktop see full MCP schemas; we approximate with explore + hints.
    """
    cheat = """=== Parameter cheat sheet (confirm details in explore below) ===
- search_links: q (required), page, page_size, sort_by, sort_order, from_, to_, language, sources, theme,
  content_type, has_documents, search_in, fields, url_contains, parent_url, dynamic_search (string for GET).
- search_links_post: same idea; dynamic_search may be an object.
- search_documents: keywords, keyword_operator (and|or), domains, doc_type, date_from, date_to, tags, link_id, page, size.
- links_aggregations: facet (content_type|tags|entities|breadcrumbs|data_type), agg_size, from_, to_.
- list_fields: index = links | documents.
- check_health / explore: usually no arguments.
"""
    chunks: list[str] = [cheat]
    try:
        ex = client.call_tool("explore", {})
        ex_s = json.dumps(ex, indent=2, ensure_ascii=False) if isinstance(ex, (dict, list)) else str(ex)
        # Larger slice so parameter docs are less often cut off mid-schema
        chunks.append("=== Live MCP explore (truncated) ===\n" + ex_s[:32000])
    except Exception as e:
        chunks.append(f"explore failed: {e}")
    try:
        ct = fetch_content_type_values(client)
        chunks.append("\n=== content_type values (exact strings for search_links) ===\n" + json.dumps(ct[:200], ensure_ascii=False))
    except Exception:
        pass
    return "\n".join(chunks)


def main() -> None:
    st.set_page_config(
        page_title="Beyond News",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔎",
    )
    inject_theme()

    client, llm_provider, llm_key, llm_model = sidebar_connection(show_llm=True)

    hero(
        "Beyond News Search",
        "Tabs below for search, charts, and fields. Sidebar: MCP + BN token, then pick **OpenAI** or **Anthropic (Claude)** and paste the matching API key.",
    )

    if client is None:
        st.info("Set **MCP URL** and **BN API key** in the sidebar (or `BN_MCP_URL` / `BN_API_TOKEN`).")
        return

    tab_search, tab_charts, tab_fields, tab_adv = st.tabs(["Search", "Charts", "Index fields", "Advanced"])

    with tab_search:
        prov_label = "Claude (Anthropic)" if llm_provider == "anthropic" else "OpenAI"
        st.caption(
            f"Active planner: **{prov_label}** · model **{llm_model}** — set **Provider** in the sidebar."
        )
        st.text_area(
            "Ask anything",
            height=120,
            key="nl_query",
            placeholder="e.g. Press releases about AI in English from last week, newest first",
            label_visibility="collapsed",
        )
        if st.button("Run", type="primary", key="nl_run"):
            q = str(st.session_state.get("nl_query") or "").strip()
            if not q:
                st.warning("Enter a question.")
            elif not llm_key:
                need = (
                    "**Anthropic API key** (or `ANTHROPIC_API_KEY`)"
                    if llm_provider == "anthropic"
                    else "**OpenAI API key** (or `OPENAI_API_KEY`)"
                )
                st.warning(f"Add {need} for the selected provider.")
            else:
                try:
                    with st.spinner("Loading context…"):
                        ctx = build_llm_context(client)
                    with st.spinner("Planning…"):
                        tool, arguments, raw_json = plan_mcp_call(
                            provider=llm_provider,
                            api_key=llm_key,
                            model=llm_model,
                            user_message=q,
                            mcp_context=ctx,
                        )
                    with st.expander("Plan (tool + args)", expanded=False):
                        st.code(
                            f"{tool}\n\n" + json.dumps(arguments, indent=2, ensure_ascii=False),
                            language="json",
                        )
                        st.code(raw_json, language="json")
                    with st.spinner("Calling MCP…"):
                        result: Any = client.call_tool(tool, arguments)
                    render_mcp_result(tool, result)
                except Exception as e:
                    st.error(str(e))

    with tab_charts:
        st.caption("Direct **links_aggregations** — bar chart + table in expander.")
        c1, c2, c3 = st.columns(3)
        with c1:
            facet = st.selectbox(
                "Facet",
                ("content_type", "tags", "entities", "breadcrumbs", "data_type"),
                key="agg_facet",
            )
        with c2:
            agg_size = st.number_input("Bucket limit", min_value=10, max_value=500, value=50, key="agg_size")
        with c3:
            st.text_input("from (optional)", key="agg_from", placeholder="e.g. 1 month ago")
        st.text_input("to (optional)", key="agg_to", placeholder="e.g. now")
        if st.button("Run aggregation", type="primary", key="agg_run"):
            args: dict[str, Any] = {"facet": facet, "agg_size": int(agg_size)}
            af = str(st.session_state.get("agg_from", "")).strip()
            at = str(st.session_state.get("agg_to", "")).strip()
            if af:
                args["from_"] = af
            if at:
                args["to_"] = at
            try:
                with st.spinner("Fetching…"):
                    agg_result = client.call_tool("links_aggregations", args)
                render_aggregations(agg_result)
            except Exception as e:
                st.error(str(e))

    with tab_fields:
        st.caption("**list_fields** for `dynamic_search` paths.")
        ix = st.radio("Index", ("links", "documents"), horizontal=True, key="lf_index")
        if st.button("Load fields", type="primary", key="lf_run"):
            try:
                with st.spinner("Loading…"):
                    fields = client.call_tool("list_fields", {"index": ix})
                st.session_state["_lf_cache"] = fields
                st.session_state["_lf_for"] = ix
            except Exception as e:
                st.error(str(e))
        cached = st.session_state.get("_lf_cache")
        cached_ix = st.session_state.get("_lf_for")
        if cached is not None:
            if cached_ix != ix:
                st.warning(f"Showing **{cached_ix}** — click **Load fields** for **{ix}**.")
            qf = st.text_input("Filter", key="lf_filter", placeholder="path fragment…")
            if isinstance(cached, list) and cached:
                df = pd.DataFrame([r for r in cached if isinstance(r, dict)])
                if qf.strip():
                    m = df.astype(str).apply(lambda r: r.str.contains(qf.strip(), case=False, na=False).any(), axis=1)
                    df = df[m]
                st.dataframe(dataframe_for_st(df), width="stretch", hide_index=True, height=480)
                with st.expander("Raw JSON"):
                    _lf_raw = json.dumps(cached, indent=2, ensure_ascii=False)
                    _lf_k = str(cached_ix or ix or "fields")
                    st.download_button(
                        "Download full JSON",
                        data=_lf_raw.encode("utf-8"),
                        file_name="list_fields.json",
                        mime="application/json",
                        key=f"lf_tab_raw_dl_{_lf_k}",
                    )
                    st.code(_lf_raw, language="json")
            else:
                render_list_fields_clean(cached)

    with tab_adv:
        st.caption("Raw MCP tool + JSON arguments.")
        mtool = st.selectbox("Tool", MANUAL_TOOLS, index=MANUAL_TOOLS.index("search_links"), key="man_tool")
        if st.session_state.get("_man_prev") != mtool:
            st.session_state.man_args = json.dumps(MANUAL_DEFAULTS.get(mtool, {}), indent=2)
            st.session_state._man_prev = mtool
        if "man_args" not in st.session_state:
            st.session_state.man_args = json.dumps(MANUAL_DEFAULTS["search_links"], indent=2)
        st.text_area("Arguments", key="man_args", height=200)
        if st.button("Run tool", type="primary", key="man_go"):
            try:
                args = json.loads(st.session_state.man_args)
            except json.JSONDecodeError as e:
                st.error(str(e))
            else:
                if not isinstance(args, dict):
                    st.error("Arguments must be a JSON object.")
                else:
                    try:
                        with st.spinner("Calling…"):
                            res = client.call_tool(mtool, args)
                        render_mcp_result(mtool, res)
                    except Exception as e:
                        st.error(str(e))

    client.close()


if __name__ == "__main__":
    main()
