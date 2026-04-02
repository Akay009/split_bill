"""Clean MCP result views: title, link, content; charts; optional raw JSON."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

_DF_KW = {"width": "stretch", "hide_index": True}
_LINK_SUMMARY_MAX = 520
_DOC_SUMMARY_MAX = 600


def _cell_display_str(v: Any) -> str:
    """Normalize mixed API values so PyArrow can build arrays (list_fields `example`, etc.)."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def dataframe_for_st(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce object-typed columns to strings before st.dataframe.
    Avoids pyarrow.lib.ArrowTypeError when a column mixes str, bool, bytes, etc.
    """
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        out[c] = out[c].map(_cell_display_str)
    return out


def _raw_expander(
    payload: Any,
    *,
    title: str = "Raw response (JSON)",
    download_key: str = "bn_raw",
) -> None:
    """Full JSON (no truncation). Large payloads: use download if the browser feels slow."""
    with st.expander(title, expanded=False):
        if isinstance(payload, (dict, list)):
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            text = str(payload)
        st.download_button(
            "Download full JSON",
            data=text.encode("utf-8"),
            file_name="bn_response.json",
            mime="application/json",
            key=f"{download_key}_json_dl",
        )
        st.code(text, language="json")


def _meta_pair(it: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    m0 = it.get("meta_data_without_llm")
    m1 = it.get("meta_data_with_llm")
    return (
        m0 if isinstance(m0, dict) else {},
        m1 if isinstance(m1, dict) else {},
    )


def _first_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _breadcrumbs_line(m0: dict[str, Any], m1: dict[str, Any]) -> str:
    for src in (m1, m0, {}):
        bc = src.get("breadcrumbs")
        if isinstance(bc, list) and bc:
            parts = []
            for x in bc[-4:]:
                if isinstance(x, str) and x.strip():
                    parts.append(x.strip())
                elif isinstance(x, dict):
                    t = x.get("title") or x.get("name") or x.get("label")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            if parts:
                return " › ".join(parts)
    return ""


def _item_headline_link(it: dict[str, Any], *, idx: int) -> tuple[str, str]:
    """(headline, kind) kind = title|breadcrumb|summary|fallback"""
    m0, m1 = _meta_pair(it)
    t = _first_str(it.get("title"), m0.get("title"), m1.get("title"))
    if t:
        return t, "title"
    bc = _breadcrumbs_line(m0, m1)
    if bc:
        return bc, "breadcrumb"
    s = _first_str(m1.get("summary"))
    if s:
        one = s.split("\n", 1)[0].strip()
        return (one[:140] + "…") if len(one) > 140 else one, "summary"
    return f"Result {idx}", "fallback"


def _item_meta_line(it: dict[str, Any], m0: dict[str, Any], m1: dict[str, Any]) -> str:
    parts: list[str] = []
    dom = _first_str(it.get("domain_url"), m0.get("domain_url"), m1.get("domain_url"))
    if dom:
        parts.append(dom.replace("www.", ""))
    d = _first_str(m0.get("parse_date"), m1.get("parse_date"), it.get("parse_date"))
    if d:
        parts.append(d[:10])
    ct = _first_str(m1.get("content_type"), m0.get("content_type"))
    if ct:
        parts.append(ct)
    return " · ".join(parts)


def _item_best_url(it: dict[str, Any], m0: dict[str, Any], m1: dict[str, Any]) -> str:
    for v in (
        it.get("url"),
        it.get("page_url"),
        it.get("documents_link"),
        it.get("link"),
        m0.get("url"),
        m0.get("documents_link"),
        m1.get("documents_link"),
    ):
        if isinstance(v, str) and v.startswith("http"):
            return v
    return ""


def _item_teaser(it: dict[str, Any], m0: dict[str, Any], m1: dict[str, Any]) -> str:
    s = _first_str(m1.get("summary"))
    if s:
        return s
    c = _first_str(m0.get("content"))
    if c:
        return c.replace("\n", " ").strip()
    return ""


def _truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _title_from_url(url: str) -> str:
    p = urlparse(url)
    path = unquote(p.path or "")
    seg = path.rstrip("/").split("/")[-1] if path else ""
    if not seg and p.netloc:
        return p.netloc.replace("www.", "")
    if not seg:
        return "Document"
    base = seg.rsplit(".", 1)[0] if "." in seg else seg
    t = base.replace("_", " ").replace("-", " ").strip()
    return t or "Document"


def _doc_extract_url(it: dict[str, Any]) -> str:
    keys = (
        "url",
        "document_url",
        "pdf_url",
        "file_url",
        "documents_link",
        "download_url",
        "href",
    )
    for k in keys:
        v = it.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    for nest in ("file", "asset", "document", "attachment", "meta", "source"):
        n = it.get(nest)
        if isinstance(n, dict):
            for k in keys:
                v = n.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    return v
    return ""


def _doc_primary_title(it: dict[str, Any], idx: int, url: str) -> tuple[str, str]:
    """(title, kicker_label) — kicker explains where the title came from."""
    for k in ("title", "name", "file_name", "filename", "label", "headline"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return _truncate(v.strip(), 220), "From index"
    s = it.get("summary")
    if isinstance(s, str) and s.strip():
        line = s.split("\n", 1)[0].strip()
        return _truncate(line, 220), "From summary"
    for nest in ("document", "meta", "source", "attachment"):
        n = it.get(nest)
        if isinstance(n, dict):
            for k in ("title", "name", "file_name", "summary"):
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    return _truncate(v.strip(), 220), "From nested record"
    if url:
        return _truncate(_title_from_url(url), 220), "From file name / URL"
    dt = it.get("doc_type")
    if isinstance(dt, str) and dt.strip():
        lid = it.get("link_id")
        if lid is not None:
            return f"{dt.strip()} · ref #{lid}", "Type + reference"
        return _truncate(dt.strip(), 220), "Document type"
    lid = it.get("link_id")
    if lid is not None:
        return f"Attachment · link #{lid}", "Reference only"
    return f"Row {idx}", "Sparse data"


def _doc_snippet_text(it: dict[str, Any]) -> str:
    s = it.get("summary")
    if isinstance(s, str) and s.strip():
        body = s.strip()
        first = body.split("\n", 1)[0]
        if len(body) > len(first) + 2:
            return _truncate(body, 360)
        return _truncate(body, 360)
    for nest in ("document", "meta"):
        n = it.get(nest)
        if isinstance(n, dict):
            for k in ("summary", "description", "abstract"):
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    return _truncate(v.strip(), 360)
    return ""


def _link_table_rows(items: list[Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    page = int(result.get("page") or 1)
    ps = int(result.get("page_size") or len(items) or 10)
    base = (page - 1) * ps
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        m0, m1 = _meta_pair(it)
        t_direct = _first_str(it.get("title"), m0.get("title"), m1.get("title"))
        if t_direct:
            title_cell = _truncate(t_direct, 220)
        else:
            headline, _kind = _item_headline_link(it, idx=i + 1)
            title_cell = _truncate(headline, 220)

        ct = _first_str(
            it.get("content_type"),
            m1.get("content_type"),
            m0.get("content_type"),
        )
        summ = _first_str(m1.get("summary"), it.get("summary"))
        cont = _first_str(m0.get("content"), it.get("content"))
        if summ and cont and summ.strip() == cont.strip():
            body = summ
        elif summ and cont:
            body = f"{summ}\n\n{cont}"
        else:
            body = summ or cont or _item_teaser(it, m0, m1)
        body_disp = _truncate(body.replace("\n", " ").strip(), _LINK_SUMMARY_MAX) if body else "—"

        url = _item_best_url(it, m0, m1)
        link_cell = url if isinstance(url, str) and url.startswith("http") else None

        rows.append(
            {
                "#": base + i + 1,
                "Title": title_cell,
                "Content type": ct or "—",
                "Summary": body_disp,
                "URL": link_cell,
            }
        )
    return rows


def _document_table_rows(items: list[Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    page = int(result.get("page") or 1)
    sz = int(result.get("size") or result.get("page_size") or len(items) or 10)
    base = (page - 1) * sz
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        url = _doc_extract_url(it)
        title, _k = _doc_primary_title(it, i + 1, url)
        snippet = _doc_snippet_text(it)
        dt = it.get("doc_type")
        dt_s = dt.strip() if isinstance(dt, str) else "—"
        lid = it.get("link_id")
        lid_s = str(lid) if lid is not None else "—"
        tags = it.get("tags")
        if isinstance(tags, list):
            tags_s = ", ".join(str(t).strip() for t in tags[:16] if t is not None and str(t).strip())
        else:
            tags_s = ""
        rows.append(
            {
                "#": base + i + 1,
                "Title": _truncate(title, 220),
                "Doc type": dt_s,
                "Summary": _truncate(snippet, _DOC_SUMMARY_MAX) if snippet else "—",
                "URL": url if isinstance(url, str) and url.startswith("http") else None,
                "Link ID": lid_s,
                "Tags": tags_s or "—",
            }
        )
    return rows


def _show_link_results_table(df: pd.DataFrame, *, height_px: int) -> None:
    col_map: dict[str, Any] = {
        "#": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Title": st.column_config.TextColumn("Title", width="large"),
        "Content type": st.column_config.TextColumn("Content type", width="small"),
        "Summary": st.column_config.TextColumn("Summary / content", width="large"),
        "URL": st.column_config.LinkColumn(
            "URL",
            width="medium",
            display_text="Open ↗",
            help="Page or document hub link from the index",
        ),
    }
    cfg = {c: col_map[c] for c in df.columns if c in col_map}
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=min(height_px, 720),
        column_config=cfg,
    )


def _show_document_results_table(df: pd.DataFrame, *, height_px: int) -> None:
    col_map: dict[str, Any] = {
        "#": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Title": st.column_config.TextColumn("Title", width="large"),
        "Doc type": st.column_config.TextColumn("Doc type", width="small"),
        "Summary": st.column_config.TextColumn("Summary", width="large"),
        "URL": st.column_config.LinkColumn(
            "URL",
            width="medium",
            display_text="Open ↗",
        ),
        "Link ID": st.column_config.TextColumn("Link ID", width="small"),
        "Tags": st.column_config.TextColumn("Tags", width="medium"),
    }
    cfg = {c: col_map[c] for c in df.columns if c in col_map}
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=min(height_px, 720),
        column_config=cfg,
    )


def render_links_search_clean(result: dict[str, Any], *, show_raw: bool = True) -> None:
    items = result.get("items")
    if not isinstance(items, list) or not items:
        th = result.get("total_hits")
        if th == 0:
            st.info("No matches for this query. Try broader keywords or the **Advanced** tab.")
        else:
            st.warning("No rows on this page.")
        if show_raw:
            _raw_expander(result, download_key="links_search_empty")
        return

    th = result.get("total_hits")
    pg = result.get("page") or 1
    tpg = result.get("total_pages")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total matches", f"{th:,}" if isinstance(th, int) else "—")
    with m2:
        st.metric("Rows here", str(len(items)))
    with m3:
        st.metric("Page", f"{pg} / {tpg}" if tpg is not None else str(pg))

    st.subheader("Results")
    st.caption(
        "Columns match the BN links index: **title**, **content type**, **summary / content**, **URL** "
        "(from `documents_link` or page URL). Expand **Raw response** for full JSON."
    )

    rows = _link_table_rows(items, result)
    df = pd.DataFrame(rows)
    row_h = 48
    _show_link_results_table(df, height_px=96 + row_h * max(len(rows), 3))

    if show_raw:
        _raw_expander(result, download_key="links_search")


def render_documents_clean(result: dict[str, Any], *, show_raw: bool = True) -> None:
    items = result.get("items") or result.get("documents")
    if not isinstance(items, list) or not items:
        st.warning("No documents in this response.")
        if show_raw:
            _raw_expander(result, download_key="documents_search_empty")
        return

    th = result.get("total_hits", result.get("total"))
    page = int(result.get("page") or 1)
    tpg = result.get("total_pages")
    sz = int(result.get("size") or result.get("page_size") or len(items) or 10)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total in index", f"{th:,}" if isinstance(th, int) else "—")
    with c2:
        st.metric("Listed below", str(len(items)))
    with c3:
        st.metric("Page", f"{page} / {tpg}" if tpg is not None else str(page))
    with c4:
        st.metric("Page size", f"{sz}")

    if isinstance(th, int) and th > len(items):
        st.info(
            f"The API reports **{th:,}** matching documents, but this response only includes **{len(items)}** "
            f"rows. Use the **Advanced** tab to raise **page_size** or change **page**, or ask in Search for “page 2…”."
        )

    st.subheader("Documents")
    st.caption(
        "Tabular view for the **documents** index: title, **doc type**, summary, file/page **URL**, link id, tags. "
        "Shape follows the API (see **Raw response** if a column is sparse)."
    )

    drows = _document_table_rows(items, result)
    if not drows:
        st.warning("No parseable document rows.")
    else:
        df = pd.DataFrame(drows)
        row_h = 48
        _show_document_results_table(df, height_px=96 + row_h * max(len(drows), 3))

    if show_raw:
        _raw_expander(result, download_key="documents_search")


def render_aggregations(result: dict[str, Any], *, show_raw: bool = True) -> None:
    if not isinstance(result, dict):
        _raw_expander(result, download_key="aggregations_nondict")
        return

    total = result.get("total")
    if total is not None:
        st.caption(f"**{total:,}** records in scope")

    found = False
    for key, block in result.items():
        if key == "total" or not isinstance(block, dict):
            continue
        opts = block.get("options")
        if not isinstance(opts, list) or not opts:
            continue
        found = True
        rows = []
        for o in opts:
            if not isinstance(o, dict):
                continue
            val = o.get("value")
            cnt = o.get("count", 0)
            rows.append(
                {
                    "value": str(val) if val is not None else "",
                    "count": int(cnt) if isinstance(cnt, (int, float)) else 0,
                }
            )
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("count", ascending=False)
        st.subheader(str(key).replace("_", " ").title())
        fig = px.bar(
            df.head(40),
            x="count",
            y="value",
            orientation="h",
            labels={"count": "Count", "value": key},
            color="count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            showlegend=False,
            height=min(480, 72 + 20 * len(df.head(40))),
            margin=dict(l=8, r=8, t=24, b=8),
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        with st.expander(f"Counts table · {key}"):
            st.dataframe(df, **_DF_KW)

    if not found:
        st.info("No buckets in this response.")

    if show_raw:
        _raw_expander(result, download_key="aggregations")


def render_health_clean(result: Any, *, show_raw: bool = True) -> None:
    if isinstance(result, dict):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("API", str(result.get("status", "—")))
        with c2:
            st.metric("OpenSearch", str(result.get("opensearch", "—")))
    if show_raw:
        _raw_expander(result, download_key="health")


def render_explore_clean(result: Any, *, show_raw: bool = True) -> None:
    st.caption("API reference payload — see raw JSON for full detail.")
    if show_raw:
        _raw_expander(result, title="Raw explore (JSON)", download_key="explore")


def render_list_fields_clean(result: Any, *, show_raw: bool = True) -> None:
    if isinstance(result, list):
        rows = [r for r in result if isinstance(r, dict)]
        if rows:
            st.dataframe(dataframe_for_st(pd.DataFrame(rows)), **_DF_KW)
            if show_raw:
                _raw_expander(result, download_key="list_fields_list")
            return
    if isinstance(result, dict) and "fields" in result:
        st.dataframe(dataframe_for_st(pd.DataFrame(result["fields"])), **_DF_KW)
        if show_raw:
            _raw_expander(result, download_key="list_fields_dict")
        return
    st.caption("Unexpected shape — open raw JSON.")
    _raw_expander(result, download_key="list_fields_other")


def render_mcp_result(tool: str, result: Any) -> None:
    """Dispatch to clean layout; raw JSON always available in expander."""
    if isinstance(result, dict) and tool in ("search_links", "search_links_post"):
        render_links_search_clean(result)
    elif isinstance(result, dict) and tool in ("search_documents", "search_documents_post"):
        render_documents_clean(result)
    elif isinstance(result, dict) and tool == "links_aggregations":
        render_aggregations(result)
    elif tool == "check_health":
        render_health_clean(result)
    elif tool == "explore":
        render_explore_clean(result)
    elif tool == "list_fields":
        render_list_fields_clean(result)
    else:
        st.caption("Structured response")
        _raw_expander(result, download_key=f"mcp_tool_{tool}")
