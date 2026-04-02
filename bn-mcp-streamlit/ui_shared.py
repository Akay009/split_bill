"""Shared layout: theme, sidebar credentials, OpenAI + Anthropic (Claude) planner options."""

from __future__ import annotations

import os

import streamlit as st

from bn_client import BNClient

OPENAI_MODEL_CHOICES: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-mini",
    "o1",
    "gpt-4.1-mini",
    "gpt-4.1",
)

ANTHROPIC_MODEL_CHOICES: tuple[str, ...] = (
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
)

CSS = """
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Fraunces', Georgia, serif !important;
        letter-spacing: -0.02em;
    }
    .stApp {
        background: linear-gradient(165deg, #0f1419 0%, #1a2332 45%, #0d1117 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151b24 0%, #0f1419 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Fraunces', Georgia, serif;
    }
    .bn-hero {
        padding: 1.25rem 0 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.25rem;
    }
    .bn-hero p {
        color: rgba(255,255,255,0.65);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        max-width: 52rem;
    }
    .bn-pill {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5eead4;
        background: rgba(94, 234, 212, 0.12);
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stMetricContainer"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
    }
    /* Document result cards */
    .bn-doc-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        margin-bottom: 14px;
        overflow: hidden;
    }
    .bn-doc-card-grid {
        display: flex;
        flex-wrap: wrap;
        align-items: stretch;
    }
    .bn-doc-main {
        flex: 1 1 260px;
        padding: 14px 18px;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    .bn-doc-actions {
        flex: 0 0 auto;
        min-width: 108px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 14px 18px;
        background: rgba(0,0,0,0.2);
    }
    .bn-doc-kicker {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: rgba(94, 234, 212, 0.88);
        margin-bottom: 6px;
        font-weight: 600;
    }
    .bn-doc-heading {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.12rem;
        font-weight: 600;
        margin: 0 0 8px 0;
        line-height: 1.35;
        color: rgba(255,255,255,0.96);
    }
    .bn-doc-chips {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.52);
        margin-bottom: 10px;
        line-height: 1.4;
    }
    .bn-doc-snippet {
        font-size: 0.88rem;
        line-height: 1.55;
        color: rgba(255,255,255,0.7);
        margin: 0;
    }
    .bn-doc-missing {
        color: rgba(255,255,255,0.42);
        font-style: italic;
    }
    .bn-doc-missing code { color: rgba(94, 234, 212, 0.7); font-style: normal; }
    .bn-doc-host {
        margin-top: 10px;
        font-size: 0.76rem;
        color: rgba(255,255,255,0.38);
    }
    .bn-doc-btn {
        display: inline-block;
        padding: 10px 18px;
        background: linear-gradient(160deg, rgba(94,234,212,0.22), rgba(45,212,191,0.08));
        border: 1px solid rgba(94,234,212,0.42);
        border-radius: 8px;
        color: #7ff5e1 !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.86rem;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .bn-doc-btn:hover {
        background: rgba(94,234,212,0.18);
        border-color: rgba(94,234,212,0.65);
    }
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def env_defaults() -> tuple[str, str, str, str, str, str]:
    bn_tok = os.environ.get("BN_API_TOKEN", "").strip()
    mcp_url = os.environ.get("BN_MCP_URL", "https://beyondnews-mcp.fastmcp.app/mcp").strip()
    oa_key = os.environ.get("OPENAI_API_KEY", "").strip()
    oa_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    an_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    an_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
    try:
        if hasattr(st, "secrets"):
            bn_tok = (st.secrets.get("BN_API_TOKEN") or bn_tok or "").strip()
            mcp_url = (st.secrets.get("BN_MCP_URL") or mcp_url or "").strip()
            oa_key = (st.secrets.get("OPENAI_API_KEY") or oa_key or "").strip()
            oa_model = (st.secrets.get("OPENAI_MODEL") or oa_model or "gpt-4o-mini").strip()
            an_key = (st.secrets.get("ANTHROPIC_API_KEY") or an_key or "").strip()
            an_model = (st.secrets.get("ANTHROPIC_MODEL") or an_model or "claude-sonnet-4-20250514").strip()
    except (FileNotFoundError, KeyError, RuntimeError):
        pass
    return bn_tok, mcp_url.rstrip("/"), oa_key, oa_model, an_key, an_model


def init_session_defaults() -> None:
    bn_tok_e, mcp_url_e, oa_key_e, oa_model_e, an_key_e, an_model_e = env_defaults()
    if "in_mcp_url" not in st.session_state:
        st.session_state.in_mcp_url = mcp_url_e
    if "in_token" not in st.session_state:
        st.session_state.in_token = bn_tok_e
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = oa_key_e
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = oa_model_e
    if "openai_model_choice" not in st.session_state:
        st.session_state.openai_model_choice = (
            oa_model_e if oa_model_e in OPENAI_MODEL_CHOICES else "Custom"
        )
    if "openai_model_custom" not in st.session_state:
        st.session_state.openai_model_custom = (
            oa_model_e if oa_model_e not in OPENAI_MODEL_CHOICES else "gpt-4o-mini"
        )
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = an_key_e
    if "anthropic_model_choice" not in st.session_state:
        st.session_state.anthropic_model_choice = (
            an_model_e if an_model_e in ANTHROPIC_MODEL_CHOICES else "Custom"
        )
    if "anthropic_model_custom" not in st.session_state:
        st.session_state.anthropic_model_custom = (
            an_model_e if an_model_e not in ANTHROPIC_MODEL_CHOICES else "claude-sonnet-4-20250514"
        )
    if "planner_provider" not in st.session_state:
        # Prefer env; else migrate legacy label key if present
        env_p = os.environ.get("PLANNER_LLM", "").strip().lower()
        if env_p in ("anthropic", "claude"):
            st.session_state.planner_provider = "anthropic"
        else:
            legacy = str(st.session_state.get("planner_llm", ""))
            if legacy.startswith("Anthropic"):
                st.session_state.planner_provider = "anthropic"
            else:
                st.session_state.planner_provider = "openai"


def resolved_openai_model() -> str:
    choice = str(st.session_state.get("openai_model_choice", "gpt-4o-mini"))
    if choice == "Custom":
        return str(st.session_state.get("openai_model_custom", "") or "gpt-4o-mini").strip()
    return choice


def resolved_anthropic_model() -> str:
    choice = str(st.session_state.get("anthropic_model_choice", "claude-sonnet-4-20250514"))
    if choice == "Custom":
        return str(
            st.session_state.get("anthropic_model_custom", "") or "claude-sonnet-4-20250514"
        ).strip()
    return choice


def planner_provider() -> str:
    """Stable internal value: 'openai' | 'anthropic' (set by sidebar radio)."""
    p = str(st.session_state.get("planner_provider", "openai")).strip().lower()
    return p if p in ("openai", "anthropic") else "openai"


def sidebar_connection(*, show_llm: bool = True) -> tuple[BNClient | None, str, str, str]:
    """
    Renders sidebar. Returns (client, provider, api_key, model).
    provider is \"openai\" or \"anthropic\". Client is None if MCP URL or BN token missing.
    """
    init_session_defaults()
    with st.sidebar:
        st.markdown("### Connection")
        st.text_input("MCP URL", key="in_mcp_url", help="Streamable HTTP endpoint for the BN MCP server.")
        st.text_input("BN API key", type="password", key="in_token", help="Sent as x-api-key to MCP.")

        if show_llm:
            st.markdown("### Search planner (LLM)")
            st.radio(
                "Provider",
                options=["openai", "anthropic"],
                format_func=lambda x: "OpenAI (GPT)" if x == "openai" else "Anthropic (Claude)",
                key="planner_provider",
                horizontal=True,
                help="Stores openai vs anthropic internally so the correct API is always used.",
            )
            st.text_input(
                "OpenAI API key",
                type="password",
                key="openai_key",
                help="Used when Provider is OpenAI. Env: OPENAI_API_KEY",
            )
            opts_o = list(OPENAI_MODEL_CHOICES) + ["Custom"]
            st.selectbox("OpenAI model", opts_o, key="openai_model_choice")
            if st.session_state.get("openai_model_choice") == "Custom":
                st.text_input("Custom OpenAI model id", key="openai_model_custom", placeholder="e.g. gpt-4o")

            st.divider()
            st.text_input(
                "Anthropic API key",
                type="password",
                key="anthropic_key",
                help="Used when Provider is Anthropic. Env: ANTHROPIC_API_KEY",
            )
            opts_a = list(ANTHROPIC_MODEL_CHOICES) + ["Custom"]
            st.selectbox("Claude model", opts_a, key="anthropic_model_choice")
            if st.session_state.get("anthropic_model_choice") == "Custom":
                st.text_input(
                    "Custom Claude model id",
                    key="anthropic_model_custom",
                    placeholder="e.g. claude-sonnet-4-20250514",
                )

        st.divider()
        st.caption(
            "Env / secrets: BN_MCP_URL, BN_API_TOKEN, OPENAI_* or ANTHROPIC_API_KEY, ANTHROPIC_MODEL, "
            "optional PLANNER_LLM=anthropic."
        )

    mcp_url = str(st.session_state.get("in_mcp_url", "")).strip()
    bn_tok = str(st.session_state.get("in_token", "")).strip()
    prov = planner_provider()
    if prov == "anthropic":
        pkey = str(st.session_state.get("anthropic_key", "")).strip()
        pmodel = resolved_anthropic_model()
    else:
        pkey = str(st.session_state.get("openai_key", "")).strip()
        pmodel = resolved_openai_model()

    if not mcp_url or not bn_tok:
        return None, prov, pkey, pmodel
    return BNClient(mcp_url, bn_tok), prov, pkey, pmodel


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="bn-hero"><span class="bn-pill">Beyond News</span><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
