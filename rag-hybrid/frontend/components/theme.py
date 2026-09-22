"""Small shared styling helpers. Colours match the architecture diagram: teal = vector, purple = graph, coral = guardrails."""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  .chip { display: inline-block; padding: 1px 9px; margin: 0 6px 4px 0; border-radius: 999px; border: 1px solid;
          font-size: 0.78rem; line-height: 1.5; white-space: nowrap; }
  .chip.vector { background: #E1F5EE; color: #085041; border-color: #9FE1CB; }
  .chip.graph  { background: #EEEDFE; color: #3C3489; border-color: #CECBF6; }
  .chip.guard  { background: #FAECE7; color: #712B13; border-color: #F5C4B3; }
  .chip.muted  { background: #F1EFE8; color: #444441; border-color: #D3D1C7; }
  .chip.bad    { background: #FCEBEB; color: #791F1F; border-color: #F7C1C1; }
  .panel-head { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 1.05rem; margin-bottom: 2px; }
  .panel-head .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .panel-head.vector .dot { background: #0F6E56; }
  .panel-head.graph .dot { background: #534AB7; }
  .panel-head.guard .dot { background: #993C1D; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def chip(text: str, kind: str = "muted") -> str:
    return f'<span class="chip {kind}">{html.escape(str(text))}</span>'


def panel_head(text: str, kind: str) -> None:
    st.markdown(f'<div class="panel-head {kind}"><span class="dot"></span>{html.escape(text)}</div>', unsafe_allow_html=True)
