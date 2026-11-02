"""Small shared styling helpers. Colours match the architecture diagram: blue = vector, apricot = graph, mint = guardrails."""

from __future__ import annotations

import html

import streamlit as st

# Light-only, pale formal theme. Sizes are Streamlit's own; this only changes colour, weight, glow and motion.
_CSS = """
<style>
  :root { --rag-ink: #1B2540; --rag-muted: #66708A; --rag-line: #DCE4F2; --rag-card: #FFFFFF;
          --rag-blue: #3D63E0; --rag-blue-2: #7C9BFF; --rag-blue-pale: #EDF2FF; --rag-blue-line: #C7D5FF;
          --rag-apricot: #D9822B; --rag-apricot-pale: #FFF4E8; --rag-apricot-line: #F6D2AB;
          --rag-mint: #1E9E6A; --rag-mint-pale: #E8F8F0; --rag-mint-line: #B4E6CE; --rag-lilac: #8B6CF0;
          --rag-glow: 0 0 0 1px rgba(61,99,224,.16), 0 0 18px rgba(61,99,224,.16), 0 10px 30px rgba(27,37,64,.05);
          --rag-glow-hi: 0 0 0 1px rgba(61,99,224,.45), 0 0 26px rgba(61,99,224,.32), 0 14px 34px rgba(27,37,64,.08); }

  /* pale canvas with soft light from the top corners */
  [data-testid="stAppViewContainer"] {
    background: radial-gradient(900px 380px at 12% -8%, rgba(124,155,255,.20), transparent 70%),
                radial-gradient(700px 320px at 100% -6%, rgba(217,130,43,.10), transparent 70%), #F5F7FB;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #EEF3FB 0%, #F8FAFE 100%); }
  [data-testid="stSidebar"] h3 { font-weight: 800; }
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  label p, [data-testid="stWidgetLabel"] p { font-weight: 700; }

  /* title: bold, with a band of light passing across */
  [data-testid="stMainBlockContainer"] h1 {
    background: linear-gradient(100deg, #1B2540 0%, #1B2540 38%, #3D63E0 48%, #7C9BFF 52%, #1B2540 62%, #1B2540 100%);
    background-size: 260% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: rag-shine 6s ease-in-out infinite; letter-spacing: -.03em;
  }

  /* metric cards: bright glow border, bold counted-up values */
  [data-testid="stMetric"] {
    background: var(--rag-card); border: 1px solid var(--rag-blue-line); border-radius: 14px; padding: 12px 14px;
    box-shadow: var(--rag-glow); position: relative; overflow: hidden;
    animation: rag-rise .6s cubic-bezier(.2,.8,.2,1) both; transition: transform .25s, box-shadow .25s;
  }
  [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: var(--rag-glow-hi); }
  [data-testid="stMetric"]::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.9) 50%, transparent 70%);
    transform: translateX(-110%); animation: rag-sweep 1.5s .35s ease-out both;
  }
  [data-testid="stMetricLabel"] p { font-weight: 700; color: var(--rag-muted); }
  [data-testid="stMetricValue"] { font-family: Outfit, Manrope, sans-serif; font-weight: 800; color: var(--rag-blue);
                                  text-shadow: 0 0 16px rgba(61,99,224,.25); font-variant-numeric: tabular-nums; }
  [data-testid="stSidebar"] [data-testid="stMetric"] { padding: 8px 10px; }

  /* tabs as a glowing pill switch */
  [data-baseweb="tab-list"] { background: var(--rag-card); border: 1px solid var(--rag-line); border-radius: 14px; padding: 4px;
                              gap: 4px; width: fit-content; box-shadow: var(--rag-glow); }
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
  [data-testid="stTab"] { border-radius: 10px; padding: 4px 16px; height: auto; transition: background .2s, box-shadow .2s; }
  [data-testid="stTab"] p { font-weight: 700; color: var(--rag-muted); }
  [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, #3D63E0, #7C9BFF); box-shadow: 0 6px 18px rgba(61,99,224,.35); }
  [data-testid="stTab"][aria-selected="true"] p { color: #FFFFFF; font-weight: 800; }

  /* buttons */
  [data-testid^="stBaseButton-primary"] {
    background: linear-gradient(135deg, #3D63E0, #7C9BFF) !important; border: 0 !important; color: #fff !important; font-weight: 800;
    box-shadow: 0 0 0 4px rgba(61,99,224,.12), 0 10px 26px rgba(61,99,224,.38); position: relative; overflow: hidden; transition: transform .15s;
  }
  [data-testid^="stBaseButton-primary"] p { font-weight: 800; }
  [data-testid^="stBaseButton-primary"]::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(110deg, transparent 35%, rgba(255,255,255,.55) 50%, transparent 65%);
    transform: translateX(-120%); animation: rag-btn-sweep 3.2s ease-in-out infinite;
  }
  [data-testid^="stBaseButton-primary"]:hover { transform: translateY(-1px); }
  [data-testid^="stBaseButton-secondary"] { background: var(--rag-card); border-color: var(--rag-blue-line) !important;
                                            box-shadow: var(--rag-glow); transition: box-shadow .2s; }
  [data-testid^="stBaseButton-secondary"] p { font-weight: 700; color: var(--rag-blue); }
  [data-testid^="stBaseButton-secondary"]:hover { box-shadow: var(--rag-glow-hi); }

  /* cards: form, expanders, tables, charts */
  [data-testid="stForm"] { background: var(--rag-card); border: 1px solid var(--rag-blue-line) !important; box-shadow: var(--rag-glow);
                           transition: box-shadow .25s; }
  [data-testid="stForm"]:hover { box-shadow: var(--rag-glow-hi); }
  [data-baseweb="textarea"], [data-baseweb="input"] { transition: box-shadow .2s; }
  [data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
    box-shadow: 0 0 0 4px rgba(61,99,224,.10), 0 0 16px rgba(61,99,224,.18); }
  [data-testid="stExpander"] details { background: var(--rag-card); border-color: var(--rag-line); transition: box-shadow .2s, border-color .2s; }
  [data-testid="stExpander"] details:hover { border-color: var(--rag-blue-line); box-shadow: 0 0 18px rgba(61,99,224,.2); }
  [data-testid="stExpander"] summary p { font-weight: 700; }
  [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stGraphVizChart"], [data-testid="stVegaLiteChart"] {
    animation: rag-rise .55s cubic-bezier(.2,.8,.2,1) both;
  }
  [data-testid="stDataFrame"] { border-radius: 12px; box-shadow: 0 0 0 1px rgba(217,130,43,.16), 0 0 18px rgba(217,130,43,.12); }

  /* answer card: flowing multicolour border with a soft halo */
  .st-key-answer_card {
    border: 2px solid transparent !important; border-radius: 18px;
    background: linear-gradient(#FFFFFF, #FFFFFF) padding-box,
                linear-gradient(120deg, #3D63E0, #8B6CF0, #D9822B, #1E9E6A, #3D63E0) border-box;
    background-size: 100% 100%, 300% 300%;
    box-shadow: 0 0 0 6px rgba(61,99,224,.07), 0 0 34px rgba(61,99,224,.22), 0 18px 44px rgba(27,37,64,.07);
    animation: rag-rise .6s cubic-bezier(.2,.8,.2,1) both, rag-border 7s linear infinite;
  }
  .st-key-answer_card strong { font-weight: 800; }

  /* knowledge graph drawing: nodes pop in, edges draw themselves, then keep flowing */
  [data-testid="stGraphVizChart"] { border: 1px solid var(--rag-apricot-line); border-radius: 14px; padding: 8px;
                                    background: radial-gradient(circle at 50% 50%, #FFF8EF, #FFFFFF 70%); }
  [data-testid="stGraphVizChart"] g.node { transform-box: fill-box; transform-origin: center;
                                           animation: rag-pop .7s cubic-bezier(.2,1.5,.4,1) both; }
  [data-testid="stGraphVizChart"] g.node:nth-of-type(2n) { animation-delay: .15s; }
  [data-testid="stGraphVizChart"] g.node:nth-of-type(3n) { animation-delay: .3s; }
  [data-testid="stGraphVizChart"] g.node path, [data-testid="stGraphVizChart"] g.node polygon {
    filter: drop-shadow(0 0 5px rgba(217,130,43,.45)); }
  [data-testid="stGraphVizChart"] g.edge path { stroke-dasharray: 5 6; animation: rag-dash 1.2s linear infinite; }

  /* chips: pale tints with a highlight gliding across */
  .chip { display: inline-block; position: relative; overflow: hidden; padding: 2px 11px; margin: 0 6px 6px 0; border-radius: 999px;
          border: 1px solid; font-size: 0.78rem; font-weight: 800; line-height: 1.5; white-space: nowrap;
          animation: rag-rise .5s cubic-bezier(.2,.8,.2,1) both; }
  .chip::after { content: ""; position: absolute; inset: 0; pointer-events: none;
                 background: linear-gradient(110deg, transparent 35%, rgba(255,255,255,.95) 50%, transparent 65%);
                 background-size: 250% 100%; animation: rag-glint 3.2s ease-in-out infinite; }
  .chip.vector { background: var(--rag-blue-pale); color: var(--rag-blue); border-color: var(--rag-blue-line); box-shadow: 0 0 14px rgba(61,99,224,.22); }
  .chip.graph  { background: var(--rag-apricot-pale); color: var(--rag-apricot); border-color: var(--rag-apricot-line); box-shadow: 0 0 14px rgba(217,130,43,.22); }
  .chip.guard  { background: var(--rag-mint-pale); color: var(--rag-mint); border-color: var(--rag-mint-line); box-shadow: 0 0 14px rgba(30,158,106,.22); }
  .chip.muted  { background: #F4F6FB; color: var(--rag-muted); border-color: var(--rag-line); }
  .chip.bad    { background: #FDECEE; color: #9B1C25; border-color: #F5B8BE; box-shadow: 0 0 14px rgba(198,40,51,.2); }
  .panel-head { display: flex; align-items: center; gap: 10px; font-family: Outfit, Manrope, sans-serif; font-weight: 800;
                font-size: 1.08rem; margin-bottom: 2px; }
  .panel-head .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; color: var(--c); background: var(--c);
                     animation: rag-pulse 2.2s ease-out infinite; }
  .panel-head.vector .dot { --c: #3D63E0; }
  .panel-head.graph .dot { --c: #D9822B; }
  .panel-head.guard .dot { --c: #1E9E6A; }

  @keyframes rag-shine { 0% { background-position: 100% 0; } 55%, 100% { background-position: 0 0; } }
  @keyframes rag-rise  { from { opacity: .25; transform: translateY(10px); } to { opacity: 1; transform: none; } }
  @keyframes rag-sweep { to { transform: translateX(110%); } }
  @keyframes rag-btn-sweep { 0%, 55% { transform: translateX(-120%); } 100% { transform: translateX(120%); } }
  @keyframes rag-glint { 0% { background-position: 160% 0; } 55%, 100% { background-position: -60% 0; } }
  @keyframes rag-border { to { background-position: 0 0, 300% 0; } }
  @keyframes rag-pulse { 0% { box-shadow: 0 0 0 0 currentColor; } 70%, 100% { box-shadow: 0 0 0 10px transparent; } }
  @keyframes rag-pop { from { transform: scale(.2); opacity: .2; } to { transform: scale(1); opacity: 1; } }
  @keyframes rag-dash { to { stroke-dashoffset: -22; } }

  @media (prefers-reduced-motion: reduce) {
    [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * { animation: none !important; transition: none !important; }
  }
</style>
"""

# Counts every st.metric value up from zero when it first appears or changes. It swaps Streamlit's text node for a
# temporary span and puts the original node back afterwards, so React keeps ownership of the real value.
_COUNT_UP_JS = """
<script>
(function () {
  if (window.__ragCountUp) return;
  window.__ragCountUp = true;
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const seen = new WeakMap();
  const fmt = (s, e) => {
    const dec = (s.split('.')[1] || '').length, v = parseFloat(s.replace(/,/g, '')) * e;
    return s.includes(',') ? v.toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec}) : v.toFixed(dec);
  };
  function animate(el) {
    if (el.__ragBusy) return;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let tn = null;
    while (walker.nextNode()) { if (/\\d/.test(walker.currentNode.nodeValue)) { tn = walker.currentNode; break; } }
    if (!tn || seen.get(el) === tn.nodeValue) return;
    const final = tn.nodeValue, parts = final.split(/(\\d[\\d,]*(?:\\.\\d+)?)/), parent = tn.parentNode;
    seen.set(el, final);
    const span = document.createElement('span');
    span.textContent = final;
    parent.replaceChild(span, tn);
    el.__ragBusy = true;
    const t0 = performance.now(), dur = 1100;
    (function step(now) {
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      if (!span.isConnected) { el.__ragBusy = false; return; }
      span.textContent = parts.map((s, i) => (i % 2 ? fmt(s, e) : s)).join('');
      if (p < 1) { requestAnimationFrame(step); return; }
      span.parentNode.replaceChild(tn, span);
      seen.set(el, tn.nodeValue);
      el.__ragBusy = false;
    })(t0);
  }
  const scan = () => document.querySelectorAll('[data-testid="stMetricValue"]').forEach(animate);
  new MutationObserver(() => { clearTimeout(window.__ragScan); window.__ragScan = setTimeout(scan, 80); })
    .observe(document.body, {childList: true, subtree: true, characterData: true});
  scan();
})();
</script>
"""


def inject() -> None:
    st.html(_CSS)
    st.html(_COUNT_UP_JS, unsafe_allow_javascript=True)


def chip(text: str, kind: str = "muted") -> str:
    return f'<span class="chip {kind}">{html.escape(str(text))}</span>'


def panel_head(text: str, kind: str) -> None:
    st.markdown(f'<div class="panel-head {kind}"><span class="dot"></span>{html.escape(text)}</div>', unsafe_allow_html=True)
