"""Graphviz DOT rendering of the retrieved knowledge-graph neighbourhood."""

from __future__ import annotations


def _esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def to_dot(nodes: list[dict], edges: list[dict]) -> str:
    """Seed entities (linked from the question) are filled purple; neighbours are light."""
    index = {n["id"]: f"n{i}" for i, n in enumerate(nodes)}
    lines = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.7;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#AFA9EC", fillcolor="#EEEDFE", fontcolor="#26215C"];',
        '  edge [fontname="Helvetica", fontsize=10, color="#7F77DD", fontcolor="#3C3489", arrowsize=0.7];',
    ]
    for n in nodes:
        style = ', fillcolor="#534AB7", color="#3C3489", fontcolor="#FFFFFF"' if n.get("is_seed") else ""
        lines.append(f'  {index[n["id"]]} [label="{_esc(n["label"])}"{style}];')
    for e in edges:
        if e["source"] in index and e["target"] in index:
            lines.append(f'  {index[e["source"]]} -> {index[e["target"]]} [label="{_esc(e["label"])}"];')
    lines.append("}")
    return "\n".join(lines)
