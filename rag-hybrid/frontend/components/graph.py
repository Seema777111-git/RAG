"""Graphviz DOT rendering of the retrieved knowledge-graph neighbourhood."""

from __future__ import annotations


def _esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def to_dot(nodes: list[dict], edges: list[dict]) -> str:
    """Seed entities (linked from the question) are filled amber; neighbours are light."""
    index = {n["id"]: f"n{i}" for i, n in enumerate(nodes)}
    lines = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.7;',
        '  node [shape=box, style="rounded,filled", fontname="Manrope, Helvetica", fontsize=11, color="#F0C190", fillcolor="#FFF4E8", fontcolor="#5C3800"];',
        '  edge [fontname="Manrope, Helvetica", fontsize=10, color="#D9822B", fontcolor="#A85F00", arrowsize=0.7];',
    ]
    for n in nodes:
        style = ', fillcolor="#D9822B", color="#B86A1C", fontcolor="#FFFFFF"' if n.get("is_seed") else ""
        lines.append(f'  {index[n["id"]]} [label="{_esc(n["label"])}"{style}];')
    for e in edges:
        if e["source"] in index and e["target"] in index:
            lines.append(f'  {index[e["source"]]} -> {index[e["target"]]} [label="{_esc(e["label"])}"];')
    lines.append("}")
    return "\n".join(lines)
