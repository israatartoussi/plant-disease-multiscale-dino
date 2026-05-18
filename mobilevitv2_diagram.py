#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a concise MobileViTv2 + Attention diagram (horizontal or vertical)
Outputs: PDF (vector) + optional PNG 600dpi
"""

import argparse
from graphviz import Digraph
from pathlib import Path

COL_BG   = "#FFFFFF"
COL_TXT  = "#111111"
COL_BRD  = "#333333"
COL_NODE = "#F7F7F7"
COL_BB   = "#E6F2FF"    # Backbone
COL_ATT  = "#FFF4CC"    # Attention
COL_HEAD = "#F4E6FF"    # Head

def make_diagram(attn_label="SAM", orientation="h", out_prefix="MobileViTv2_Attention_Effect",
                 fontsize=14, nodesep=0.45, ranksep=0.55, node_margin="0.10,0.08",
                 center=True, png=True, dpi=600):
    rankdir = "LR" if orientation.lower().startswith("h") else "TB"
    g = Digraph(out_prefix, format="pdf")
    g.attr(rankdir=rankdir, splines="line", center="true" if center else "false",
           bgcolor=COL_BG, fontsize=str(fontsize), fontname="Helvetica",
           label="", labelloc="t",
           pad="0.15" if rankdir=="LR" else "0.0",
           margin="0.0",
           nodesep=str(nodesep), ranksep=str(ranksep))

    node = dict(shape="box", style="rounded,filled",
                fillcolor=COL_NODE, color=COL_BRD,
                fontcolor=COL_TXT, fontname="Helvetica",
                penwidth="1.6", margin=node_margin)

    # small invisible junction points to keep arrows outside boxes
    def jpt(name):
        g.node(name, label="", shape="point", width="0.01", height="0.01", style="invis")

    # junctions between blocks
    jpt("J1"); jpt("J2"); jpt("J3")

    # nodes
    g.node("inp",  "Input\n(3, 224, 224)", **node)
    g.node("bb",   "MobileViTv2 Backbone", **{**node, "fillcolor": COL_BB})
    g.node("att",  f"{attn_label} — Attention Module\n(refines feature map)", **{**node, "fillcolor": COL_ATT})
    g.node("head", "Head\nGlobal Avg Pool → Linear", **{**node, "fillcolor": COL_HEAD})

    # place junctions using transparent edges so arrows stop at borders
    if rankdir == "LR":
        g.edge("J1", "bb",  color="transparent")
        g.edge("bb", "J2",  color="transparent")
        g.edge("J3", "head", color="transparent")

        g.edge("inp:e",  "J1",  headport="w", arrowsize="0.9")
        g.edge("bb:e",   "J2",  headport="w", arrowsize="0.9")
        g.edge("att:e",  "J3",  headport="w", arrowsize="0.9")
    else:
        g.edge("J1", "bb",  color="transparent")
        g.edge("bb", "J2",  color="transparent")
        g.edge("J3", "head", color="transparent")

        g.edge("inp:s",  "J1",  headport="n", arrowsize="0.9")
        g.edge("bb:s",   "J2",  headport="n", arrowsize="0.9")
        g.edge("att:s",  "J3",  headport="n", arrowsize="0.9")

    # visible flow between main blocks (for layout)
    if rankdir == "LR":
        g.edge("inp", "bb",  color="transparent")
        g.edge("bb",  "att", color="transparent")
        g.edge("att", "head", color="transparent")
    else:
        g.edge("inp", "bb",  color="transparent")
        g.edge("bb",  "att", color="transparent")
        g.edge("att", "head", color="transparent")

    # render PDF (vector)
    out_pdf = Path(f"{out_prefix}.pdf")
    g.render(out_pdf.with_suffix("").as_posix(), cleanup=True)
    print(f"Saved PDF: {out_pdf}")

    # optional PNG preview @ dpi
    if png:
        g.format = "png"
        g.attr(dpi=str(dpi))
        out_png = Path(f"{out_prefix}_{dpi}dpi.png")
        g.render(out_png.with_suffix("").as_posix(), cleanup=True)
        print(f"Saved PNG: {out_png}")

def main():
    p = argparse.ArgumentParser(description="MobileViTv2 + Attention concise diagram")
    p.add_argument("--label", default="SAM", help="Attention label (e.g., SAM, C2PSA, CBAM, BAM)")
    p.add_argument("--orientation", default="h", choices=["h","v"],
                   help="h = horizontal (LR), v = vertical (TB)")
    p.add_argument("--out", default="MobileViTv2_Attention_Effect_H",
                   help="output filename prefix (no extension)")
    p.add_argument("--fontsize", type=int, default=14)
    p.add_argument("--nodesep", type=float, default=0.45)
    p.add_argument("--ranksep", type=float, default=0.55)
    p.add_argument("--png", action="store_true", help="also export PNG (600 dpi)")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    make_diagram(attn_label=args.label,
                 orientation=args.orientation,
                 out_prefix=args.out,
                 fontsize=args.fontsize,
                 nodesep=args.nodesep,
                 ranksep=args.ranksep,
                 png=args.png,
                 dpi=args.dpi)

if __name__ == "__main__":
    main()
