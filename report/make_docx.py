#!/usr/bin/env python3
"""Build the Word (.docx) version of the report from the LaTeX source.

Word cannot render TikZ, subfigures, or LaTeX cross-references, so the .tex is
first rewritten into a pandoc-friendly variant:

  * the TikZ pipeline diagram      -> figures/pipeline_diagram.png (pre-rendered)
  * subfigure environments         -> one figure per panel
  * \\ref{...}                      -> the literal number, read from MLCB_report.aux
                                      (so the .docx and the .pdf always agree)
  * captions                       -> hard-numbered ("Figure 1. ...", "Table 2. ...")

Usage:  python3 make_docx.py          (run from report/; needs pandoc + a fresh .aux)
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "MLCB_report.tex"
AUX = HERE / "MLCB_report.aux"
TMP = HERE / "_MLCB_report_docx.tex"
OUT = HERE / "MLCB_report.docx"

TITLE = ("Communication-Aware Machine Learning for Major Depressive Disorder: "
         "Testing a microglia → neuron communication axis from "
         "single-nucleus RNA-seq of the human prefrontal cortex")


def read_labels(aux: Path) -> dict:
    """label -> printed number, from the \\newlabel entries LaTeX just wrote."""
    labels = {}
    for line in aux.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", line)
        if m and not m.group(1).startswith("sub@"):
            labels[m.group(1)] = m.group(2)
    return labels


def expand_macro(text: str, name: str, wrap) -> str:
    """Expand \\name{arg} with brace matching (LaTeX macros pandoc shouldn't guess)."""
    out, i, tok = [], 0, "\\" + name + "{"
    while True:
        j = text.find(tok, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, depth = j + len(tok), 1
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        out.append(wrap(text[j + len(tok):k - 1]))
        i = k


def env_span(text: str, start: int, env: str):
    """Return (begin_idx, end_idx) of the environment opened at `start`."""
    b, e = "\\begin{%s}" % env, "\\end{%s}" % env
    depth, i = 0, start
    while i < len(text):
        nb, ne = text.find(b, i), text.find(e, i)
        if ne < 0:
            raise ValueError("unterminated " + env)
        if 0 <= nb < ne:
            depth, i = depth + 1, nb + len(b)
        else:
            depth, i = depth - 1, ne + len(e)
            if depth == 0:
                return start, i
    raise ValueError("unterminated " + env)


def flatten_figures(tex: str, labels: dict) -> str:
    """Split every subfigure-bearing figure into one plain figure per panel."""
    out, i = [], 0
    while True:
        j = tex.find("\\begin{figure}", i)
        if j < 0:
            out.append(tex[i:])
            return "".join(out)
        s, e = env_span(tex, j, "figure")
        block = tex[s:e]
        out.append(tex[i:s])

        if "\\begin{subfigure}" not in block:
            out.append(block)
        else:
            parent_lab = re.search(r"\\label\{(fig:[^}]+)\}\s*\\end\{figure\}", block)
            parent_cap = None
            for m in re.finditer(r"\\caption\{", block):
                pass  # the last \caption{...} in the block is the parent's
            if m:
                k, depth = m.end(), 1
                while depth:
                    depth += (block[k] == "{") - (block[k] == "}")
                    k += 1
                parent_cap = block[m.end():k - 1]
            pnum = labels.get(parent_lab.group(1), "?") if parent_lab else "?"

            panels = []
            p = 0
            while True:
                q = block.find("\\begin{subfigure}", p)
                if q < 0:
                    break
                qs, qe = env_span(block, q, "subfigure")
                sub = block[qs:qe]
                img = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", sub)
                cap = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end\{subfigure\})",
                                sub, re.S)
                panels.append((img.group(1) if img else None,
                               cap.group(1).strip() if cap else ""))
                p = qe

            letters = "abcdefgh"
            for n, (img, cap) in enumerate(panels):
                if not img:
                    continue
                tag = "Figure %s%s." % (pnum, letters[n])
                body = ("\\begin{figure}[h]\n\\centering\n"
                        "\\includegraphics[width=0.85\\textwidth]{%s}\n"
                        "\\caption{%s %s}\n\\end{figure}\n" % (img, tag, cap))
                out.append(body)
            if parent_cap:
                out.append("\n\\noindent\\textbf{Figure %s.} %s\n\n" % (pnum, parent_cap))
        i = e


def main() -> int:
    if not AUX.exists():
        sys.exit("MLCB_report.aux not found - run pdflatex first.")
    labels = read_labels(AUX)
    tex = TEX.read_text(encoding="utf-8")

    # 1. TikZ pipeline diagram -> pre-rendered PNG
    tex = re.sub(r"\\resizebox\{\\textwidth\}\{!\}\{%\s*\\begin\{tikzpicture\}.*?"
                 r"\\end\{tikzpicture\}%\s*\}",
                 "\\\\includegraphics[width=\\\\textwidth]{figures/pipeline_diagram.png}",
                 tex, flags=re.S)

    # 2. subfigures -> one figure per panel (before captions are numbered)
    tex = flatten_figures(tex, labels)

    # 3. hard-number the remaining captions from their \label
    def number_caption(m):
        block = m.group(0)
        lab = re.search(r"\\label\{((?:fig|tab):[^}]+)\}", block)
        if not lab:
            return block
        kind = "Figure" if lab.group(1).startswith("fig:") else "Table"
        num = labels.get(lab.group(1), "?")
        return block.replace("\\caption{", "\\caption{%s %s. " % (kind, num), 1)

    tex = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", number_caption, tex, flags=re.S)

    # 4. cross-references -> literal numbers; drop labels
    tex = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), tex)
    tex = re.sub(r"\\label\{[^}]+\}", "", tex)

    # 4b. citations: pandoc has no bibliography engine here, so number \cite{} by
    #     \bibitem order and rebuild the reference list as plain numbered paragraphs.
    bib = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}",
                    tex, re.S)
    entries = re.split(r"\\bibitem\{([^}]+)\}", bib.group(1))[1:]
    keys = entries[0::2]
    bodies = [" ".join(b.split()) for b in entries[1::2]]
    cite_no = {k: str(n) for n, k in enumerate(keys, 1)}

    refs = "\\section*{References}\n\n" + "\n\n".join(
        "\\textbf{[%d]}~%s" % (n, b) for n, b in enumerate(bodies, 1))
    tex = tex[:bib.start()] + refs + tex[bib.end():]
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: "[%s]" % ", ".join(cite_no.get(k.strip(), "?")
                                              for k in m.group(1).split(",")),
                 tex)

    # 5. constructs pandoc mishandles (drop the macro definitions before expanding)
    tex = re.sub(r"\\newcommand\{\\(factorfour|code)\}(\[\d\])?\{[^\n]*\}\n", "", tex)
    tex = expand_macro(tex, "code", lambda a: "\\texttt{%s}" % a)
    tex = expand_macro(tex, "mbox", lambda a: a)
    tex = tex.replace("\\factorfour", "\\textbf{Factor 4}")
    tex = re.sub(r"\\rowcolor\{[^}]*\}", "", tex)
    tex = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", tex)
    tex = re.sub(r"\\(cmidrule|midrule|toprule|bottomrule)(\(lr\))?(\{[^}]*\})?", "", tex)

    # 6. a clean title block (pandoc turns these into docx metadata)
    tex = re.sub(r"\\title\{.*?\}\s*\n\s*\\author\{.*?\}\s*\n\s*\\date\{[^}]*\}",
                 "\\\\title{%s}\n\\\\author{Giorgos Boulogeorgos \\\\and Andreas Mici}\n"
                 "\\\\date{MLCB Team Project --- Re-analysis of Maitra et al. (2023), "
                 "Nature Communications 14, 2912 --- June 2026}" % TITLE,
                 tex, flags=re.S, count=1)

    TMP.write_text(tex, encoding="utf-8")

    cmd = ["pandoc", str(TMP), "-f", "latex", "-t", "docx", "--standalone",
           "--resource-path", str(HERE), "-o", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return r.returncode
    if r.stderr.strip():
        print(r.stderr.strip())
    TMP.unlink()
    print("wrote", OUT, "(%.0f kB)" % (OUT.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
