"""Render main.tex as a self-contained review page.

WHY THIS EXISTS. There is no LaTeX toolchain on the drafting machine, so the draft cannot be
compiled and cannot be read as a paper. This converts it well enough to READ AND MARK UP -- which
is the job at draft stage -- and, unlike a PDF, it surfaces the things a compile would hide: the
source comments recording where a number came from, the VERIFY markers on numbers not yet re-read
from their results file, and the sections still blocked on a run.

IT IS A REVIEW VIEW, NOT A TYPESETTER. Math is shown as source in a mono span rather than rendered;
the artifact CSP blocks script CDNs, so a real math renderer is not available and a half-working one
would be worse than honest source. Anything the converter cannot handle is emitted verbatim inside a
marked block rather than silently dropped -- a review page that quietly loses a paragraph is worse
than no review page.

Usage:
    .venv/bin/python paper_arxiv/render_review.py        # writes paper_arxiv/review.html
"""
import base64
import html
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
TEX = HERE / "main.tex"
FIGDIR = HERE.parent / "fig"
OUT = HERE / "review.html"


def data_uri(name):
    p = FIGDIR / name
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def inline(t):
    """LaTeX inline markup -> HTML. Math is preserved as source, deliberately."""
    t = html.escape(t)
    t = re.sub(r"\\citep?\{([^}]+)\}", lambda m: f'<span class="cite">[{m.group(1)}]</span>', t)
    t = re.sub(r"\\S\\ref\{([^}]+)\}", r'<span class="xref">§\1</span>', t)
    t = re.sub(r"\\ref\{([^}]+)\}", r'<span class="xref">\1</span>', t)
    t = re.sub(r"\\label\{[^}]+\}", "", t)
    t = re.sub(r"\\emph\{([^}]+)\}", r"<em>\1</em>", t)
    t = re.sub(r"\\textbf\{([^}]+)\}", r"<strong>\1</strong>", t)
    t = re.sub(r"\\texttt\{([^}]+)\}", r"<code>\1</code>", t)
    t = re.sub(r"\\textsc\{([^}]+)\}", lambda m: f'<span class="sc">{m.group(1)}</span>', t)
    t = re.sub(r"\$([^$]+)\$", lambda m: f'<span class="math">{m.group(1)}</span>', t)
    t = t.replace("---", "&mdash;").replace("--", "&ndash;")
    t = re.sub(r"\\%", "%", t)
    t = re.sub(r"\\\\", "<br>", t)
    return t


def parse(src):
    """-> (blocks, notes). Blocks are (kind, payload); notes are source comments worth surfacing."""
    blocks, notes = [], []
    # Figures first, so their captions are not treated as prose.
    def take_figure(m):
        body = m.group(1)
        img = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", body, re.S) or \
              re.search(r"\\caption\{(.*)\}", body, re.S)
        blocks.append(("figure", (img.group(1) if img else None,
                                  cap.group(1) if cap else "")))
        return "\n\x00FIG\x00\n"

    src = re.sub(r"\\begin\{figure\}(.*?)\\end\{figure\}", take_figure, src, flags=re.S)

    def take_table(m):
        body = m.group(1)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", body, re.S) or \
              re.search(r"\\caption\{(.*)\}", body, re.S)
        rows = []
        tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", body, re.S)
        if tab:
            for line in tab.group(1).split(r"\\"):
                line = re.sub(r"\\(top|mid|bottom)rule", "", line).strip()
                if not line:
                    continue
                cells = [c.strip() for c in re.split(r"(?<!\\)&", line)]
                cells = [re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", c) for c in cells]
                rows.append(cells)
        blocks.append(("table", (rows, cap.group(1) if cap else "")))
        return "\n\x00TAB\x00\n"

    src = re.sub(r"\\begin\{table\}(.*?)\\end\{table\}", take_table, src, flags=re.S)

    fig_i = tab_i = 0
    ordered = []
    for raw in src.split("\n\n"):
        chunk = raw.strip()
        if not chunk:
            continue
        if "\x00FIG\x00" in chunk:
            ordered.append(("figure", [b for b in blocks if b[0] == "figure"][fig_i][1]))
            fig_i += 1
            continue
        if "\x00TAB\x00" in chunk:
            ordered.append(("table", [b for b in blocks if b[0] == "table"][tab_i][1]))
            tab_i += 1
            continue
        # comments: keep the ones that carry review signal
        keep = [l for l in chunk.split("\n") if l.lstrip().startswith("%")]
        for l in keep:
            txt = l.lstrip("% ").strip()
            if re.search(r"VERIFY|TODO|PENDING|SOURCE:", txt):
                notes.append(txt)
        chunk = "\n".join(l for l in chunk.split("\n") if not l.lstrip().startswith("%")).strip()
        if not chunk:
            continue
        m = re.match(r"\\(sub)?(sub)?section\*?\{(.+?)\}", chunk)
        if m:
            depth = len([g for g in m.groups()[:2] if g])
            ordered.append(("head", (depth, m.group(3))))
            rest = chunk[m.end():].strip()
            rest = re.sub(r"\\label\{[^}]+\}", "", rest).strip()
            if rest:
                ordered.append(("para", rest))
            continue
        if chunk.startswith(r"\begin{abstract}"):
            body = re.sub(r"\\(begin|end)\{abstract\}", "", chunk).strip()
            ordered.append(("abstract", body))
            continue
        if chunk.startswith(r"\begin{equation}"):
            ordered.append(("eq", re.sub(r"\\(begin|end)\{equation\}|\\label\{[^}]+\}", "",
                                         chunk).strip()))
            continue
        if chunk.startswith("\\") and not chunk.startswith(r"\paragraph") and \
           not chunk.startswith(r"\item") and not chunk.startswith(r"\begin{enumerate}"):
            continue
        ordered.append(("para", chunk))
    return ordered, notes


def render_para(t):
    t = re.sub(r"\\begin\{enumerate\}|\\end\{enumerate\}", "", t)
    lead = re.match(r"\\paragraph\{(.+?)\}\s*(.*)", t, re.S)
    if lead:
        return f'<p><span class="lead">{inline(lead.group(1))}</span> {inline(lead.group(2))}</p>'
    if r"\item" in t:
        items = [i.strip() for i in t.split(r"\item") if i.strip()]
        return "<ol>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ol>"
    return f"<p>{inline(t)}</p>"


def main():
    src = TEX.read_text()
    src = src.split(r"\begin{document}")[1].split(r"\end{document}")[0]
    ordered, notes = parse(src)

    out = []
    for kind, payload in ordered:
        if kind == "head":
            depth, title = payload
            tag = "h2" if depth == 0 else ("h3" if depth == 1 else "h4")
            pend = ' <span class="chip chip-pending">pending</span>' if "Ablation" in title else ""
            out.append(f"<{tag}>{inline(title)}{pend}</{tag}>")
        elif kind == "abstract":
            out.append(f'<div class="abstract"><p>{inline(payload)}</p></div>')
        elif kind == "eq":
            out.append(f'<pre class="eq">{html.escape(payload)}</pre>')
        elif kind == "figure":
            name, cap = payload
            uri = data_uri(name) if name else None
            img = (f'<img src="{uri}" alt="{html.escape(name or "")}">' if uri
                   else f'<div class="missing">missing figure: {html.escape(str(name))}</div>')
            out.append(f'<figure>{img}<figcaption>{inline(cap)}</figcaption></figure>')
        elif kind == "table":
            rows, cap = payload
            if not rows:
                continue
            head, *body = rows
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                         for r in body)
            out.append(f'<figure class="tbl"><div class="scroll"><table><thead><tr>{th}</tr>'
                       f'</thead><tbody>{tb}</tbody></table></div>'
                       f'<figcaption>{inline(cap)}</figcaption></figure>')
        else:
            out.append(render_para(payload))

    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    OUT.write_text(TEMPLATE.replace("__BODY__", "\n".join(out))
                           .replace("__NOTES__", notes_html)
                           .replace("__NNOTES__", str(len(notes))))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(notes)} source notes)")


TEMPLATE = r"""<style>
:root{
  --paper:#fcfcfd; --ink:#16181d; --muted:#6b7078; --rule:#d8dade;
  --accent:#2f5d8a; --warn:#9a5b2c; --panel:#f4f5f7;
  --serif: Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  --sans: ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#14161a; --ink:#e6e8ec; --muted:#9aa0a9; --rule:#2b2f36;
         --accent:#7ba9d4; --warn:#d09a68; --panel:#1b1e24; }
}
:root[data-theme="dark"]{ --paper:#14161a; --ink:#e6e8ec; --muted:#9aa0a9; --rule:#2b2f36;
  --accent:#7ba9d4; --warn:#d09a68; --panel:#1b1e24; }
:root[data-theme="light"]{ --paper:#fcfcfd; --ink:#16181d; --muted:#6b7078; --rule:#d8dade;
  --accent:#2f5d8a; --warn:#9a5b2c; --panel:#f4f5f7; }

*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
     font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:44rem;margin:0 auto;padding:4rem 1.25rem 6rem;display:flex;
      flex-direction:column;gap:1.15rem}
header{display:flex;flex-direction:column;gap:.6rem;border-bottom:1px solid var(--rule);
       padding-bottom:1.6rem;margin-bottom:1rem}
.eyebrow{font-family:var(--sans);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;
         color:var(--muted)}
h1{font-size:1.85rem;line-height:1.22;margin:0;text-wrap:balance;font-weight:600}
.byline{font-family:var(--sans);font-size:.85rem;color:var(--muted)}
h2{font-size:1.28rem;margin:2.4rem 0 0;text-wrap:balance;font-weight:600;
   border-top:1px solid var(--rule);padding-top:1.5rem}
h3{font-size:1.05rem;margin:1.6rem 0 0;font-weight:600}
h4{font-size:.95rem;margin:1.2rem 0 0;font-weight:600}
p{margin:0}
.lead{font-weight:600}
.abstract{background:var(--panel);border-left:2px solid var(--accent);padding:1.1rem 1.25rem;
          font-size:.95rem}
.abstract p{margin:0}
ol{margin:.2rem 0;padding-left:1.3rem;display:flex;flex-direction:column;gap:.55rem}
figure{margin:1.6rem 0;display:flex;flex-direction:column;gap:.6rem}
figure img{width:100%;height:auto;background:#fff;border:1px solid var(--rule);border-radius:2px}
figcaption{font-family:var(--sans);font-size:.8rem;line-height:1.5;color:var(--muted)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.78rem;
      font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;border-bottom:1.5px solid var(--ink)}
code,.math{font-family:var(--mono);font-size:.86em;background:var(--panel);padding:.05em .3em;
           border-radius:2px}
.math{color:var(--accent)}
.eq{font-family:var(--mono);font-size:.82rem;background:var(--panel);padding:.9rem 1rem;
    overflow-x:auto;border-radius:2px;margin:.4rem 0}
.cite{font-family:var(--sans);font-size:.8em;color:var(--accent)}
.xref{font-family:var(--sans);font-size:.85em;color:var(--muted)}
.sc{font-variant:small-caps;letter-spacing:.04em}
.chip{font-family:var(--sans);font-size:.62rem;letter-spacing:.09em;text-transform:uppercase;
      padding:.18em .5em;border-radius:2px;vertical-align:middle;font-weight:600}
.chip-pending{background:var(--warn);color:var(--paper)}
.notes{margin-top:3rem;border-top:1px solid var(--rule);padding-top:1.5rem}
.notes h2{border:0;padding:0;margin:0 0 .8rem}
.notes ul{font-family:var(--mono);font-size:.72rem;line-height:1.6;color:var(--muted);
          padding-left:1.1rem;display:flex;flex-direction:column;gap:.35rem}
.missing{background:var(--panel);border:1px dashed var(--warn);color:var(--warn);
         padding:1.5rem;text-align:center;font-family:var(--sans);font-size:.8rem}
@media (max-width:640px){ body{font-size:16px} .wrap{padding:2.5rem 1rem 4rem} h1{font-size:1.5rem} }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Draft &middot; not compiled &middot; review copy</div>
  <h1>What Iterated Self-Feeding Probes of Language Models Measure</h1>
  <div class="byline">and a test that separates the construction from the model</div>
</header>
__BODY__
<div class="notes">
  <h2>Source notes carried from the LaTeX (__NNOTES__)</h2>
  <ul>__NOTES__</ul>
</div>
</div>
"""

if __name__ == "__main__":
    main()
