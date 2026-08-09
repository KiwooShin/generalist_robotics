"""Convert research.md into a styled standalone HTML page."""

import html
import re
import sys

SRC = "/home/kiwoos/work/generalist_robotics/research.md"
OUT = "research_page.html"


CODE_SENTINEL = "\x00CODE{}\x00"


def inline(text):
    """Convert inline markdown to HTML, keeping code spans opaque to emphasis."""
    codes = []

    def stash(match):
        codes.append(match.group(1))
        return CODE_SENTINEL.format(len(codes) - 1)

    s = re.sub(r"`([^`]+)`", stash, text)
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    for idx, code in enumerate(codes):
        s = s.replace(CODE_SENTINEL.format(idx), "<code>" + html.escape(code) + "</code>")
    return s


def is_quote(line):
    """True when a line opens a blockquote rather than merely starting with '>'."""
    stripped = line.strip()
    return stripped == ">" or stripped.startswith("> ")


def slugify(text):
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", s)


def render_table(rows):
    head, body = rows[0], rows[2:]
    cells = [c.strip() for c in head.strip().strip("|").split("|")]
    out = ['<div class="tablewrap"><table><thead><tr>']
    out += [f"<th>{inline(c)}</th>" for c in cells]
    out.append("</tr></thead><tbody>")
    for row in body:
        cs = [c.strip() for c in row.strip().strip("|").split("|")]
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cs) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(md):
    lines = md.split("\n")
    out, toc = [], []
    i = 0
    list_stack = []

    def close_lists(to=0):
        while len(list_stack) > to:
            out.append("</ul>")
            list_stack.pop()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        # table
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            close_lists()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(render_table(block))
            continue

        # blockquote
        if is_quote(line):
            close_lists()
            block = []
            while i < len(lines) and is_quote(lines[i]):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(block)) + "</blockquote>")
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            close_lists()
            level, text = len(m.group(1)), m.group(2)
            status = ""
            for mark, cls, label in (("✅", "ok", "verified"), ("⏳", "pend", "pending"),
                                     ("❌", "bad", "refuted")):
                if mark in text:
                    text = text.replace(mark, "").strip()
                    status = f'<span class="status {cls}">{label}</span>'
            slug = slugify(text)
            if level == 2:
                num = re.match(r"^(\d+)\.\s*(.*)$", text)
                if num:
                    eyebrow = f'<span class="sec-num">§{num.group(1)}</span>'
                    label = num.group(2)
                else:
                    eyebrow, label = "", text
                toc.append((num.group(1) if num else "", label, slug))
                out.append(f'<h2 id="{slug}">{eyebrow}<span class="sec-title">{inline(label)}</span>{status}</h2>')
            else:
                out.append(f"<h{level} id=\"{slug}\">{inline(text)}{status}</h{level}>")
            i += 1
            continue

        # horizontal rule
        if re.match(r"^(\*\*\*|---|___)$", stripped):
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        # list item
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            depth = len(m.group(1)) // 2 + 1
            while len(list_stack) < depth:
                out.append("<ul>")
                list_stack.append(1)
            close_lists(depth)
            content = m.group(2)
            # continuation lines
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip() or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^#{1,4}\s", nxt.strip()) \
                        or nxt.strip().startswith("|") or is_quote(nxt):
                    break
                content += " " + nxt.strip()
                j += 1
            i = j
            out.append(f"<li>{inline(content)}</li>")
            continue

        # paragraph
        close_lists()
        para = [stripped]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip() or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^#{1,4}\s", nxt.strip()) \
                    or nxt.strip().startswith("|") or is_quote(nxt):
                break
            para.append(nxt.strip())
            j += 1
        i = j
        out.append(f"<p>{inline(' '.join(para))}</p>")

    close_lists()
    return "\n".join(out), toc


CSS = """
:root{
  --ground:#F6F5F2; --panel:#FCFBF9; --ink:#191B1F; --ink-soft:#42464E;
  --muted:#6E727B; --rule:#DEDCD5; --rule-soft:#E9E7E1;
  --accent:#1B4DB1; --accent-soft:#E4EAF7;
  --flag:#9E4E14; --flag-soft:#F6E9DD;
  --ok:#2F6B4F; --ok-soft:#E2EFE8;
  --serif:'Iowan Old Style','Charter','Palatino Linotype',Palatino,Georgia,'Times New Roman',serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
  --mono:ui-monospace,'SF Mono','JetBrains Mono','Cascadia Mono',Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#131519; --panel:#191C21; --ink:#E4E5E2; --ink-soft:#B9BCC2;
    --muted:#8B9099; --rule:#2C3038; --rule-soft:#23272E;
    --accent:#82A8F2; --accent-soft:#1B2436;
    --flag:#DE9459; --flag-soft:#2E2118;
    --ok:#77BE9A; --ok-soft:#17251E;
  }
}
:root[data-theme="dark"]{
  --ground:#131519; --panel:#191C21; --ink:#E4E5E2; --ink-soft:#B9BCC2;
  --muted:#8B9099; --rule:#2C3038; --rule-soft:#23272E;
  --accent:#82A8F2; --accent-soft:#1B2436;
  --flag:#DE9459; --flag-soft:#2E2118;
  --ok:#77BE9A; --ok-soft:#17251E;
}
:root[data-theme="light"]{
  --ground:#F6F5F2; --panel:#FCFBF9; --ink:#191B1F; --ink-soft:#42464E;
  --muted:#6E727B; --rule:#DEDCD5; --rule-soft:#E9E7E1;
  --accent:#1B4DB1; --accent-soft:#E4EAF7;
  --flag:#9E4E14; --flag-soft:#F6E9DD;
  --ok:#2F6B4F; --ok-soft:#E2EFE8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.shell{display:grid;grid-template-columns:250px minmax(0,1fr);gap:56px;
  max-width:1180px;margin:0 auto;padding:0 28px}

/* masthead */
.masthead{grid-column:1/-1;border-bottom:2px solid var(--ink);
  padding:52px 0 22px;margin-bottom:44px}
.kicker{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin:0 0 14px}
h1{font-family:var(--sans);font-weight:680;letter-spacing:-.025em;
  font-size:clamp(2rem,5vw,3.1rem);line-height:1.04;margin:0 0 16px;text-wrap:balance}
.standfirst{font-size:1.12rem;color:var(--ink-soft);max-width:60ch;margin:0 0 22px}
.meta{display:flex;flex-wrap:wrap;gap:8px;font-family:var(--mono);font-size:11.5px}
.chip{border:1px solid var(--rule);padding:4px 9px;color:var(--muted);
  background:var(--panel);white-space:nowrap}
.chip b{color:var(--ink);font-weight:600}
.chip.ok{border-color:var(--ok);color:var(--ok);background:var(--ok-soft)}
.chip.flag{border-color:var(--flag);color:var(--flag);background:var(--flag-soft)}

/* nav rail */
nav{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;
  padding:6px 0 40px;font-family:var(--sans)}
nav .navlabel{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);padding:0 0 12px;
  border-bottom:1px solid var(--rule);margin-bottom:12px}
nav ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:1px}
nav a{display:grid;grid-template-columns:26px 1fr;gap:6px;align-items:baseline;
  text-decoration:none;color:var(--ink-soft);font-size:13.5px;line-height:1.35;
  padding:7px 8px;border-left:2px solid transparent}
nav a span:first-child{font-family:var(--mono);font-size:11px;color:var(--muted)}
nav a:hover{background:var(--accent-soft);color:var(--accent);border-left-color:var(--accent)}
nav a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

/* content */
main{min-width:0;padding-bottom:120px}
h2{font-family:var(--sans);font-weight:680;letter-spacing:-.02em;
  font-size:1.72rem;line-height:1.15;margin:76px 0 20px;padding-top:22px;
  border-top:1px solid var(--ink);display:flex;flex-wrap:wrap;align-items:baseline;
  gap:12px;text-wrap:balance;scroll-margin-top:16px}
h2:first-of-type{margin-top:0}
.sec-num{font-family:var(--mono);font-size:.82rem;font-weight:500;color:var(--accent);
  letter-spacing:.04em}
h3{font-family:var(--sans);font-weight:640;font-size:1.16rem;letter-spacing:-.012em;
  line-height:1.28;margin:42px 0 12px;color:var(--ink);text-wrap:balance;
  scroll-margin-top:16px}
h4{font-family:var(--mono);font-weight:600;font-size:.86rem;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);margin:30px 0 10px}
p{margin:0 0 16px;max-width:68ch}
ul{margin:0 0 18px;padding-left:1.15rem;max-width:68ch}
li{margin:0 0 9px}
li::marker{color:var(--muted)}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-soft)}
a:hover{border-bottom-color:var(--accent)}
strong{font-weight:660;color:var(--ink)}
code{font-family:var(--mono);font-size:.855em;background:var(--panel);
  border:1px solid var(--rule-soft);padding:1px 5px;color:var(--ink-soft)}
blockquote{margin:0 0 26px;padding:20px 24px;background:var(--panel);
  border-left:3px solid var(--flag);font-size:.95rem;color:var(--ink-soft);max-width:70ch}
blockquote strong{color:var(--ink)}
hr{border:0;border-top:1px solid var(--rule);margin:36px 0}

/* status pills */
.status{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;
  text-transform:uppercase;padding:3px 7px;border:1px solid;font-weight:500;
  position:relative;top:-2px}
.status.ok{color:var(--ok);border-color:var(--ok);background:var(--ok-soft)}
.status.pend{color:var(--flag);border-color:var(--flag);background:var(--flag-soft)}
.status.bad{color:var(--flag);border-color:var(--flag);background:var(--flag-soft)}

/* tables */
.tablewrap{overflow-x:auto;margin:0 0 26px;border:1px solid var(--rule);
  background:var(--panel)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:13.5px;
  line-height:1.45;font-variant-numeric:tabular-nums}
th{text-align:left;font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);font-weight:500;
  padding:11px 14px;border-bottom:1px solid var(--ink);white-space:nowrap}
td{padding:11px 14px;border-bottom:1px solid var(--rule-soft);
  color:var(--ink-soft);vertical-align:top}
tr:last-child td{border-bottom:0}
td strong{color:var(--ink)}
td code{font-size:.9em}

@media (max-width:940px){
  .shell{grid-template-columns:1fr;gap:0}
  nav{position:static;max-height:none;margin-bottom:36px;
    border-bottom:1px solid var(--rule);padding-bottom:16px}
  nav ol{flex-direction:row;flex-wrap:nowrap;overflow-x:auto;gap:6px;padding-bottom:6px}
  nav a{border-left:0;border:1px solid var(--rule);white-space:nowrap;
    grid-template-columns:auto auto;padding:6px 10px;background:var(--panel)}
  .masthead{padding-top:36px}
  body{font-size:16px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def main():
    md = open(SRC, encoding="utf-8").read()
    # drop the H1 and the leading blockquote intro; we rebuild them as the masthead
    md = re.sub(r"^# .*?\n", "", md, count=1)
    body, toc = convert(md)

    nav_items = "".join(
        f'<li><a href="#{slug}"><span>{"§" + num if num else "·"}</span>'
        f'<span>{html.escape(label)}</span></a></li>'
        for num, label, slug in toc
    )

    page = f"""<title>Generalist Robotics — Research Survey</title>
<style>{CSS}</style>
<div class="shell">
  <header class="masthead">
    <p class="kicker">Living survey · compiled 2026-08-09</p>
    <h1>Cross-embodiment robot policies</h1>
    <p class="standfirst">Can a policy pretrained on many robots adapt to a new robot far
    faster than training it from scratch &mdash; and how is that best achieved? A survey of
    the labs, models, datasets and mechanisms behind generalist robotics.</p>
    <div class="meta">
      <span class="chip"><b>9</b> sections</span>
      <span class="chip"><b>60+</b> papers &amp; releases</span>
      <span class="chip ok"><b>48</b> claims verified</span>
      <span class="chip flag"><b>9</b> corrected · <b>2</b> refuted</span>
      <span class="chip">DGX Spark &mdash; GB10 / aarch64</span>
    </div>
  </header>
  <nav>
    <div class="navlabel">Contents</div>
    <ol>{nav_items}</ol>
  </nav>
  <main>{body}</main>
</div>
"""
    open(OUT, "w", encoding="utf-8").write(page)
    print(f"wrote {OUT}  ({len(page)} bytes, {len(toc)} sections)")


if __name__ == "__main__":
    main()
