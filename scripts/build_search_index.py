#!/usr/bin/env python3
"""Build the docs search index from the pages themselves.

WHY. Search used to run on a hand-written list in js/main.js: 99 entries for
25 pages, and on 2026-10-01 twenty of them (one in five) pointed at anchors that
no longer existed. A reader searching for "bootstrap" or "disparate impact"
landed at the top of a 312,000 px page. A list someone has to remember to update
goes stale; an index read from the pages cannot point at a heading that is not
there.

WHAT. Every page in docs/site, every h1 to h4 that has an anchor (its own id or
the nearest enclosing element's), and every documented API object
(.api-function-name), each with the page it lives on, the section above it and
the first sentence of the text that follows. Status chips, badges and other
decoration inside a heading are not part of its name and are dropped.

    python scripts/build_search_index.py           write docs/site/data/search-index.json
    python scripts/build_search_index.py --check   exit 1 if the published index is stale

Standard library only, so it runs wherever the docs are built.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
OUT = SITE / "data" / "search-index.json"

VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
# Text inside these never describes the content a reader is looking for.
SKIP = {"script", "style", "svg", "noscript", "template", "button", "nav", "header", "footer"}
# Decoration that sits INSIDE headings: status chips, badges, labels.
DECOR = re.compile(r"\b(cap-status|cap-scope|badge|limitation-label|version-badge|chip|sr-only)\b")
# Pages that are not documentation a reader searches.
EXCLUDE = {"404.html", "legal/license.html"}


class Node:
    __slots__ = ("tag", "attrs", "children", "parent", "text")

    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: list = []
        self.text = ""


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", [], None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def text_of(n, skip_decor=True) -> str:
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
            return
        if x.tag in SKIP:
            return
        if skip_decor and DECOR.search(x.attrs.get("class") or ""):
            return
        for c in x.children:
            walk(c)

    walk(n)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def walk_nodes(n):
    for c in n.children:
        if isinstance(c, Node):
            yield c
            yield from walk_nodes(c)


def anchor_of(n):
    q = n
    while q is not None and q.tag != "#root":
        if q.attrs.get("id"):
            return q.attrs["id"]
        q = q.parent
    return None


def next_text(n, limit=170) -> str:
    """The first paragraph-like text after node n, inside the same parent chain."""
    q = n
    while q is not None and q.parent is not None:
        sibs = q.parent.children
        i = sibs.index(q)
        for s in sibs[i + 1 :]:
            if isinstance(s, Node):
                if s.tag in ("h1", "h2", "h3", "h4", "section"):
                    return ""
                t = text_of(s)
                if len(t) > 30 and s.tag in ("p", "div", "ul", "ol", "table", "dl"):
                    return (t[: limit - 1] + "…") if len(t) > limit else t
        q = q.parent
        if q.tag in ("section", "main", "article"):
            break
    return ""


def page_url(rel: str) -> str:
    return rel[: -len("index.html")] if rel.endswith("index.html") else rel


def index_page(path: Path) -> list[dict]:
    rel = path.relative_to(SITE).as_posix()
    t = Tree()
    t.feed(path.read_text(encoding="utf-8"))
    nodes = list(walk_nodes(t.root))
    title = next((text_of(n) for n in nodes if n.tag == "title"), rel)
    title = re.split(r"\s+[|—–-]\s+(vfairness|validant)", title)[0].strip() or rel
    desc = next(
        (
            n.attrs.get("content", "")
            for n in nodes
            if n.tag == "meta" and n.attrs.get("name") == "description"
        ),
        "",
    )
    url = page_url(rel)
    main = next((n for n in nodes if n.tag == "main"), t.root)
    entries = [{"t": title, "u": url, "s": title, "k": "page", "d": desc[:170]}]
    seen = {url}
    section = title
    for n in walk_nodes(main):
        cls = n.attrs.get("class") or ""
        if n.tag in ("h1", "h2", "h3", "h4"):
            name = text_of(n)
            if not name or len(name) > 140:
                continue
            if n.tag == "h2":
                section = name
            a = anchor_of(n)
            if not a:
                continue
            u = f"{url}#{a}"
            if u in seen:
                continue
            seen.add(u)
            entries.append(
                {
                    "t": name,
                    "u": u,
                    "s": title if n.tag in ("h1", "h2") else section,
                    "k": "section",
                    "d": next_text(n),
                }
            )
        elif "api-function-name" in cls.split():
            name = re.sub(r"^(class|def)\s+", "", text_of(n))
            a = anchor_of(n)
            if not name or not a:
                continue
            u = f"{url}#{a}"
            key = u + "|" + name
            if key in seen:
                continue
            seen.add(key)
            body = n
            while (
                body.parent is not None
                and "api-function" not in (body.attrs.get("class") or "").split()
            ):
                body = body.parent
            d = ""
            for m in walk_nodes(body):
                if m.tag == "p":
                    d = text_of(m)
                    break
            entries.append(
                {
                    "t": name,
                    "u": u,
                    "s": section,
                    "k": "api",
                    "d": (d[:169] + "…") if len(d) > 170 else d,
                }
            )
    return entries


def build() -> dict:
    pages = sorted(
        p
        for p in SITE.rglob("*.html")
        if p.relative_to(SITE).as_posix() not in EXCLUDE and "test-objects" not in p.parts
    )
    items, seen = [], set()
    for p in pages:
        for it in index_page(p):
            # A class documented under its own heading appears twice (the
            # heading and its API name): one entry per name and place.
            key = (it["t"].lower(), it["u"])
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    return {"version": 1, "count": len(items), "items": items}


def main(argv) -> int:
    data = build()
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(
                f"STALE: {OUT.relative_to(ROOT)} does not match the pages; "
                "run scripts/build_search_index.py"
            )
            return 1
        print(f"ok: {data['count']} entries")
        return 0
    OUT.write_text(text, encoding="utf-8")
    kinds = {}
    for i in data["items"]:
        kinds[i["k"]] = kinds.get(i["k"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}: {data['count']} entries {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
