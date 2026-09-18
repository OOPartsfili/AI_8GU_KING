"""Build a completely offline study reader from Markdown source files."""
from pathlib import Path
import collections, hashlib, html, json, re, sys
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "vendor"))
import markdown
from latex2mathml.converter import convert

PATTERN = re.compile(r"^## ([A-Z]+\d{2})｜(.+?)（(P[012])）\s*$", re.M)
CORE = """M01 M02 M03 M05 T01 T02 T04 T05 T06 T14 SFT01 SFT03
P03 P04 P05 P06 P08 P10 P11 P12 RL04 RL06 RL08 RL12
SA01 SA02 SA05 SA06 SA12 E01 E05 E07 E08 E09
R01 R04 R06 R09 R11 R14 A03 A05 A07 A09 I03 I10 V03 V08""".split()
assert len(CORE) == len(set(CORE)) == 48
BEGINNER = """M01 M04 M09 M02 M05 M07 T01 T02 T05 T04
SFT01 SFT03 P01 P06 P08 RL01 RL06 E01 R01 A01""".split()
assert len(BEGINNER) == len(set(BEGINNER)) == 20
DEEP_MARKER = "\n### 进一步：公式、边界与追问\n"
MATH_ERRORS = []
MATH_RECORDS = []
MATH_ALPHABET = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ"
    "𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫"
    "𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡"))
ET.register_namespace("", "http://www.w3.org/1998/Math/MathML")


def checked_math(latex, display):
    value = convert(latex, display=display)
    node = ET.fromstring(value)
    # MathML Core supports normal on mi, not legacy mathematical alphabets.
    # Emit the actual Unicode alphabet so Chromium displays expectation/set
    # symbols correctly instead of silently treating them as plain E and R.
    for e in node.iter():
        variant = e.get("mathvariant")
        if variant == "double-struck":
            if list(e) or any(c not in MATH_ALPHABET for c in (e.text or "")):
                raise ValueError("unsupported double-struck mapping")
            e.text = "".join(MATH_ALPHABET[c] for c in e.text)
            e.set("mathvariant", "normal")
        elif variant not in (None, "normal"):
            raise ValueError(f"math alphabet needs explicit Unicode mapping: {variant}")
    value = ET.tostring(node, encoding="unicode")
    tokens = "".join(node.itertext())
    if any(e.tag.rsplit("}", 1)[-1] == "merror" for e in node.iter()):
        raise ValueError("MathML error node")
    if re.search(r"\\[A-Za-z]+|MATHPLACEHOLDER|\ufffd", tokens):
        raise ValueError("unconverted command or invalid text in MathML")
    MATH_RECORDS.append({"latex": latex, "display": display})
    return value


def read(path):
    return path.read_text(encoding="utf-8")


def save(path, text):
    path.write_text(text, encoding="utf-8")


def markdown_link_targets(text):
    # Let Markdown distinguish links from code such as self.norms[0](x).
    rendered = markdown.markdown(text, extensions=["fenced_code"])
    return [html.unescape(value) for value in re.findall(r'<a\b[^>]*href="([^"]+)"', rendered)]


def render(text):
    chunks = []
    def formula(m):
        latex = m.group(1).strip()
        try:
            value = checked_math(latex, display="block")
        except Exception as exc:
            MATH_ERRORS.append({"formula": latex, "error": str(exc)})
            value = "<pre>" + html.escape(latex) + "</pre>"
        token = f"MATHPLACEHOLDER{len(chunks)}END"
        chunks.append('<div class="formula" tabindex="0" role="group" aria-label="数学公式；超出宽度时可横向滚动">' + value + '</div>')
        return "\n\n" + token + "\n\n"
    text = re.sub(r"\$\$(.*?)\$\$", formula, text, flags=re.S)
    if "$$" in text:
        MATH_ERRORS.append({"formula": text, "error": "unmatched block math delimiter"})
    # Inline math is uncommon but kept consistent with block mathematics.
    def inline(m):
        latex = m.group(1)
        token = f"MATHPLACEHOLDER{len(chunks)}END"
        try:
            value = checked_math(latex, display="inline")
        except Exception as exc:
            MATH_ERRORS.append({"formula": latex, "error": str(exc)})
            value = html.escape(latex)
        chunks.append(value)
        return token
    text = re.sub(r"(?<!\\)\$([^$\n]+)\$", inline, text)
    output = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for i, value in enumerate(chunks):
        marker = f"MATHPLACEHOLDER{i}END"
        output = output.replace("<p>"+marker+"</p>", value).replace(marker, value)
    return output


class Linkifier(HTMLParser):
    """Link only plain text; never touch existing anchors, code, or MathML."""
    def __init__(self, questions, sources):
        super().__init__(convert_charrefs=False)
        self.questions, self.sources, self.out = questions, sources, []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        self.out.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in self.stack:
            index = len(self.stack)-1-self.stack[::-1].index(tag)
            self.stack = self.stack[:index]
        self.out.append(f"</{tag}>")

    def handle_data(self, text):
        if any(t in self.stack for t in ("a", "code", "pre", "math", "script", "style")):
            self.out.append(text)
            return
        def replace(m):
            token = m.group(0)
            if token in self.questions:
                return f'<a class="ref" href="#q-{token}">{token}</a>'
            if token in self.sources:
                return f'<a class="ref" href="#source-{token}">{token}</a>'
            return token
        self.out.append(re.sub(r"(?<![A-Za-z0-9_])[A-Z]+\d{2,3}(?![A-Za-z0-9_])", replace, text))

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")

    def handle_comment(self, text):
        self.out.append("<!--"+text+"-->")


def linkify(text, question_ids, source_ids):
    parser = Linkifier(question_ids, source_ids)
    parser.feed(text)
    return "".join(parser.out)


def build_sources(sources):
    lines = ["# 来源索引", "", "核查基准日：2026-09-15。使用作者原论文、官方文档和作者/大学教材。网络可达只证明链接返回内容，不证明论文全部结论或正式接收；本资料不声称复现了这些论文。",
             "", "S 编号用于定位依据。正文的教学算例、工程方案与岗位建议是独立组织的解释，不能当作原论文的实测结果。", ""]
    for s in sources:
        lines += [f'<a id="{s["id"]}"></a>', f'## {s["id"]} · {s["title"]}', "",
                  f'[打开一手来源]({s["url"]})', "",
                  f'**用于：** {s["scope"]}', "",
                  f'**核查：** {s["checked_on"]} · {s["status"]}', ""]
    save(HERE / "来源索引.md", "\n".join(lines))


def main():
    sources = json.loads(read(HERE / "sources.json"))
    source_ids = {s["id"] for s in sources}
    if len(source_ids) != len(sources):
        raise ValueError("duplicate source IDs")
    build_sources(sources)
    files = ([ROOT / "README.md", ROOT / "00_导读/新人从这里开始.md",
              ROOT / "00_导读/术语翻译小词表.md", ROOT / "00_导读/岗位地图与学习路线.md"]
             + sorted((ROOT / "01_知识主线").glob("*.md"))
             + sorted((ROOT / "02_项目与面试").glob("*.md"))
             + sorted((ROOT / "03_代码实验").glob("*.md"))
             + [HERE / "来源索引.md", HERE / "维护规范.md", HERE / "新题与研究记录模板.md", HERE / "验收记录.md", HERE / "CHANGELOG.md", HERE / "可视化与手写教学设计.md"])
    questions, documents, errors = [], [], []
    raw_by_path = {}
    for path in files:
        text = read(path)
        if path.parent.name == "01_知识主线":
            text = text.split("\n## 本章来源")[0].rstrip()
            refs = sorted(set(re.findall(r"(?<![A-Z0-9])S\d{2,3}(?!\d)", text)), key=lambda sid: int(sid[1:]))
            if refs:
                text += "\n\n## 本章来源\n\n" + " · ".join(
                    f'[{sid}](../90_维护与来源/来源索引.md#{sid})' for sid in refs) + "\n"
        # Stable anchors also make the Markdown index useful outside this reader.
        text = re.sub(r'^<a id="([A-Z]+\d{2})"></a>\n(?=## \1｜)', "", text, flags=re.M)
        text = PATTERN.sub(lambda m: f'<a id="{m[1]}"></a>\n'+m[0], text)
        if text != read(path):
            save(path, text)
        rel = path.relative_to(ROOT).as_posix()
        doc_id = "d"+hashlib.sha1(rel.encode()).hexdigest()[:10]
        raw_by_path[rel] = text
        matches = list(PATTERN.finditer(text))
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match[1] if title_match else path.stem
        overview = text[:matches[0].start()] if matches else text
        overview = re.sub(r'<a id="[A-Z]+\d{2}"></a>\s*$', "", overview)
        doc_questions = []
        for i, match in enumerate(matches):
            body = text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)]
            body = re.sub(r'<a id="[A-Z]+\d{2}"></a>\s*$', "", body).strip()
            body = body.split("\n## 本章来源")[0].strip()
            qid, qtitle, priority = match.groups()
            source_refs = sorted(set(re.findall(r"(?<![A-Z0-9])S\d{2,3}(?!\d)", body)), key=lambda sid: int(sid[1:]))
            if any(s not in source_ids for s in source_refs):
                errors.append(f"{qid}: unknown source")
            if len(body) < 100:
                errors.append(f"{qid}: insufficient body")
            if DEEP_MARKER not in body:
                errors.append(f"{qid}: missing beginner/deeper reading structure")
            elif "**直答：**" not in body.split(DEEP_MARKER)[0]:
                errors.append(f"{qid}: missing plain-language direct answer")
            q = {"id": qid, "title": qtitle, "priority": priority, "doc": doc_id,
                 "path": rel, "body": body, "core": qid in CORE,
                 "beginner": qid in BEGINNER, "sources": source_refs}
            questions.append(q)
            doc_questions.append(qid)
        documents.append({"id": doc_id, "title": title, "path": rel,
                          "overview": overview, "questions": doc_questions})
    qids = [q["id"] for q in questions]
    if len(qids) != len(set(qids)):
        errors.append("duplicate question IDs")
    if any(q not in qids for q in CORE):
        errors.append("missing core questions")
    if any(q not in qids for q in BEGINNER):
        errors.append("missing beginner questions")
    # Validate Markdown file links and known literal anchors, excluding web links.
    for rel, text in raw_by_path.items():
        for target in markdown_link_targets(text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            clean, _, anchor = target.partition("#")
            target_path = (ROOT / rel).parent / clean
            if not target_path.exists():
                errors.append(f"{rel}: broken link {target}")
            elif anchor and target_path.suffix == ".md":
                if f'id="{anchor}"' not in read(target_path):
                    errors.append(f"{rel}: missing anchor {anchor}")
    qset = set(qids)
    for q in questions:
        intro, _, deeper = q["body"].partition(DEEP_MARKER)
        q["html"] = (linkify(render(intro), qset, source_ids)
                     + '<details class="deep"><summary>我已读懂例子，继续看公式、边界与追问</summary>'
                     + '<div class="deepbody">'
                     + linkify(render(deeper), qset, source_ids)
                     + '</div></details>')
    for d in documents:
        d["html"] = linkify(render(d["overview"]), qset, source_ids)
    # Convert generated source-footer links into in-reader navigation.
    for item in questions + documents:
        item["html"] = re.sub(r'href="\.\./90_维护与来源/来源索引\.md#(S\d{2,3})"',
                              r'href="#source-\1"', item["html"])
    if MATH_ERRORS:
        errors += [f"math render: {x}" for x in MATH_ERRORS]
    chars = sum(len(v) for k, v in raw_by_path.items() if k.startswith(("01_", "02_")))
    counts = dict(collections.Counter(q["priority"] for q in questions))
    report = {"question_count": len(questions), "main_chapters": len(list((ROOT/"01_知识主线").glob("*.md"))),
              "source_count": len(sources), "authored_main_characters": chars,
              "priority_counts": counts, "core_count": len(CORE),
              "beginner_count": len(BEGINNER), "teaching_coverage": len(questions),
              "math_errors": MATH_ERRORS,
              "math_count": len(MATH_RECORDS),
              "structural_errors": errors,
              "scope": "结构、引用存在性与数学转换检查；不替代内容和浏览器审查"}
    save(HERE / "build-report.json", json.dumps(report, ensure_ascii=False, indent=2))
    save(HERE / "formula-index.json", json.dumps(MATH_RECORDS, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    index = ["# 全部题目索引", "", f"共 {len(questions)} 题；按固定题号维护。P0/P1/P2 是岗位优先级，不是实际公司题频。", "",
             "| 题号 | 问题 | 优先级 |", "|---|---|---|"]
    for q in questions:
        index.append(f'| {q["id"]} | [{q["title"]}](../{q["path"]}#{q["id"]}) | {q["priority"]} |')
    save(ROOT / "00_导读/全部题目索引.md", "\n".join(index)+"\n")
    lookup = {q["id"]: q for q in questions}
    cards = ["# 48 道口述核心题", "", "这份清单由正文生成；先闭卷回答，再跳转原题接受追问。", ""]
    for qid in CORE:
        q = lookup[qid]
        direct = re.search(r"\*\*直答(?:骨架)?[：:]\*\*\s*(.+)", q["body"])
        brief = direct[1] if direct else "按反馈可得性、验证可靠性与总预算选择；完整对照表见原题。"
        cards += [f'## {qid} · {q["title"]}', "", brief, "", f'[展开与追问](../{q["path"]}#{qid})', ""]
    save(ROOT / "00_导读/48道口述核心题.md", "\n".join(cards))
    # Additional generated views are embedded too.
    for rel in ["00_导读/全部题目索引.md", "00_导读/48道口述核心题.md"]:
        body = read(ROOT / rel)
        value = render(body)
        value = re.sub(r'href="\.\./[^"]+?#([A-Z]+\d{2})"', r'href="#q-\1"', value)
        documents.append({"id": "index" if "全部" in rel else "core-guide",
                          "title": "全部题目索引" if "全部" in rel else "48 道口述核心题",
                          "path": rel, "html": value, "questions": []})
    # Keep only the directory in the startup JSON. Large records stay in inert
    # local data blocks and are parsed when a question/document is requested.
    # This remains a single HTML file that works directly over file://.
    question_meta = [{k: v for k, v in q.items() if k not in ("body", "html", "path")}
                     for q in questions]
    document_meta = [{k: v for k, v in d.items() if k not in ("overview", "html")}
                     for d in documents]
    payload = {"version": "1.5", "checked": "2026-09-16", "questions": question_meta,
               "documents": document_meta, "sources": sources, "core": CORE,
               "beginner": BEGINNER, "stats": report}
    def local_json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    records = []
    for q in questions:
        records.append(f'<script id="data-q-{q["id"]}" type="application/json">'
                       + local_json({"body": q["body"], "html": q["html"]}) + '</script>')
    for d in documents:
        records.append(f'<script id="data-doc-{d["id"]}" type="application/json">'
                       + local_json({"html": d["html"]}) + '</script>')
    template = read(HERE / "reader_template.html")
    template = template.replace("__VISUAL_CSS__", read(HERE / "reader_visuals.css"))
    template = template.replace("__VISUAL_JS__", read(HERE / "reader_visuals.js"))
    assets = {name: read(ROOT / "03_代码实验" / name) for name in (
        "transformer_handwrite.py", "test_transformer_handwrite.py", "Transformer手写运行与练习.md")}
    template = template.replace("__CODE_ASSETS__", local_json(assets))
    result = template.replace("__KNOWLEDGE_DATA__", local_json(payload))
    result = result.replace("__KNOWLEDGE_RECORDS__", "\n".join(records))
    save(ROOT / "开始阅读.html", result)
    save(HERE / "question-index.json", json.dumps(
        [{k:v for k,v in q.items() if k not in ("html", "body")} for q in questions],
        ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
