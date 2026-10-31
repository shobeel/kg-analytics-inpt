"""Pure-Python text helpers for the bronze -> silver job.

Shipped to executors with --py-files, and unit-tested without Spark (tests/spark_jobs).
Token counts are estimates: 1 token ~= 0.75 English words (common rule of thumb), so the
default 800-token chunk ~= 600 words and the 100-token overlap ~= 75 words.
"""

import base64
import io
import math
import re
import unicodedata
import xml.etree.ElementTree as ET

WORDS_PER_TOKEN = 0.75
MIN_MAIN_CONTENT_WORDS = 50   # below this, trafilatura output is treated as a miss

# ---------------------------------------------------------------------------- cleaning

DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
             "svg", "iframe", "button", "figure", "link", "meta"]
# Wikipedia (MediaWiki parser output) noise: citations, navboxes, maintenance banners.
DROP_SELECTORS = [".mw-editsection", "sup.reference", ".reflist", ".references",
                  ".mw-references-wrap", ".navbox", ".navbox-styles", ".metadata", ".hatnote",
                  ".sistersitebox", ".noprint", ".shortdescription", ".ambox", ".sidebar",
                  ".mw-empty-elt", ".thumb", ".gallery"]
# Everything from these section headings onward is references / links, not content.
STOP_SECTIONS = {"references", "notes", "external links", "further reading", "see also",
                 "bibliography", "citations", "sources", "footnotes"}
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
              "table", "br", "section", "article", "blockquote", "pre", "dd", "dt"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    lines = (re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup  # imported here so the module loads without bs4

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for sel in DROP_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()
    for h2 in soup.find_all("h2"):
        if h2.get_text(" ", strip=True).lower() in STOP_SECTIONS:
            start = h2.parent if "mw-heading" in (h2.parent.get("class") or []) else h2
            for sib in list(start.find_next_siblings()):
                sib.decompose()
            start.decompose()
            break
    # Prefer the main content container of full web pages.
    root = soup
    for candidate in (soup.find("article"), soup.find("main")):
        if candidate is not None and len(candidate.get_text(strip=True)) > 500:
            root = candidate
            break
    # Source line breaks inside text are just spaces in HTML (except in <pre>).
    for s in root.find_all(string=True):
        if s.find_parent("pre") is None and re.search(r"\s", s):
            s.replace_with(re.sub(r"\s+", " ", s))
    for tag in root.find_all(BLOCK_TAGS):
        tag.append("\n")
    for cell in root.find_all(["td", "th"]):
        cell.append(" ")
    return normalize(root.get_text(""))


def atom_entry_to_text(xml: str) -> str:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml)
    title = " ".join((entry.findtext("a:title", default="", namespaces=ns)).split())
    summary = " ".join((entry.findtext("a:summary", default="", namespaces=ns)).split())
    return normalize(f"{title}\n{summary}")


REFERENCES_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*$", re.I | re.M)


def cut_references(text: str) -> str:
    """Drop a paper's reference list (and anything after it), as done for Wikipedia sections:
    bibliographies yield many off-topic author/paper entities. Uses the LAST heading line that
    is exactly 'References'/'Bibliography', and only in the second half of the text."""
    last = None
    for m in REFERENCES_HEADING.finditer(text):
        last = m
    if last is not None and last.start() > len(text) // 2:
        return text[:last.start()].rstrip()
    return text


def pdf_to_text(b64: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(base64.b64decode(b64)))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    return normalize(cut_references(text))


def is_full_web_page(html: str) -> bool:
    """A whole site page (nav, footer, comments...) rather than a content fragment such as
    MediaWiki parser output or an RSS item body."""
    head = html[:5000].lower()
    return "mw-parser-output" not in head and ("<html" in head or "<body" in head)


def web_page_to_text(html: str) -> str:
    """Main-content extraction for full web pages (boilerplate removal with trafilatura)."""
    import trafilatura

    text = trafilatura.extract(html, include_comments=False, include_tables=True,
                               include_links=False, favor_precision=True, deduplicate=True)
    return normalize(text or "")


def extract_text_with_method(content_type: str, encoding: str, raw: str) -> tuple:
    """Raw bronze content -> (clean plain text, method used). Raises on unparseable input."""
    if not raw:
        return "", "empty"
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype == "application/pdf":
        return pdf_to_text(raw), "pypdf"
    if encoding == "base64":
        raw = base64.b64decode(raw).decode("utf-8", "replace")
    if ctype == "application/atom+xml":
        return atom_entry_to_text(raw), "atom"
    if ctype in ("text/html", "application/xhtml+xml") or "<" in raw[:1000]:
        if is_full_web_page(raw):
            text = web_page_to_text(raw)
            if count_words(text) >= MIN_MAIN_CONTENT_WORDS:
                return text, "trafilatura"
            return html_to_text(raw), "bs4-fallback"   # extractor found too little content
        return html_to_text(raw), "bs4"
    return normalize(raw), "plain"


def extract_text(content_type: str, encoding: str, raw: str) -> str:
    return extract_text_with_method(content_type, encoding, raw)[0]


# ---------------------------------------------------------------------------- language

def detect_language(text: str) -> tuple:
    """(iso code, probability); ('unknown', 0.0) for short or undetectable text."""
    if len(text) < 50:
        return "unknown", 0.0
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0                               # deterministic results
    try:
        best = detect_langs(text[:5000])[0]
        return best.lang, round(best.prob, 4)
    except LangDetectException:
        return "unknown", 0.0


# ---------------------------------------------------------------------------- chunking

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def estimate_tokens(n_words: int) -> int:
    return math.ceil(n_words / WORDS_PER_TOKEN)


def _units(text: str, max_words: int) -> list:
    """Sentence-ish spans (start, end, n_words); spans longer than max_words are split."""
    units = []
    for m in re.finditer(r"[^\n.!?]*(?:[.!?]+(?=\s|$)|\n|$)", text):
        start, end = m.span()
        words = list(re.finditer(r"\S+", text[start:end]))
        if not words:
            continue
        if len(words) <= max_words:
            units.append((start + words[0].start(), start + words[-1].end(), len(words)))
            continue
        for i in range(0, len(words), max_words):
            part = words[i:i + max_words]
            units.append((start + part[0].start(), start + part[-1].end(), len(part)))
    return units


def chunk_text(text: str, chunk_tokens: int = 800, overlap_tokens: int = 100) -> list:
    """Pack whole sentences into chunks of ~chunk_tokens, repeating the last ~overlap_tokens
    of each chunk at the start of the next. Offsets index into `text` (provenance)."""
    max_words = max(1, int(chunk_tokens * WORDS_PER_TOKEN))
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)
    units = _units(text, max_words)
    chunks, i = [], 0
    while i < len(units):
        j, words = i, 0
        while j < len(units) and (words + units[j][2] <= max_words or j == i):
            words += units[j][2]
            j += 1
        start, end = units[i][0], units[j - 1][1]
        chunks.append({"chunk_index": len(chunks), "text": text[start:end], "char_start": start,
                       "char_end": end, "n_words": words, "n_tokens_est": estimate_tokens(words)})
        if j >= len(units):
            break
        # Step back over whole units until ~overlap_words are repeated, but always advance.
        k, back = j, 0
        while k - 1 > i and back + units[k - 1][2] <= overlap_words:
            k -= 1
            back += units[k][2]
        i = k
    return chunks
