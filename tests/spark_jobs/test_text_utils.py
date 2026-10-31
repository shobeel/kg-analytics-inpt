import base64

import text_utils as tu

WIKI_HTML = """<div class="mw-parser-output">
<p><b>OpenAI</b> is an American AI organization<sup class="reference">[1]</sup> founded in
<a href="/wiki/2015">2015</a>.</p>
<table class="infobox"><tr><th>Founders</th><td>Sam Altman</td></tr></table>
<div class="navbox">Navigation junk</div>
<script>var x = 1;</script>
<div class="mw-heading mw-heading2"><h2 id="History">History</h2></div>
<p>It released GPT-4.</p>
<div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
<ol class="references"><li>Cite me</li></ol>
<p>After references.</p>
</div>"""

ATOM_ENTRY = ('<entry xmlns="http://www.w3.org/2005/Atom"><title>LLaMA:\n  Open Models</title>'
              '<summary>We introduce\n LLaMA.</summary></entry>')


def test_html_cleaning_keeps_content_drops_noise():
    text = tu.extract_text("text/html", "utf-8", WIKI_HTML)
    assert "OpenAI is an American AI organization founded in 2015." in text
    assert "Founders Sam Altman" in text                 # infobox facts kept
    assert "It released GPT-4." in text
    for noise in ("[1]", "Navigation junk", "var x", "References", "Cite me", "After references"):
        assert noise not in text


def test_article_tag_preferred_on_full_pages():
    body = "<p>" + "Real article sentence. " * 40 + "</p>"
    html = f"<html><nav>Menu</nav><div>Sidebar promo</div><article>{body}</article></html>"
    text = tu.extract_text("text/html", "utf-8", html)
    assert "Sidebar promo" not in text and text.startswith("Real article sentence.")


def test_atom_entry_title_and_summary():
    assert tu.extract_text("application/atom+xml", "utf-8", ATOM_ENTRY) == "LLaMA: Open Models\nWe introduce LLaMA."


def test_pdf_extraction():
    from pypdf import PdfWriter
    import io

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    assert tu.extract_text("application/pdf", "base64", base64.b64encode(buf.getvalue()).decode()) == ""


def test_language_detection():
    assert tu.detect_language("The transformer architecture is used by most large language models today.")[0] == "en"
    assert tu.detect_language("Le modèle de langage est entraîné sur un grand corpus de textes en français.")[0] == "fr"
    assert tu.detect_language("short") == ("unknown", 0.0)


def test_chunking_size_overlap_and_offsets():
    sentences = [f"Sentence number {i} talks about models and data." for i in range(300)]  # 8 words each
    text = " ".join(sentences)
    chunks = tu.chunk_text(text, chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c["n_words"] <= 600 and c["n_tokens_est"] <= 800
        assert text[c["char_start"]:c["char_end"]] == c["text"]          # provenance offsets
        assert c["text"].startswith("Sentence") and c["text"].endswith(".")   # sentence boundaries
    for a, b in zip(chunks, chunks[1:]):
        overlap = a["char_end"] - b["char_start"]
        assert overlap > 0                                                 # chunks overlap
        assert 64 <= len(text[b["char_start"]:a["char_end"]].split()) <= 75   # ~100 tokens
    assert chunks[-1]["char_end"] == len(text)                             # nothing lost


def test_chunking_splits_overlong_sentence_and_handles_short_text():
    long = " ".join(["word"] * 1500)                                     # no punctuation
    chunks = tu.chunk_text(long)
    assert all(c["n_words"] <= 600 for c in chunks) and chunks[-1]["char_end"] == len(long)
    assert tu.chunk_text("") == []
    [one] = tu.chunk_text("Just one short sentence.")
    assert one["text"] == "Just one short sentence." and one["chunk_index"] == 0


BOILERPLATE_PAGE = """<html><head><title>Blog</title></head><body>
<nav><a href="/">Home</a> <a href="/models">Models</a> <a href="/pricing">Pricing</a></nav>
<div class="promo">Sign up or log in to comment. Upvote 62.</div>
<article><h1>Welcome Beam, a new open model</h1>
""" + "".join(f"<p>Paragraph {i}: the Beam model was trained by the Reflection lab on a large mixture of "
              f"public datasets and released with open weights under a permissive license.</p>"
              for i in range(8)) + """
</article>
<footer>Press contact: Jane Doe, jane@example.com, 617-000-0000. Related stories. Privacy policy.</footer>
</body></html>"""


def test_full_web_page_uses_main_content_extraction():
    text, method = tu.extract_text_with_method("text/html", "utf-8", BOILERPLATE_PAGE)
    assert method == "trafilatura"
    assert "trained by the Reflection lab" in text and "Paragraph 7" in text
    for junk in ("Sign up or log in", "Pricing", "jane@example.com", "Privacy policy"):
        assert junk not in text


def test_wikipedia_fragment_and_short_snippet_use_bs4():
    assert tu.extract_text_with_method("text/html", "utf-8", WIKI_HTML)[1] == "bs4"
    assert tu.extract_text_with_method("text/html", "utf-8", "<p>Short feed summary.</p>") == ("Short feed summary.", "bs4")


def test_pdf_reference_list_is_cut():
    body = "\n".join(f"Body sentence {i} about attention." for i in range(40))
    text = f"Intro\n{body}\nReferences\n[1] A. Author. Some paper. 2015.\n[2] B. Author. Other. 1993."
    out = tu.cut_references(text)
    assert out.endswith("Body sentence 39 about attention.") and "[1] A. Author" not in out
    # a 'References' line early in the document (e.g. a table of contents) is not a cut point
    early = "Contents\nReferences\n" + body
    assert tu.cut_references(early) == early
    assert tu.cut_references("no heading here") == "no heading here"
