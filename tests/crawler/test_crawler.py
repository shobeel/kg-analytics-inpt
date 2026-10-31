from urllib.robotparser import RobotFileParser

from crawler.connectors import arxiv, rss
from crawler.http import PoliteSession
from crawler.models import BronzeDocument, doc_id
from crawler.store import BronzeStore

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2302.13971v1</id>
    <published>2023-02-27T18:57:36Z</published>
    <updated>2023-02-27T18:57:36Z</updated>
    <title>LLaMA: Open and Efficient
      Foundation Language Models</title>
    <summary>We introduce LLaMA, a collection of foundation language models.</summary>
    <author><name>Hugo Touvron</name></author>
    <author><name>Thibaut Lavril</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2302.13971v1" rel="related"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""

RSS_FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
  <item><title>New model</title><link>https://example.com/a</link>
    <pubDate>Tue, 03 Jun 2025 10:00:00 GMT</pubDate>
    <description>short</description><content:encoded>&lt;p&gt;full&lt;/p&gt;</content:encoded></item>
  <item><title>No link</title></item>
</channel></rss>"""

ATOM_FEED = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>Post</title><link rel="alternate" href="https://example.com/b"/>
    <published>2025-06-01T00:00:00Z</published><summary>sum</summary></entry>
</feed>"""


def test_doc_id_is_stable_hash_of_url():
    assert doc_id("https://arxiv.org/abs/1706.03762") == doc_id(" https://arxiv.org/abs/1706.03762 ")
    assert len(doc_id("x")) == 24 and doc_id("a") != doc_id("b")


def test_arxiv_parse_canonical_url_and_metadata():
    [doc] = arxiv.parse_feed(ARXIV_FEED)
    assert doc.url == "https://arxiv.org/abs/2302.13971"          # version stripped
    assert doc.id == doc_id(doc.url)
    assert doc.title == "LLaMA: Open and Efficient Foundation Language Models"
    assert doc.metadata["authors"] == ["Hugo Touvron", "Thibaut Lavril"]
    assert doc.metadata["primary_category"] == "cs.CL"
    assert "<summary>" in doc.raw_content or "summary>" in doc.raw_content


def test_rss_and_atom_parsing():
    items = rss.parse_feed(RSS_FEED)
    assert [i["link"] for i in items] == ["https://example.com/a"]  # item without link dropped
    assert items[0]["content"] == "<p>full</p>"                     # content:encoded preferred
    assert items[0]["published"].startswith("2025-06-03T10:00:00")
    [entry] = rss.parse_feed(ATOM_FEED)
    assert entry["link"] == "https://example.com/b" and entry["content"] == "sum"


def test_robots_enforced_for_web_not_api():
    http = PoliteSession({"defaults": {"min_interval_s": 0}})
    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /private/", "Crawl-delay: 7"])
    http._robots["example.com"] = rp
    assert http.allowed("https://example.com/public/page")
    assert not http.allowed("https://example.com/private/page")
    assert http._interval("example.com") == 7                      # crawl-delay honoured


class FakeFS:
    def __init__(self):
        self.files = {}

    def list_dir(self, path):
        path = path.rstrip("/") + "/"
        names = {p[len(path):].split("/")[0] for p in self.files if p.startswith(path)}
        return [{"pathSuffix": n, "type": "FILE" if path + n in self.files else "DIRECTORY"}
                for n in sorted(names)]

    def write(self, path, data):
        self.files[path] = data


def test_store_is_idempotent_across_runs():
    fs = FakeFS()
    doc = BronzeDocument.create(source="rss", url="https://example.com/a", raw_content="x",
                                content_type="text/html", retrieved_at="2026-10-28T10:00:00+00:00")
    first = BronzeStore(fs)
    first.load_index()
    assert first.save(doc) is True
    assert list(fs.files) == [f"/datalake/bronze/rss/2026-10-28/{doc.id}.json"]

    second = BronzeStore(fs)                     # new run, new day: still skipped
    assert second.load_index() == 1
    later = doc.model_copy(update={"retrieved_at": "2026-10-29T09:00:00+00:00"})
    assert second.save(later) is False and len(fs.files) == 1
