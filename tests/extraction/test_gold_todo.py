import json

import pytest

from eval.gold_todo import ENTS, FormatError, convert, diff, make, parse, render

CHUNK = {"chunk_id": "abc_0001", "doc_id": "abc", "source": "rss", "title": "T", "url": "u",
         "text": "OpenAI developed GPT-4. Sam Altman leads OpenAI."}
PRE = {"chunk_id": "abc_0001", "model": "m", "status": "ok",
       "entities": [{"name": "OpenAI", "type": "Organization"}, {"name": "GPT-4", "type": "Product"},
                    {"name": "Sam Altman", "type": "Person"}],
       "relations": [{"source": "GPT-4", "relation": "DEVELOPED_BY", "target": "OpenAI",
                      "evidence_text": "OpenAI developed GPT-4", "schema_valid": True, "evidence_verified": True},
                     {"source": "Sam Altman", "relation": "WORKS_AT", "target": "OpenAI",
                      "evidence_text": "made up", "schema_valid": True, "evidence_verified": False}]}


def test_render_parse_roundtrip_keeps_prefill():
    g = parse(render(1, 1, CHUNK, PRE), "f")
    assert g["status"] == "todo"
    assert g["entities"] == [{k: e[k] for k in ("name", "type")} for e in PRE["entities"]]
    assert [r["relation"] for r in g["relations"]] == ["DEVELOPED_BY", "WORKS_AT"]
    assert "evidence not in text" in render(1, 1, CHUNK, PRE)


def test_parse_rejects_bad_lines_with_location():
    bad = render(1, 1, CHUNK, PRE).replace("Organization | OpenAI", "Company | OpenAI")
    with pytest.raises(FormatError, match=r"f:\d+: unknown entity type"):
        parse(bad, "f")
    dangling = render(1, 1, CHUNK, PRE).replace("Person | Sam Altman", "")
    with pytest.raises(FormatError, match="not listed under ENTITIES"):
        parse(dangling, "f")


def test_diff_reports_retype_removal_and_addition():
    text = (render(1, 1, CHUNK, PRE).replace("status: todo", "status: done")
            .replace("Product | GPT-4", "AIModel | GPT-4")
            .replace("Sam Altman | WORKS_AT | OpenAI", "")
            .replace(ENTS, ENTS + "\nPerson | Ilya Sutskever"))
    d = diff(PRE, parse(text, "f"))
    assert d["entities"]["retyped"] == [{"name": "GPT-4", "from": "Product", "to": "AIModel"}]
    assert d["entities"]["kept"] == 2 and [e["name"] for e in d["entities"]["added"]] == ["Ilya Sutskever"]
    assert d["relations"]["removed"] == [{"source": "Sam Altman", "relation": "WORKS_AT", "target": "OpenAI"}]
    assert d["relations"]["kept"] == 1 and d["relations"]["added"] == []


def test_make_never_overwrites_and_convert_writes_scorer_input(tmp_path):
    (tmp_path / "annotation_batch.jsonl").write_text(json.dumps(CHUNK) + "\n")
    (tmp_path / "prefill.jsonl").write_text(json.dumps(PRE) + "\n")
    make(tmp_path)
    f = next((tmp_path / "todo").glob("*.txt"))
    f.write_text(f.read_text().replace("status: todo", "status: done"))
    make(tmp_path)                                           # must keep the edited file
    assert "status: done" in f.read_text()
    assert convert(tmp_path) == 0
    row = json.loads((tmp_path / "gold.jsonl").read_text())
    assert row["status"] == "done" and row["text"] == CHUNK["text"] and len(row["relations"]) == 2
    ch = json.loads((tmp_path / "changes.jsonl").read_text())
    assert ch["entities"]["kept"] == 3 and ch["relations"]["removed"] == []


def test_make_waits_for_missing_prefill(tmp_path):
    other = {**CHUNK, "chunk_id": "def_0002"}
    (tmp_path / "annotation_batch.jsonl").write_text(json.dumps(CHUNK) + "\n" + json.dumps(other) + "\n")
    (tmp_path / "prefill.jsonl").write_text(json.dumps(PRE) + "\n")
    make(tmp_path)
    assert [f.name for f in (tmp_path / "todo").glob("*.txt")] == ["01_abc_0001.txt"]
    make(tmp_path, empty=True)
    assert len(list((tmp_path / "todo").glob("*.txt"))) == 2
