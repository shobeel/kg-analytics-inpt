import argparse
import json
import logging
import sys

from .config import settings

parser = argparse.ArgumentParser(prog="extraction")
sub = parser.add_subparsers(dest="cmd", required=True)
sub.add_parser("test-call", help="one tiny call to EXTRACTION_MODEL")
p = sub.add_parser("extract", help="chunks -> entities/relations (silver)")
p.add_argument("--limit", type=int, help="pilot: only N chunks spread over sources")
p.add_argument("--show-json", action="store_true", help="print each chunk's validated JSON")
p.add_argument("--no-write", action="store_true", help="do not rewrite silver tables")
p.add_argument("--cached-only", action="store_true",
               help="no LLM calls: rebuild silver from the cache with the current validation rules")
sub.add_parser("resolve", help="entity resolution: KG_raw and KG_resolved (silver)")
sub.add_parser("status", help="extraction coverage + flag rates from cache vs silver chunks (no LLM)")
sub.add_parser("relation-stats", help="relation types, schema violations, proposed tweaks (no LLM)")
p = sub.add_parser("sample-gold", help="sample chunks into eval/gold/extraction for annotation")
p.add_argument("--n", type=int, default=25)
p.add_argument("--out", default="/app/eval/gold/extraction/annotation_batch.jsonl")
p = sub.add_parser("prefill-gold", help="draft gold annotations with ANSWER_MODEL (to be corrected by hand)")
p.add_argument("--gold", default="/app/eval/gold/extraction/annotation_batch.jsonl")
p.add_argument("--out", default="/app/eval/gold/extraction/prefill.jsonl")
p = sub.add_parser("export-predictions", help="dump cached predictions for the chunks in a gold file")
p.add_argument("--gold", required=True)
p.add_argument("--out", required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "openai", "py4j", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
s = settings()

if args.cmd == "test-call":
    from .llm import LLMClient
    raw, usage = LLMClient(s).chat_json(
        system="Return JSON.", user="Which organization developed the Transformer architecture?",
        schema={"type": "object", "properties": {"organization": {"type": "string"}}, "required": ["organization"]},
        schema_name="test", purpose="test-call")
    print(json.dumps({"model": s.extraction_model, "response": json.loads(raw), "usage": usage}, indent=2))
elif args.cmd == "extract":
    from .extract import run
    summary = run(s, limit=args.limit, show_json=args.show_json, write=not args.no_write,
                  cached_only=args.cached_only)
    print("SUMMARY " + json.dumps(summary, indent=2))
    with open("/app/logs/extraction_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if summary["quota_exhausted"]:
        print(f"\nSTOPPED: daily Gemini quota reached. {summary['chunks_remaining']} chunks remain; "
              "results so far are cached and written. Re-run `make extract` after the quota resets.",
              file=sys.stderr)
        sys.exit(3)
elif args.cmd == "resolve":
    from .er import run_resolution
    summary = run_resolution(s)
    print("SUMMARY " + json.dumps(summary, indent=2))
    with open("/app/logs/er_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
elif args.cmd == "status":
    from .extract import status
    print("STATUS " + json.dumps(status(s), indent=2))
elif args.cmd == "relation-stats":
    from .extract import current_records
    from .relstats import relation_stats
    _, recs = current_records(s)
    stats = relation_stats(recs)
    print("RELATION_STATS " + json.dumps(stats, indent=2))
    with open("/app/logs/relation_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
elif args.cmd == "sample-gold":
    from .gold import sample_gold
    print(json.dumps(sample_gold(s, args.n, args.out), indent=2))
elif args.cmd == "prefill-gold":
    from .gold import prefill_gold
    print(json.dumps(prefill_gold(s, args.gold, args.out), indent=2))
elif args.cmd == "export-predictions":
    from .gold import export_predictions
    print(json.dumps(export_predictions(s, args.gold, args.out), indent=2))
