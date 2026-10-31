from eval.extraction_scorer import names_match, relaxed_name, score

TOY = "/app/eval/gold/extraction/toy"


def test_relaxed_normalization():
    assert relaxed_name("The Google DeepMind Ltd.") == "google deepmind"
    assert names_match("Altman", "Sam Altman", "relaxed") and not names_match("Altman", "Sam Altman", "strict")
    assert names_match("openai", " OpenAI ", "strict")


def test_toy_scores_match_hand_computation():
    r = score(f"{TOY}/gold.jsonl", f"{TOY}/pred.jsonl")
    assert r["chunks_scored"] == 3
    # entities strict: TP 5 (OpenAI, GPT-4, London, Anthropic, OpenAI), FP 4, FN 3
    assert (r["entity_strict"]["tp"], r["entity_strict"]["fp"], r["entity_strict"]["fn"]) == (5, 4, 3)
    assert r["entity_strict"]["f1"] == round(10 / 17, 4)
    # relaxed adds Altman~Sam Altman, "Google DeepMind Ltd", AlphaFold despite wrong type
    assert (r["entity_relaxed"]["tp"], r["entity_relaxed"]["fp"], r["entity_relaxed"]["fn"]) == (8, 1, 0)
    assert r["entity_relaxed"]["recall"] == 1.0
    # relations strict: GPT-4 DEVELOPED_BY OpenAI + symmetric COMPETES_WITH reversed
    assert (r["relation_strict"]["tp"], r["relation_strict"]["fp"], r["relation_strict"]["fn"]) == (2, 3, 3)
    # relaxed: + Altman WORKS_AT, + DEVELOPED_BY "Google DeepMind Ltd"; reversed LOCATED_IN stays wrong
    assert (r["relation_relaxed"]["tp"], r["relation_relaxed"]["fp"], r["relation_relaxed"]["fn"]) == (4, 1, 1)
    assert r["relation_relaxed"]["f1"] == 0.8
