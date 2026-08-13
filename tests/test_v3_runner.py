"""Wiring tests for the v3 condition.

Covers the experiment-runner path without model weights: the runner must reach
``LazyGraphConstraint``, skip the DFS, and emit a record in the same shape as
the baseline's so ``compute_hits`` scores both conditions identically.
"""

import experiment
import main
from utils import PATH_END, PATH_START


class StubBuilder:
    PATH_START_TOKEN = PATH_START
    PATH_END_TOKEN = PATH_END

    def process_input(self, data, return_tire=False):
        return f"Question: {data['question']}", [], None


class StubModel:
    """Decodes greedily through whatever constraint it is handed."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.seen_constraint = None

    def prepare_model_prompt(self, query):
        return query

    def generate_sentence(self, llm_input, trie, **kwargs):
        self.seen_constraint = trie
        end_id = self.tokenizer.convert_tokens_to_ids(PATH_END)
        ids = self.tokenizer(PATH_START, add_special_tokens=False).input_ids
        for _ in range(200):
            allowed = trie.get(ids)
            if not allowed:
                break
            ids.append(allowed[0])
            if ids[-1] == end_id:
                break
        return self.tokenizer.decode(ids) + " # Answer: stub"


def test_every_method_maps_to_registered_conditions():
    registered = {
        "GCR_Baseline", "DCA_v1_Static", "DCA_v2_Dynamic", "DCA_v2_NoGates",
        "DCA_v3_Lazy", "DCA_v3_NoGates",
    }
    for method, conditions in main.CONDITIONS_BY_METHOD.items():
        assert set(conditions) <= registered, method


def test_run_v3_produces_a_baseline_shaped_record(tokenizer, webqsp):
    from lazy_constraint import LazyGraphConstraint

    question = webqsp[0]
    model = StubModel(tokenizer)
    prep = experiment.PrepCache.build(question, 2, enumerate_paths=False)
    assert prep.all_paths == [], "v3 must not pay for the DFS"

    result, trie_ok = experiment._run_v3(
        model, StubBuilder(), question, question["id"], "DCA_v3_Lazy", prep,
        index_len=2, max_new_tokens=256, beam_size=5,
    )

    assert trie_ok
    assert isinstance(model.seen_constraint, LazyGraphConstraint)
    assert result["mode"] == "DCA_v3_Lazy"
    assert result["prediction"].startswith(PATH_START)
    assert result["lazy_stats"]["frontier_builds"] > 0

    # Same keys the baseline emits, so compute_hits treats them alike.
    for key in ("id", "question", "prediction", "ground_truth", "mode"):
        assert key in result


def test_nogates_condition_disables_the_gates(tokenizer, webqsp):
    question = webqsp[0]
    model = StubModel(tokenizer)
    prep = experiment.PrepCache.build(question, 2, enumerate_paths=False)

    experiment._run_v3(
        model, StubBuilder(), question, question["id"], "DCA_v3_NoGates", prep,
        index_len=2, max_new_tokens=256, beam_size=5, gates_enabled=False,
    )
    assert model.seen_constraint.gates_enabled is False


def test_hits_evaluation_accepts_a_v3_record(tokenizer, webqsp):
    question = webqsp[0]
    model = StubModel(tokenizer)
    prep = experiment.PrepCache.build(question, 2, enumerate_paths=False)
    result, _ = experiment._run_v3(
        model, StubBuilder(), question, question["id"], "DCA_v3_Lazy", prep,
        index_len=2, max_new_tokens=256, beam_size=5,
    )
    assert experiment.compute_hits([result]) in (0, 1)
