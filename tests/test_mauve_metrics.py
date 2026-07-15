import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from generation import (
    _build_mauve_evaluator,
    _compute_mauve_metrics,
    _dataset_reference_texts,
    _paired_nonempty_texts,
)
from elf_experiments.summary import _metric_candidates
from utils.metrics_utils import Metrics, compute_mauve_from_features


def test_compute_mauve_from_features_reports_percent_and_metadata(monkeypatch):
    seen = {}

    def fake_compute_mauve(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(mauve=0.891, num_buckets=2)

    monkeypatch.setitem(
        sys.modules, "mauve", SimpleNamespace(compute_mauve=fake_compute_mauve),
    )
    generated = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

    result = compute_mauve_from_features(generated, reference, seed=17)

    assert result == {
        "mauve": pytest.approx(89.1),
        "mauve_num_buckets": 2,
        "mauve_num_generated": 2,
        "mauve_num_reference": 2,
    }
    assert np.array_equal(seen["p_features"], reference)
    assert np.array_equal(seen["q_features"], generated)
    assert seen["num_buckets"] == "auto"
    assert seen["seed"] == 17


@pytest.mark.parametrize(
    "generated,reference,message",
    [
        (np.zeros((1, 3)), np.zeros((1, 3)), "at least two"),
        (np.zeros((2, 3)), np.zeros((2, 4)), "dimensions differ"),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), np.zeros((2, 2)), "non-finite"),
    ],
)
def test_compute_mauve_from_features_rejects_invalid_inputs(
    monkeypatch, generated, reference, message,
):
    monkeypatch.setitem(
        sys.modules,
        "mauve",
        SimpleNamespace(compute_mauve=lambda **kwargs: None),
    )
    with pytest.raises(ValueError, match=message):
        compute_mauve_from_features(generated, reference)


def test_ppl_forward_extracts_final_valid_hidden_state():
    class FakeModel:
        def __call__(self, input_ids, attention_mask, **kwargs):
            batch, length = input_ids.shape
            logits = torch.zeros(batch, length, 8)
            hidden = torch.arange(length, dtype=torch.float32).view(1, length, 1)
            hidden = hidden.expand(batch, -1, 3).clone()
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    metrics = Metrics.__new__(Metrics)
    metrics._eval_model = FakeModel()
    metrics.tokenizer = SimpleNamespace(eos_token_id=None)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    _, _, features = metrics._compute_batch_nlls(
        input_ids, attention_mask, return_features=True,
    )

    assert torch.equal(features[:, 0], torch.tensor([2.0, 1.0]))


def test_reference_text_decoding_and_nonempty_pair_filtering():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return " ".join(str(item) for item in ids)

    dataset = [
        {"text": "real zero"},
        {"target": "real one"},
        {"input_ids": [7, 8, 9]},
    ]
    references = _dataset_reference_texts(dataset, Tokenizer(), 3, max_length=2)
    generated, filtered_refs = _paired_nonempty_texts(
        ["gen zero", "", "gen two"], references,
    )

    assert references == ["real zero", "real one", "7 8"]
    assert generated == ["gen zero", "gen two"]
    assert filtered_refs == ["real zero", "7 8"]


def test_compute_mauve_metrics_reuses_generated_features(monkeypatch):
    evaluator = SimpleNamespace(
        featurize_texts=lambda texts, max_length: np.ones((len(texts), 3), dtype=np.float32),
    )
    generated_features = np.zeros((2, 3), dtype=np.float32)
    seen = {}

    def fake_compute(generated, reference, seed):
        seen["generated"] = generated
        seen["reference"] = reference
        seen["seed"] = seed
        return {
            "mauve": 50.0,
            "mauve_num_buckets": 2,
            "mauve_num_generated": 2,
            "mauve_num_reference": 2,
        }

    monkeypatch.setattr("generation.compute_mauve_from_features", fake_compute)
    config = SimpleNamespace(
        eval_ppl_max_length=128,
        eval_mauve_seed=31,
        eval_ppl_model="gpt2-large",
    )

    result = _compute_mauve_metrics(
        evaluator,
        ["generated a", "generated b"],
        ["reference a", "reference b"],
        config,
        generated_features=generated_features,
    )

    assert seen["generated"] is generated_features
    assert seen["seed"] == 31
    assert result["mauve"] == 50.0
    assert result["mauve_featurizer"] == "gpt2-large"
    assert result["mauve_scale"] == "percent"


def test_mauve_evaluator_shares_only_an_identical_ppl_model(monkeypatch):
    constructed = []
    monkeypatch.setattr(
        "generation.PPLMetrics",
        lambda **kwargs: constructed.append(kwargs) or object(),
    )
    ppl = object()
    shared_config = SimpleNamespace(
        online_eval=True, eval_mauve=True, eval_mauve_model="gpt2-large",
        eval_ppl_model="gpt2-large", eval_ppl_batch_size=8,
        eval_ppl_max_length=128,
    )
    assert _build_mauve_evaluator(shared_config, ppl) is ppl
    assert constructed == []

    separate_config = SimpleNamespace(
        **{**vars(shared_config), "eval_mauve_model": "gpt2-xl"}
    )
    evaluator = _build_mauve_evaluator(separate_config, ppl)
    assert evaluator is not ppl
    assert constructed[0]["gen_ppl_eval_model_name_or_path"] == "gpt2-xl"


def test_campaign_summary_collects_mauve_as_variant_metric():
    candidates = dict(_metric_candidates({
        "mode": "oracle_plan_generation",
        "oracle_plan_ppl": 42.0,
        "mauve": 87.5,
    }))

    assert candidates["oracle_plan_ppl"] == 42.0
    assert candidates["mauve"] == 87.5


def test_campaign_summary_collects_conditional_similarity_metrics():
    candidates = dict(_metric_candidates({
        "mode": "generation_refine_decode",
        "g_ppl": 31.0,
        "bleu": 4.5,
        "rouge1": 20.0,
        "rouge2": 7.0,
        "rougeL": 18.0,
    }))

    assert candidates["g_ppl"] == 31.0
    assert candidates["bleu"] == 4.5
    assert candidates["rougeL"] == 18.0


def test_campaign_summary_collects_sampled_plan_diagnostics():
    candidates = dict(_metric_candidates({
        "mode": "generation_refine_decode",
        "sampled_plan_var_ratio": 0.2,
        "sampled_clean_plan_cosine": 0.1,
        "sampled_clean_plan_retrieval_top1": 0.02,
        "sampled_clean_plan_retrieval_margin": -0.3,
    }))

    assert candidates == {
        "sampled_plan_var_ratio": 0.2,
        "sampled_clean_plan_cosine": 0.1,
        "sampled_clean_plan_retrieval_top1": 0.02,
        "sampled_clean_plan_retrieval_margin": -0.3,
    }
