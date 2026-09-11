"""Prospective replication source exclusions and fixed-model scoring checks."""

from __future__ import annotations

import numpy as np

from eval.history_audit import AuditConfig, MODELS, design_matrix, prepare_trials
from eval.ibl_replication import ReplicationRules, convert_source, score_cohort
from tests.test_history_audit import _trials
from tests.test_ibl_source_audit import _source


def test_replication_omissions_and_source_qc_are_explicit() -> None:
    _, arrays = _source()
    rows, qc = convert_source(arrays, "source", ReplicationRules())
    assert not qc["eligible"]  # Three trials cannot pass the frozen minimum.
    assert rows[1]["action"] == 2 and rows[1]["rt_ms"] is None
    assert rows[2]["prev"]["action"] == 2
    arrays["contrastRight"][1] = 0.5
    rows, qc = convert_source(arrays, "source", ReplicationRules())
    assert [r["trial_index"] for r in rows] == [0, 2]
    assert rows[1]["prev"] is None
    assert qc["dropped_contrast"] == 1


def test_identical_frozen_models_have_zero_gain_and_cannot_pass() -> None:
    df = _trials()
    config = AuditConfig()
    data = prepare_trials(df, config)
    models = {}
    for model in MODELS:
        x, features = design_matrix(data, model)
        models[model] = {"features": features, "coefficients": [0.0] * x.shape[1]}
    result = score_cohort(df, models, {sid: sid for sid in ("a", "b", "c", "d")},
                          config, ReplicationRules(min_eligible_subjects=4))
    assert result["primary_equal_subject_nll_gain"] == 0
    assert result["subject_bootstrap_95_interval"] == [0, 0]
    assert not result["criterion_met"]
    assert np.isclose(result["pooled_nll"]["evidence_history"], np.log(2))
    insufficient = score_cohort(df, models, {sid: sid for sid in ("a", "b", "c", "d")},
                                config, ReplicationRules())
    assert insufficient["status"] == "insufficient_eligible_subjects"
