"""Independent checks for causal latent-state comparison."""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from eval.generative_history import (Fit, _objective, forward_backward, prepare, predict)
from scripts.diagnose_bias_history import simulate


def test_forward_matches_enumeration() -> None:
    emission = np.array([[.2,.8],[.3,.7],[.9,.1]])
    transition = np.array([[.9,.1],[.2,.8]])
    loss, gamma, count, prior = forward_backward(emission, transition, np.array([True,False,False]))
    weights = []
    paths = list(product(range(2), repeat=3))
    for path in paths:
        weights.append(.5*emission[0,path[0]]*transition[path[0],path[1]]*emission[1,path[1]]*
                       transition[path[1],path[2]]*emission[2,path[2]])
    weights = np.asarray(weights)
    assert np.isclose(loss, -np.log(weights.sum()))
    expected = np.zeros_like(gamma)
    expected_count = np.zeros_like(count)
    for path, weight in zip(paths, weights/weights.sum()):
        for t, state in enumerate(path):
            expected[t,state] += weight
        for t in range(1,3):
            expected_count[path[t-1],path[t]] += weight
    np.testing.assert_allclose(gamma, expected)
    np.testing.assert_allclose(count, expected_count)
    np.testing.assert_allclose(prior[0], [.5,.5])


def test_analytic_gradient_matches_numerical() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20,3))
    y = rng.integers(2,size=20).astype(float)
    reset = np.zeros(20,dtype=bool)
    reset[[0,10]] = True
    for model, k in [("switching",3),("drift",5),("history",1)]:
        offsets = np.linspace(-1,1,k)
        a = .9*np.eye(k)+.1/k
        theta = rng.normal(0,.2,size=(k*3+k*k if model == "switching" else 3))
        args = (x,y,reset,model,offsets,a)
        numerical = approx_derivative(lambda p: _objective(p,*args)[0],theta).ravel()
        np.testing.assert_allclose(_objective(theta,*args)[1],numerical,atol=1e-7,rtol=1e-5)


def test_future_choices_do_not_change_past_predictions() -> None:
    raw = simulate("mixed",715).iloc[:500].copy()
    seq = prepare(raw)
    beta = np.zeros((3, seq.x.shape[1]+12))
    beta[:,0] = [-1,0,1]
    beta[:,-1] = .3
    fit = Fit("switching_history", beta, .95*np.eye(3)+.05/3, np.zeros(3),0,True,0,0,[])
    original = predict(seq,fit)
    raw.loc[250:,"action"] = np.where(raw.loc[250:,"action"] == "left","right","left")
    raw.loc[250:,"correct"] = ~raw.loc[250:,"correct"]
    changed = predict(prepare(raw),fit)
    np.testing.assert_array_equal(original[:251],changed[:251])
    assert not np.allclose(original[251:],changed[251:])


def test_gap_omission_and_session_reset() -> None:
    raw = simulate("mixed",715).iloc[:1000].copy()
    raw.loc[150,"action"] = "omit"
    raw.loc[300:,"trial_index"] += 1
    seq = prepare(raw)
    for session, trial in [("s0",151),("s0",301),("s1",1)]:
        idx = np.flatnonzero((seq.rows.session_id == session)&(seq.rows.trial_index == trial))[0]
        assert seq.reset[idx]
        assert not seq.h[idx].any()
        assert not seq.score[idx]


def test_training_independent_of_external_responses(monkeypatch) -> None:
    import scripts.compare_generative_history as runner
    # Exercise the actual outer-fold path; model fitting must exclude all test responses.
    monkeypatch.setattr(runner,"MODELS",("task","history"))
    raw = simulate("mixed",717)
    raw = raw[raw.subject.isin(["m0","m1","m2","m3"])].copy()
    folds = [["m0","m1"],["m2","m3"]]
    _, first = runner.evaluate(raw,folds,150)
    heldout = raw.subject.isin(folds[0])
    raw.loc[heldout,"action"] = np.where(raw.loc[heldout,"action"]=="left","right","left")
    raw.loc[heldout,"correct"] = ~raw.loc[heldout,"correct"]
    _, second = runner.evaluate(raw,folds,150)
    for a,b in zip(first,second):
        if a["fold"] == 0:
            np.testing.assert_array_equal(a["beta"],b["beta"])
            np.testing.assert_array_equal(a["transition"],b["transition"])


def test_simulator_ignores_original_choices_and_outcomes() -> None:
    from scripts.compare_generative_history import simulate_template
    raw = simulate("mixed",717).iloc[:500].copy()
    first, meta = simulate_template(raw,"mixed",1910,.15)
    raw["action"] = np.where(raw.action=="left","right","left")
    raw["correct"] = ~raw.correct
    second, other = simulate_template(raw,"mixed",1910,.15)
    assert first.action.equals(second.action)
    assert first.correct.equals(second.correct)
    assert meta == other


def test_recovery_gate_cannot_promote_small_or_failed_samples() -> None:
    from scripts.compare_generative_history import recovery_gate
    records = [{"regime":"static","strength":0.,"history_detected":False,"converged":True,
                "oracle_conditional_history_kl":0.} for _ in range(5)]
    assert not recovery_gate(records)['recovery_pass']
    records *= 8
    assert recovery_gate(records)['cells']['static_0.0']['pass']
    records[0]['converged'] = False
    assert not recovery_gate(records)['recovery_pass']
    assert recovery_gate(records)['fresh_cohort'] == 'deferred'
