"""Selection rules for the development-animal deep reference; no network access."""

from __future__ import annotations

import pytest

from scripts.fetch_ibl_deep_reference import protocol_family, select_sessions


def _session(eid: str, subject: str, protocol: str, start: str = "2020-01-01T10:00:00", lab: str = "lab_a") -> dict:
    return {"id": eid, "subject": subject, "lab": lab, "start_time": start, "task_protocol": protocol, "number": 1}


@pytest.mark.parametrize(("protocol", "family"), [
    ("_iblrig_tasks_biasedChoiceWorld6.4.2", "biased"),
    ("_iblrig_tasks_trainingChoiceWorld6.2.5", "training"),
    ("_iblrig_tasks_ephysChoiceWorld6.4.2", "ephys"),
    ("_iblrig_tasks_opto_biasedChoiceWorld6.4.2", None),
    ("nate_optoBiasedChoiceWorld8.22.1+dirty", None),
    ("_iblrig_tasks_biasedChoiceWorldRPE6.4.2", None),
    ("_iblrig_tasks_opto_ephysChoiceWorld6.4.2", None),
    ("_iblrig_tasks_habituationChoiceWorld6.4.2", None),
    ("_iblrig_tasks_biasedChoiceWorld", None),
    ("", None),
    (None, None),
])
def test_protocol_family_accepts_only_exact_standard_versions(protocol: str | None, family: str | None) -> None:
    assert protocol_family(protocol) == family


def test_select_keeps_all_standard_sessions_of_development_animals_in_order() -> None:
    inventory = [
        _session("c", "m2", "_iblrig_tasks_biasedChoiceWorld6.4.2"),
        _session("b", "m1", "_iblrig_tasks_biasedChoiceWorld6.4.2", "2020-02-01T10:00:00"),
        _session("a", "m1", "_iblrig_tasks_trainingChoiceWorld6.2.5", "2020-01-05T10:00:00"),
        _session("d", "m1", "nate_optoBiasedChoiceWorld8.22.1"),
        _session("e", "other", "_iblrig_tasks_biasedChoiceWorld6.4.2"),
        _session("f", "m2", "_iblrig_tasks_ephysChoiceWorld6.4.2", lab=""),
    ]
    selected = select_sessions(inventory, {"m1", "m2"}, {"held"})
    assert [(s["id"], s["family"]) for s in selected] == [("a", "training"), ("b", "biased"), ("c", "biased")]
    assert all("number" not in s for s in selected)


def test_select_deduplicates_identical_rows_and_rejects_conflicts() -> None:
    row = _session("a", "m1", "_iblrig_tasks_biasedChoiceWorld6.4.2")
    assert len(select_sessions([row, dict(row)], {"m1"}, set())) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        select_sessions([row, row | {"lab": "lab_b"}], {"m1"}, set())


def test_select_fails_closed_on_reserved_animals() -> None:
    row = _session("a", "held", "_iblrig_tasks_biasedChoiceWorld6.4.2")
    with pytest.raises(ValueError, match="Reserved"):
        select_sessions([row], {"m1"}, {"held"})
    with pytest.raises(ValueError, match="overlap"):
        select_sessions([], {"m1"}, {"m1"})


def test_variant_sensitivity_split_keeps_only_standard_biased_sessions() -> None:
    from scripts.exclude_variant_sensitivity import split_sessions

    rows = [_session("a", "m1", "_iblrig_tasks_biasedChoiceWorld6.4.2"),
            _session("b", "m2", "nate_optoBiasedChoiceWorld8.16.0"),
            _session("c", "m3", "_iblrig_tasks_biasedChoiceWorldRPE"),
            _session("d", "m4", "_iblrig_tasks_trainingChoiceWorld6.2.5")]
    kept, excluded = split_sessions(rows)
    assert [s["id"] for s in kept] == ["a"]
    assert [s["id"] for s in excluded] == ["b", "c", "d"]
