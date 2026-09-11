"""Metadata-only cohort selection must exclude known animals and task variants."""

from __future__ import annotations

from scripts.prepare_ibl_replication import select_cohort


def test_cohort_excludes_known_subjects_and_opto_variants_without_outcome_selection() -> None:
    sessions = [
        {"id": "a", "subject": "known", "lab": "lab", "start_time": "2020-01-01", "task_protocol": "_iblrig_tasks_biasedChoiceWorld5.0.0"},
        {"id": "b", "subject": "new", "lab": "lab", "start_time": "2020-01-01", "task_protocol": "_iblrig_tasks_biasedChoiceWorld5.0.0"},
        {"id": "c", "subject": "new", "lab": "lab", "start_time": "2020-01-02", "task_protocol": "_iblrig_tasks_biasedChoiceWorld5.0.0"},
        {"id": "d", "subject": "opto", "lab": "lab", "start_time": "2020-01-03", "task_protocol": "optoBiasedChoiceWorld5.0.0"},
    ]
    result = select_cohort(sessions, {"known"}, 60, 42)
    assert [s["id"] for s in result] == ["c"]
    assert result == select_cohort(list(reversed(sessions)), {"known"}, 60, 42)
    # Availability, not success/performance, determines selection.
    for session in sessions:
        session["performance"] = "ignored"
    assert [s["id"] for s in select_cohort(sessions, {"known"}, 60, 42)] == ["c"]
