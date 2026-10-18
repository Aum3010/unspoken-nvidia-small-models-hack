"""Optional validator rules (UNSPOKEN_VALIDATOR_NEGATION, UNSPOKEN_VALIDATOR_IGNORE_HISTORY): off they change
nothing; on they reject a correction that flips a negation, and ignore_input when the person is clearly trying.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import pytest

import services.coordinator.app as coord
from services.coordinator.app import CandidateSummary, EvidenceSummary, PlanRequest, ToolCall, deterministic_plan, validate


@pytest.fixture
def flags(monkeypatch):
    def set_flags(negation=False, ignore_history=False):
        monkeypatch.setattr(coord, "VALIDATOR_NEGATION", negation)
        monkeypatch.setattr(coord, "VALIDATOR_IGNORE_HISTORY", ignore_history)
    return set_flags


def req(text="I don't need water", status="ambiguous", flags=(), history=(), modality="audio"):
    return PlanRequest(message_id="m", available_inputs=["button", "keyboard"], history=list(history),
                       candidates=[CandidateSummary(id="c1", text=text, modality=modality, score=-1.0, supporting_modalities=[modality])],
                       evidence=[EvidenceSummary(modality=modality, status=status, quality_flags=list(flags), n_hypotheses=1)])


def correction(text):
    return ToolCall(name="propose_correction", by="llm", rationale="x",
                    arguments={"source_candidate_ids": ["c1"], "text": text, "input_method": "button"})


def ignore():
    return ToolCall(name="ignore_input", by="llm", rationale="x", arguments={"reason": "noise"})


@pytest.mark.parametrize("negation_flag", [False, True])
def test_dropping_a_negation(flags, negation_flag):
    flags(negation=negation_flag)
    call = validate(correction("I need water"), req("I don't need water"))
    assert call.valid is (not negation_flag)


@pytest.mark.parametrize("source, fixed", [("I don't need wader", "I don't need water"), ("I need wader", "I need water"),
                                           ("I can't breeve", "I can't breathe")])
def test_corrections_that_keep_the_negation_pass(flags, source, fixed):
    flags(negation=True)
    assert validate(correction(fixed), req(source)).valid


def test_adding_a_negation_is_rejected(flags):
    flags(negation=True)
    assert not validate(correction("I don't need water"), req("I need wader")).valid


@pytest.mark.parametrize("history", [["retake", "retake"], ["none_of_these"], ["x", "none_of_these", "y"]])
def test_ignore_after_failed_tries(flags, history):
    flags(ignore_history=False)
    assert validate(ignore(), req("i uh", history=history)).valid          # today's behaviour
    flags(ignore_history=True)
    bad = validate(ignore(), req("i uh", history=history))
    assert not bad.valid
    fallback = validate(deterministic_plan(req("i uh", history=history)), req("i uh", history=history))
    assert fallback.valid and fallback.name in ("switch_input_method", "ask_confirmation")


def test_ignore_of_an_accepted_clean_reading(flags):
    flags(ignore_history=True)
    assert not validate(ignore(), req("I need water", status="proposed")).valid


def test_noise_can_still_be_ignored(flags):
    flags(ignore_history=True)
    assert validate(ignore(), req("mm", status="ambiguous", flags=["low_word_confidence"])).valid
    assert validate(ignore(), req("i need", status="proposed", flags=["low_mouth_motion"], modality="lip")).valid
