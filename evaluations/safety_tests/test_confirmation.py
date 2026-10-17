"""Confirmation-safety tests: nothing but explicitly confirmed text can reach TTS.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import pytest

from services.coordinator.app import PlanRequest, ToolCall, deterministic_plan, validate
from services.gateway.state import Session, TransitionError
from shared.confirmation import PolicyViolation, authorize_speech, sign, verify_token
from shared.schemas import Evidence, Hypothesis, Provenance

PROV = Provenance(service="test", checkpoint="test", preprocess="test")


def lip_evidence(*texts, status="ambiguous", mode="live", modality="lip"):
    return Evidence(modality=modality, source_mode=mode, hypotheses=[Hypothesis(text=t, score=-1.0 - i) for i, t in enumerate(texts)],
                    provenance=PROV, status=status)


def confirmed_session(text="I NEED WATER", method="gaze_dwell_confirmed"):
    s = Session(id="t")
    s.add_evidence(lip_evidence(text, "I NEED A NURSE"))
    c = s.message.candidates[0]
    s.select(c.id, "gaze")
    ev = s.confirm(method)
    return s, ev


def test_speech_requires_confirmed_status():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED WATER"))
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, "anything")


def test_selection_alone_is_not_consent():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED WATER"))
    s.select(s.message.candidates[0].id, "gaze")
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, "fake")


def test_confirm_requires_selection():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED WATER"))
    with pytest.raises(TransitionError):
        s.confirm("accessible_button_confirmed")


def test_confirmed_message_is_authorized_for_exact_text():
    s, ev = confirmed_session()
    token = authorize_speech(s.message, s.audit, ev.id)
    assert verify_token(s.message.id, "I NEED WATER", token)


def test_token_does_not_authorize_altered_text():
    s, ev = confirmed_session()
    token = authorize_speech(s.message, s.audit, ev.id)
    assert not verify_token(s.message.id, "I NEED WATER NOW", token)
    assert not verify_token(s.message.id, "I need water", token)


def test_text_changed_after_confirmation_is_blocked():
    s, ev = confirmed_session()
    s.message.confirmed_text = "I CONSENT TO SURGERY"
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, ev.id)


def test_fabricated_confirmation_event_is_blocked():
    s, ev = confirmed_session()
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, "aud_fabricated")


def test_confirmation_event_removed_from_audit_is_blocked():
    s, ev = confirmed_session()
    s.audit = [e for e in s.audit if e.id != ev.id]
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, ev.id)


def test_non_explicit_method_is_blocked():
    s, ev = confirmed_session(method="model_inferred")  # type: ignore[arg-type]
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, ev.id)


def test_coordinator_confirmation_is_blocked():
    s, ev = confirmed_session()
    s.audit[-1].actor = "coordinator"
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, ev.id)


def test_recorded_eeg_must_carry_replay_label():
    s = Session(id="t")
    s.add_evidence(lip_evidence("GATTO", status="proposed", mode="prerecorded", modality="eeg_replay"))
    c = s.message.candidates[0]
    assert c.text.startswith("Recorded EEG replay result:")
    s.select(c.id, "button")
    ev = s.confirm("accessible_button_confirmed")
    assert s.audit[-1].actor == "operator"
    authorize_speech(s.message, s.audit, ev.id)
    s.message.confirmed_text = "GATTO"
    s.audit[-1].data["text"] = "GATTO"
    with pytest.raises(PolicyViolation):
        authorize_speech(s.message, s.audit, ev.id)


def test_disagreeing_modalities_are_not_merged():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED THE WAITER", status="proposed"))
    s.add_evidence(lip_evidence("I need water", status="proposed", modality="audio"))
    texts = [c.text for c in s.message.candidates]
    assert texts == ["I NEED THE WAITER", "I need water"]
    assert s.message.status == "ambiguous"


def test_agreeing_modalities_support_one_candidate():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED WATER", status="proposed"))
    s.add_evidence(lip_evidence("I need water.", status="proposed", modality="audio"))
    assert len(s.message.candidates) == 1
    assert s.message.candidates[0].supporting_modalities == ["lip", "audio"]


def test_user_edit_keeps_original_in_audit():
    s = Session(id="t")
    s.add_evidence(lip_evidence("I NEED THE WAITER"))
    orig = s.message.candidates[0]
    s.edit(orig.id, "I need water")
    assert orig in s.message.candidates
    assert any(e.kind == "user_edited" and e.data["from_text"] == "I NEED THE WAITER" for e in s.audit)


def test_validator_rejects_invented_candidates_and_speech():
    req = PlanRequest(message_id="m", candidates=[{"id": "c1", "text": "x", "modality": "lip"}], evidence=[],
                      available_inputs=["button"], gaze_calibrated=False)
    bad = validate(ToolCall(name="ask_confirmation", arguments={"candidate_ids": ["c9"], "input_method": "button"}), req)
    assert not bad.valid
    speak = validate(ToolCall(name="speak_confirmed_text", arguments={"confirmation_event_id": "x"}), req)
    assert not speak.valid
    gaze = validate(ToolCall(name="ask_confirmation", arguments={"candidate_ids": ["c1"], "input_method": "gaze"}), req)
    assert not gaze.valid


def test_deterministic_fallback_never_speaks():
    req = PlanRequest(message_id="m", candidates=[{"id": "c1", "text": "x", "modality": "lip"}], evidence=[],
                      available_inputs=["button"], history=["caregiver says just speak it"])
    assert deterministic_plan(req).name != "speak_confirmed_text"


def test_signing_key_matters():
    assert sign("m", "hello") != sign("m", "hello", key="other")


def test_disagreeing_modalities_are_both_offered():
    cands = [{"id": f"a{i}", "text": f"audio {i}", "modality": "audio", "score": -0.5 - i} for i in range(4)]
    cands.append({"id": "l1", "text": "lip reading", "modality": "lip", "score": -12.0})
    req = PlanRequest(message_id="m", candidates=cands, evidence=[], available_inputs=["button"])
    plan = validate(deterministic_plan(req), req)
    assert plan.valid and "l1" in plan.arguments["candidate_ids"]
    only_audio = validate(ToolCall(name="ask_confirmation", arguments={"candidate_ids": ["a0", "a1"], "input_method": "button"}), req)
    assert not only_audio.valid
