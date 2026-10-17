"""Canonical, protocol-aware tables shared by BNCI2014-008 and bigP3BCI.

Four layers, kept in separate files so an agent-facing observation can never pick up a hidden label:

  1. raw recordings      -> data/raw/** (original files, never modified) + recordings.csv (pointer + checksum)
  2. selection/stimulus  -> selections.csv, stimulus_events.csv       (agent-safe: no targets, no target flags)
  3. verifier-only       -> verifier_targets.csv, verifier_flash_labels.csv
  4. derived evidence    -> data/evidence/** (future; not produced in this phase)

EEG samples are never copied: a stimulus event is (recording_id, onset_sample); epochs are cut from the raw file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

AGENT_SAFE_TABLES = ("recordings", "participants", "sessions", "runs", "selections", "stimulus_events")
VERIFIER_ONLY_TABLES = ("verifier_targets", "verifier_flash_labels")

COLUMNS = {
    "recordings": ["recording_id", "dataset", "participant_id", "session", "run_id", "path", "sha256", "bytes", "fs_hz",
                   "n_samples", "duration_s", "n_eeg_channels", "eeg_channels", "has_eye_tracking", "equipment"],
    "participants": ["participant_id", "dataset", "sex", "als_flag", "alsfrs_r", "alsfrs_r_by_session", "n_sessions"],
    "sessions": ["session_id", "participant_id", "dataset", "session", "n_runs", "alsfrs_r"],
    "runs": ["run_id", "recording_id", "dataset", "participant_id", "session", "run_type", "condition", "run_index",
             "n_selections", "evidence_complete"],
    "selections": ["selection_id", "run_id", "recording_id", "dataset", "participant_id", "session", "trial_index",
                   "start_sample", "end_sample", "n_flashes", "max_flashes_in_protocol", "evidence_truncated",
                   "min_flashes_per_character", "max_flashes_per_character", "grid_rows", "grid_cols",
                   "complete_repetitions", "flashes_in_partial_repetition", "repetition_rule"],
    "stimulus_events": ["selection_id", "flash_index", "onset_sample", "onset_s", "duration_samples", "chars_shown", "repetition_index"],
    "verifier_targets": ["selection_id", "target_index", "target_char", "online_selected_index", "online_selected_char"],
    "verifier_flash_labels": ["selection_id", "flash_index", "is_target_flash"],
}


@dataclass(frozen=True)
class Protocol:
    name: str
    grid: tuple[int, int]
    flash_kind: str            # "row_col" | "checkerboard_subset"
    soa_samples: int
    flash_samples: int
    fs: int
    max_flashes: int | None    # per selection in a fixed-length run; None if not established
    notes: str = ""


@dataclass
class Issue:
    level: str                 # "ERROR" | "WARN" | "INFO"
    check: str
    where: str
    detail: str


@dataclass
class AuditResult:
    tables: dict[str, list[dict]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)

    def error(self, check: str, where: str, detail: str) -> None:
        self.issues.append(Issue("ERROR", check, where, detail))

    def warn(self, check: str, where: str, detail: str) -> None:
        self.issues.append(Issue("WARN", check, where, detail))

    def info(self, check: str, where: str, detail: str) -> None:
        self.issues.append(Issue("INFO", check, where, detail))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]
