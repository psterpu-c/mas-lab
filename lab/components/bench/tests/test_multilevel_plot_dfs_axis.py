#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Fork detection and the DFS virtual-position axis — pure-function unit tests.

These exercise tree.py's _detect_delegation_forks / _reset_branch_agent_call_ids /
_make_call_sequence / _assign_dfs_positions directly on hand-built record
lists, independent of _build_dag/_build_call_records, mirroring the
sibling-batch delegation scenario confirmed against a live trace: a
moderator dispatches 3 delegate_to_schedule_agent calls in one LLM turn (all
sharing nearly — but never exactly — the same dispatch instant, confirmed
from a real trace: 1784124930.5783172 / .578378 / .578434), but only the
first sibling's subtree is explored — and takes real wall-clock time —
before the second sibling's own marker is appended to the DFS-ordered call
sequence.
"""

from __future__ import annotations

from mas.lab.plots.multilevel_trajectory.tree import (
    _assign_dfs_positions,
    _detect_delegation_forks,
    _make_agent_sequence,
    _make_call_sequence,
    _reset_branch_agent_call_ids,
)

# Sibling dispatch instants: real traces never produce exact float ties (see
# module docstring) — microsecond-distinct, in dispatch order.
_D1, _D2, _D3 = 5.0001, 5.0002, 5.0003


def _rec(call_id, parent_call_id, level, call_type, start_ts, end_ts, agent_id=""):
    return {
        "call_id": call_id,
        "parent_call_id": parent_call_id,
        "level": level,
        "call_type": call_type,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "agent_id": agent_id,
    }


def _sibling_batch_records():
    """moderator delegates to schedule_agent 3 times in one batch (all
    dispatched within a fraction of a millisecond of each other), but only
    sibling #1's subtree is explored (5.0001 -> 10.0) before sibling #2's own
    marker (still at its original ~5.0002 dispatch instant) is appended."""
    return [
        _rec("mod", None, "agent", "AgentCall", 0.0, 20.0, "moderator"),
        _rec("deleg1", "mod", "call", "ToolCall", _D1, _D1),
        _rec("sched1", "deleg1", "agent", "AgentCall", _D1, 10.0, "schedule_agent"),
        _rec("deleg2", "mod", "call", "ToolCall", _D2, _D2),
        _rec("sched2", "deleg2", "agent", "AgentCall", _D2, 12.0, "schedule_agent"),
        _rec("deleg3", "mod", "call", "ToolCall", _D3, _D3),
        _rec("sched3", "deleg3", "agent", "AgentCall", _D3, 14.0, "schedule_agent"),
    ]


def test_detect_delegation_forks_groups_by_real_fork_parent_id():
    records = _sibling_batch_records()
    parent_of = {r["call_id"]: r["parent_call_id"] for r in records}
    rec_by_id = {r["call_id"]: r for r in records}

    groups = _detect_delegation_forks(records, parent_of, rec_by_id)

    assert set(groups) == {"mod"}
    assert [m["call_id"] for m in groups["mod"]] == ["sched1", "sched2", "sched3"]


def test_detect_delegation_forks_omits_single_delegate():
    records = [
        _rec("mod", None, "agent", "AgentCall", 0.0, 10.0, "moderator"),
        _rec("deleg1", "mod", "call", "ToolCall", 5.0, 5.0),
        _rec("itin", "deleg1", "agent", "AgentCall", 5.0, 8.0, "itinerary_agent"),
    ]
    parent_of = {r["call_id"]: r["parent_call_id"] for r in records}
    rec_by_id = {r["call_id"]: r for r in records}

    assert _detect_delegation_forks(records, parent_of, rec_by_id) == {}


def test_make_call_sequence_empty_branch_fallback_uses_processing_call_type():
    """A fork branch whose subtree makes zero real calls (cut off before its
    first LLM/Tool call — sched2/sched3 in _sibling_batch_records() have no
    call-level children at all) still needs a _branch_entry-tagged
    placeholder so the reset separator has somewhere to land. That
    placeholder must never carry call_type="AgentCall": this list backs the
    Calls lane, which only ever holds LLM/Tool/Memory/RAG/Processing
    records — an AgentCall here would render as a wrong-type, agent-colored
    "Agent" bar (dag.py's trans() has no override for it, and constants.py
    maps AgentCall to the agent palette/label and excludes it from ever
    being instant)."""
    records = _sibling_batch_records()
    parent_of = {r["call_id"]: r["parent_call_id"] for r in records}
    rec_by_id = {r["call_id"]: r for r in records}
    children_of = _children_of(records)
    fork_groups = _detect_delegation_forks(records, parent_of, rec_by_id)
    reset_ids = _reset_branch_agent_call_ids(fork_groups)
    agent_sequence = _make_agent_sequence(records, children_of, parent_of)

    call_sequence = _make_call_sequence(agent_sequence, children_of, reset_ids)

    branch_entries = {r["call_id"]: r for r in call_sequence if r.get("_branch_entry")}
    assert set(branch_entries) == {"sched2", "sched3"}
    for rec in branch_entries.values():
        assert rec["call_type"] == "ProcessingCall", rec


def test_make_call_sequence_reused_call_id_respects_next_turn_boundary():
    """Two back-to-back invocations of a repeatedly-delegated agent share one
    runtime call_id pool (children_of[shared_id]) — the first invocation
    keeps that id as its own call_id, the second gets a records.py-suffixed
    id but still queries the same pool via _reused_call_id. Rule 2 gives
    the two turns an exact zero-gap boundary, but _collect's _TS_TOL (50ms)
    padding must never let turn 1 reach across that boundary and steal a
    call that starts just after it (llm2, at +30ms) but structurally
    belongs to turn 2 — regression for exactly this: a live trip-planner
    trace where turn 1's own lookup silently claimed turn 2's real LLM
    call, leaving turn 2 looking like an empty invocation."""
    shared = "sched-u1-exec"
    records = [
        _rec("llm1", shared, "call", "LLMCall", 1.0, 1.9, "sched"),
        _rec("llm2", shared, "call", "LLMCall", 2.03, 2.9, "sched"),
    ]
    children_of = _children_of(records)
    agent_sequence = [
        {"call_id": shared, "start_ts": 1.0, "end_ts": 2.0, "agent_id": "sched"},
        {"call_id": "sched-2", "start_ts": 2.0, "end_ts": 5.0, "agent_id": "sched",
         "_reused_call_id": shared},
    ]

    call_sequence = _make_call_sequence(agent_sequence, children_of, frozenset())

    by_id = {r["call_id"]: r for r in call_sequence}
    assert set(by_id) == {"llm1", "llm2"}
    assert by_id["llm2"]["start_ts"] == 2.03, by_id["llm2"]


def test_reset_branch_agent_call_ids_excludes_rank_zero():
    records = _sibling_batch_records()
    parent_of = {r["call_id"]: r["parent_call_id"] for r in records}
    rec_by_id = {r["call_id"]: r for r in records}
    fork_groups = _detect_delegation_forks(records, parent_of, rec_by_id)

    assert _reset_branch_agent_call_ids(fork_groups) == {"sched2", "sched3"}


def _children_of(records):
    children = {}
    for r in records:
        pid = r.get("parent_call_id")
        if pid is not None:
            children.setdefault(pid, []).append(r)
    for kids in children.values():
        kids.sort(key=lambda r: r["start_ts"])
    return children


def test_make_call_sequence_tags_branch_entry_and_widens_marker_gap():
    """The delegating marker for a later sibling must stay strictly separate
    (in real ts) from that sibling's own first call, so the branch-reset
    boundary has somewhere to land — regression for the marker appearing to
    belong to the new branch instead of closing out the delegating agent's own."""
    records = _sibling_batch_records() + [
        _rec("sched1-ctx", "sched1", "call", "ProcessingCall", _D1, _D1),
        _rec("sched2-ctx", "sched2", "call", "ProcessingCall", _D2, _D2),
        _rec("sched3-ctx", "sched3", "call", "ProcessingCall", _D3, _D3),
    ]
    parent_of = {r["call_id"]: r["parent_call_id"] for r in records}
    rec_by_id = {r["call_id"]: r for r in records}
    children_of = _children_of(records)
    fork_groups = _detect_delegation_forks(records, parent_of, rec_by_id)
    reset_ids = _reset_branch_agent_call_ids(fork_groups)
    agent_sequence = _make_agent_sequence(records, children_of, parent_of)

    call_sequence = _make_call_sequence(agent_sequence, children_of, reset_ids)

    by_id = {r["call_id"]: r for r in call_sequence}
    deleg1, deleg2, deleg3 = by_id["deleg1"], by_id["deleg2"], by_id["deleg3"]
    ctx1, ctx2, ctx3 = by_id["sched1-ctx"], by_id["sched2-ctx"], by_id["sched3-ctx"]

    # All 3 markers dispatch within the same tolerance window (a genuine
    # sibling batch — see _collect's window check), so they land consecutively
    # in call order before any of their subtrees are visited: deleg1, deleg2,
    # deleg3, THEN sched1's own subtree starts. sched2/sched3's own subtrees
    # are only reached later still, in later agent_sequence iterations.
    assert [r["call_id"] for r in call_sequence[:4]] == ["deleg1", "deleg2", "deleg3", "sched1-ctx"]

    # Rank-0 branch (sched1) needs no reset tag — it's the fork's first
    # branch — but its own subtree still can't start before ALL 3 markers
    # have finished being registered.
    assert "_branch_entry" not in ctx1
    assert ctx1["start_ts"] == deleg3["end_ts"]

    # Rank>=1 branches (sched2, sched3) get tagged and land strictly after
    # the marker cluster AND after sched1's own subtree entry — a genuinely
    # separate timestamp, not one that coincides with any marker's own close.
    assert ctx2.get("_branch_entry") == "sched2"
    assert ctx2["start_ts"] > deleg2["end_ts"]
    assert ctx3.get("_branch_entry") == "sched3"
    assert ctx3["start_ts"] > ctx2["start_ts"]

    # No record ever has end_ts < start_ts (a real, non-inverted interval).
    for rec in call_sequence:
        assert rec["end_ts"] >= rec["start_ts"], rec


def test_assign_dfs_positions_monotonic_across_sibling_batch():
    """The core regression: real start_ts is NOT monotonic in DFS append
    order (sibling #2/#3's markers carry their own early dispatch ts,
    appearing in the flat list AFTER sibling #1's subtree already advanced
    real time well past it), but dfs_pos must still increase monotonically."""
    call_sequence = [
        _rec("deleg1", "mod", "call", "ToolCall", _D1, _D1 + 0.001),
        _rec("sched1-llm", "sched1", "call", "LLMCall", _D1 + 0.001, 10.0),
        _rec("deleg2", "mod", "call", "ToolCall", _D2, _D2 + 0.001),
        {**_rec("sched2-llm", "sched2", "call", "LLMCall", _D2 + 0.002, 12.0),
         "_branch_entry": "sched2"},
        _rec("deleg3", "mod", "call", "ToolCall", _D3, _D3 + 0.001),
        {**_rec("sched3-llm", "sched3", "call", "LLMCall", _D3 + 0.002, 14.0),
         "_branch_entry": "sched3"},
    ]
    # Sanity: the fixture must actually reproduce a backward jump in real
    # time — sibling #2's marker sits, in DFS order, right after sibling
    # #1's subtree already advanced real time to 10.0.
    assert call_sequence[2]["start_ts"] < call_sequence[1]["end_ts"]
    assert call_sequence[4]["start_ts"] < call_sequence[3]["end_ts"]

    ts_to_pos, reset_branch_id = _assign_dfs_positions(call_sequence)

    seen = []
    for r in call_sequence:
        seen.append(ts_to_pos[r["start_ts"]])
        seen.append(ts_to_pos[r["end_ts"]])
    assert seen == sorted(seen), seen
    assert reset_branch_id == {
        _D2 + 0.002: "sched2",
        _D3 + 0.002: "sched3",
    }


def test_assign_dfs_positions_marker_stays_before_its_own_reset():
    """The delegating marker's own end must land BEFORE the reset it
    triggers — not at/after it — so the marker renders on the parent's side
    of the branch separator, not the child's."""
    call_sequence = [
        _rec("deleg1", "mod", "call", "ToolCall", _D1, _D1 + 0.001),
        _rec("sched1-llm", "sched1", "call", "LLMCall", _D1 + 0.001, 10.0),
        _rec("deleg2", "mod", "call", "ToolCall", _D2, _D2 + 0.001),
        {**_rec("sched2-llm", "sched2", "call", "LLMCall", _D2 + 0.002, 12.0),
         "_branch_entry": "sched2"},
    ]
    ts_to_pos, reset_branch_id = _assign_dfs_positions(call_sequence)

    marker_end_pos = ts_to_pos[_D2 + 0.001]
    branch_entry_pos = ts_to_pos[_D2 + 0.002]
    assert marker_end_pos < branch_entry_pos
    assert (_D2 + 0.001) not in reset_branch_id
    assert reset_branch_id[_D2 + 0.002] == "sched2"


def test_assign_dfs_positions_preserves_relative_durations_within_branch():
    """No fork involved: dfs_pos deltas must equal real time deltas, so
    relative call durations still look proportional within a single branch."""
    call_sequence = [
        _rec("c1", "a", "call", "LLMCall", 0.0, 3.0),
        _rec("c2", "a", "call", "ToolCall", 3.0, 4.0),
        _rec("c3", "a", "call", "LLMCall", 4.0, 10.0),
    ]
    ts_to_pos, reset_branch_id = _assign_dfs_positions(call_sequence)

    assert reset_branch_id == {}
    assert ts_to_pos[3.0] - ts_to_pos[0.0] == 3.0
    assert ts_to_pos[4.0] - ts_to_pos[3.0] == 1.0
    assert ts_to_pos[10.0] - ts_to_pos[4.0] == 6.0
