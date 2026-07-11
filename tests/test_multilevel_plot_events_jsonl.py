#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Multilevel trajectory plot from native events.jsonl (OSS path)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_FIXTURE = (
    REPO_ROOT / "docs/tutorials/03-experiments-and-analysis/fixtures/events.jsonl"
)


@pytest.mark.parametrize("fmt", ["html", "svg"])
def test_multilevel_plot_renders_from_events_jsonl(fmt: str, tmp_path: Path) -> None:
    pytest.importorskip("mas.lab.plots.multilevel_trajectory")
    from mas.lab.plots.multilevel_trajectory import plot_multilevel_trajectory
    from mas.lab.plots.trajectory import load_trace

    assert EVENTS_FIXTURE.is_file(), f"missing fixture: {EVENTS_FIXTURE}"
    events = load_trace(EVENTS_FIXTURE)
    assert events, "fixture events.jsonl should not be empty"

    rendered = plot_multilevel_trajectory(events, fmt=fmt, title="Tutorial 03 fixture")
    assert len(rendered) > 200
    if fmt == "html":
        assert "<" in rendered
    else:
        assert "<svg" in rendered

    out = tmp_path / f"multilevel.{fmt if fmt != 'html' else 'html'}"
    out.write_text(rendered, encoding="utf-8")
    assert out.stat().st_size > 200


def _tool_loop_events() -> list[dict]:
    """A single-agent tool-use loop (LLM → tool → LLM), each engine op emitted
    twice (~1 ms apart) as the native observability layer does. Mirrors the
    real tutorial-02 trace. Both LLM calls correctly carry the agent's own
    call_id as parent_call_id — a real runtime fact, never reconstructed."""
    moder, tool_cid, llm1, llm2 = "moderator", "tool-1", "llm-1", "llm-2"
    return [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": moder, "parent_call_id": "mas-1",
         "agent_id": moder, "timestamp": 0.0, "input": "hi"},
        # LLM 1 (double-emitted) — child of the agent
        {"kind": "llm_call_start", "call_id": llm1, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 1, "timestamp": 0.0},
        {"kind": "llm_call_start", "call_id": llm1, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 1, "timestamp": 0.001},
        {"kind": "llm_call_end", "call_id": llm1, "agent_id": moder,
         "correlation_id": 1, "timestamp": 1.6},
        {"kind": "llm_call_end", "call_id": llm1, "agent_id": moder,
         "correlation_id": 1, "timestamp": 1.601},
        # Tool (double-emitted) — child of the agent
        {"kind": "tool_call_start", "call_id": tool_cid, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 2, "timestamp": 1.61, "tool_name": "contract_call"},
        {"kind": "tool_call_start", "call_id": tool_cid, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 2, "timestamp": 1.611, "tool_name": "contract_call"},
        {"kind": "tool_call_end", "call_id": tool_cid, "agent_id": moder,
         "correlation_id": 2, "timestamp": 8.0, "tool_name": "contract_call"},
        # LLM 2 (double-emitted) — child of the agent, same as LLM 1.
        {"kind": "llm_call_start", "call_id": llm2, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 3, "timestamp": 8.01},
        {"kind": "llm_call_start", "call_id": llm2, "parent_call_id": moder,
         "agent_id": moder, "correlation_id": 3, "timestamp": 8.011},
        {"kind": "llm_call_end", "call_id": llm2, "agent_id": moder,
         "correlation_id": 3, "timestamp": 9.3, "output": "done"},
        {"kind": "llm_call_end", "call_id": llm2, "agent_id": moder,
         "correlation_id": 3, "timestamp": 9.301, "output": "done"},
        {"kind": "execution_end", "call_id": moder, "agent_id": moder,
         "timestamp": 9.3, "output": "done"},
        {"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": 9.3},
    ]


def test_duplicate_engine_ops_are_deduped() -> None:
    """Each engine op is emitted twice by the native layer; records.py must keep
    exactly one record per op so bars are not doubled/overlapped."""
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    recs = _build_call_records(_tool_loop_events())
    calls = [r for r in recs if r["level"] == "call"]
    assert [(r["call_type"], r["label"]) for r in calls] == [
        ("LLMCall", "LLM"),
        ("ToolCall", "contract_call"),
        ("LLMCall", "LLM"),
    ]


def test_tool_loop_dag_has_no_overlap() -> None:
    """End-to-end: the DAG for a tool-use loop keeps three distinct call bars
    (LLM → tool → LLM) with no overlapping transitions in the Calls lane —
    since each is correctly parented directly to the agent (a sibling of the
    other two), not nested under one another."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    events = _tool_loop_events()
    _state_reg, lanes = _build_dag(_build_call_records(events), events)
    calls_lane = next(l for l in lanes if l.lane_id == "calls")
    bars = [el for el in calls_lane.sequence if getattr(el, "call_type", None)]
    assert [b.label for b in bars] == ["LLM", "contract_call", "LLM"]
    # No two consecutive call bars overlap in time.
    for a, b in zip(bars, bars[1:]):
        assert a.end_ts <= b.start_ts + 1e-6, f"{a.label} overlaps {b.label}"


def _delegation_events() -> list[dict]:
    """A moderator that delegates one turn to a peer (schedule_agent).

    Mirrors the native multi-agent trace: the runtime threads a real
    caller_call_id through the delegation contract (InvokeEngineIo.call_id ->
    execute_engine_tool -> DelegationContract -> RunTurnFn, resolved by the
    driver via ObservabilityOperator.call_id_for), so the peer's own
    execution_start carries the delegating tool call's own call_id as its
    parent_call_id directly on the wire — no reconstruction needed. Both
    agents' engine ops restart their correlation ids at 1 (per-agent
    operators)."""
    mod, sched = "moderator", "schedule_agent"
    return [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
        {"kind": "llm_call_start", "call_id": "m-llm-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 1, "timestamp": 0.0},
        {"kind": "llm_call_end", "call_id": "m-llm-1", "agent_id": mod,
         "correlation_id": 1, "timestamp": 1.0},
        # Delegation tool on the moderator; the peer runs inside its window.
        {"kind": "tool_call_start", "call_id": "deleg-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 2, "timestamp": 1.1, "tool_name": "contract_call"},
        {"kind": "execution_start", "call_id": "sched-exec", "parent_call_id": "deleg-1",
         "agent_id": sched, "timestamp": 1.2, "input": "check schedule"},
        {"kind": "llm_call_start", "call_id": "s-llm-1", "parent_call_id": "sched-exec",
         "agent_id": sched, "correlation_id": 1, "timestamp": 1.3},
        {"kind": "llm_call_end", "call_id": "s-llm-1", "agent_id": sched,
         "correlation_id": 1, "timestamp": 3.0, "output": "no routes"},
        {"kind": "execution_end", "call_id": "sched-exec", "agent_id": sched, "timestamp": 3.1},
        {"kind": "tool_call_end", "call_id": "deleg-1", "agent_id": mod,
         "correlation_id": 2, "timestamp": 3.2, "tool_name": "contract_call"},
        # Moderator resumes and synthesises the final answer.
        {"kind": "llm_call_start", "call_id": "m-llm-2", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 3, "timestamp": 3.3},
        {"kind": "llm_call_end", "call_id": "m-llm-2", "agent_id": mod,
         "correlation_id": 3, "timestamp": 4.5, "output": "here is your plan"},
        {"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": 4.5},
        {"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": 4.5},
    ]


def test_delegation_agent_lane_interleaves_moderator_and_peer() -> None:
    """A delegated peer must appear on the Agent lane with the delegating agent
    resuming after it: moderator → schedule_agent → moderator."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    events = _delegation_events()
    _state_reg, lanes = _build_dag(_build_call_records(events), events)
    agents = next(l for l in lanes if l.lane_id == "agents")
    labels = [el.label for el in agents.sequence if getattr(el, "call_type", None)]
    assert labels == ["moderator", "schedule_agent", "moderator"], labels


def test_delegation_peer_llm_not_deduped_against_entry() -> None:
    """Per-agent correlation ids collide (both restart at 1); the peer's LLM
    must survive dedup and reach the Calls lane."""
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    recs = _build_call_records(_delegation_events())
    llm_agents = sorted(
        r["agent_id"] for r in recs if r["call_type"] == "LLMCall"
    )
    # moderator (×2) + schedule_agent (×1) — the peer LLM was not swallowed.
    assert llm_agents == ["moderator", "moderator", "schedule_agent"], llm_agents


def test_delegation_tool_is_last_call_before_peer() -> None:
    """The Calls lane is a DFS walk of the real call tree: the delegation tool
    call is the last node of the delegating agent's own segment, immediately
    followed by the peer's own calls — not hidden. Its true span covers the
    whole delegated execution (synchronous call-and-wait), which would
    overlap the peer's own calls in this shared lane row, so it renders as a
    near-instant dispatch marker rather than at full width. It gets a tiny
    (1ms) reserved width — not truly zero — so it never collapses onto the
    identical pixel column as the peer's own first boundary, which shares
    its exact dispatch timestamp by construction (_align_record_boundaries)."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    events = _delegation_events()
    _state_reg, lanes = _build_dag(_build_call_records(events), events)
    calls = next(l for l in lanes if l.lane_id == "calls")
    trans = [el for el in calls.sequence if getattr(el, "call_type", None)]
    labels = [el.label for el in trans]
    assert labels == ["LLM", "contract_call", "LLM", "LLM"], labels
    deleg = trans[labels.index("contract_call")]
    assert deleg.is_instant, deleg
    assert deleg.end_ts > deleg.start_ts, deleg


def _delegation_events_no_tool_call_end() -> list[dict]:
    """Real native traces never emit ``tool_call_end`` for a delegation tool —
    the delegate's own execution stands in for it — so the tool call is
    always ``_end_missing`` with a meaningless ``start_ts + 1.0`` placeholder
    end. The peer's own execution_start still carries the delegating tool's
    real call_id as parent_call_id directly (the runtime resolves and
    threads it through the delegation contract regardless of whether the
    tool call itself ever closes — see execute_engine_tool/DelegationContract).
    Here the peer runs for 8s, far longer than the tool's placeholder end —
    this exercises `_is_delegation_tool` in tree.py, which recognizes a
    delegation tool structurally (does it have an agent-level child in the
    call tree) rather than by any timestamp/duration comparison, so a peer
    outliving the tool's meaningless placeholder is a non-issue."""
    mod, sched = "moderator", "schedule_agent"
    return [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
        {"kind": "llm_call_start", "call_id": "m-llm-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 1, "timestamp": 0.0},
        {"kind": "llm_call_end", "call_id": "m-llm-1", "agent_id": mod,
         "correlation_id": 1, "timestamp": 1.0},
        # Delegation tool — note: no matching tool_call_end anywhere below.
        {"kind": "tool_call_start", "call_id": "deleg-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 2, "timestamp": 1.1, "tool_name": "delegate_to_schedule_agent"},
        {"kind": "execution_start", "call_id": "sched-exec", "parent_call_id": "deleg-1",
         "agent_id": sched, "timestamp": 1.2, "input": "check schedule"},
        {"kind": "llm_call_start", "call_id": "s-llm-1", "parent_call_id": "sched-exec",
         "agent_id": sched, "correlation_id": 1, "timestamp": 1.3},
        {"kind": "llm_call_end", "call_id": "s-llm-1", "agent_id": sched,
         "correlation_id": 1, "timestamp": 9.0, "output": "no routes"},
        {"kind": "execution_end", "call_id": "sched-exec", "agent_id": sched, "timestamp": 9.1},
        # Moderator resumes and synthesises the final answer.
        {"kind": "llm_call_start", "call_id": "m-llm-2", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 3, "timestamp": 9.2},
        {"kind": "llm_call_end", "call_id": "m-llm-2", "agent_id": mod,
         "correlation_id": 3, "timestamp": 10.5, "output": "here is your plan"},
        {"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": 10.5},
        {"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": 10.5},
    ]


def test_delegation_without_tool_call_end_still_links_peer() -> None:
    """A delegation tool call missing its own ``tool_call_end`` (the norm in
    real traces) must still be recognized as the delegate marker and stay
    the last call node of the delegating agent's own segment — not silently
    disconnected from the peer because its placeholder end is too short."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    events = _delegation_events_no_tool_call_end()
    recs = _build_call_records(events)

    peer = next(r for r in recs if r["call_id"] == "sched-exec")
    deleg = next(r for r in recs if r["call_id"] == "deleg-1")
    assert peer["parent_call_id"] == deleg["call_id"]

    _state_reg, lanes = _build_dag(recs, events)
    calls = next(l for l in lanes if l.lane_id == "calls")
    trans = [el for el in calls.sequence if getattr(el, "call_type", None)]
    labels = [el.label for el in trans]
    assert labels == ["LLM", "delegate_to_schedule_a", "LLM", "LLM"], labels
    deleg_trans = trans[labels.index("delegate_to_schedule_a")]
    assert deleg_trans.is_instant, deleg_trans
    assert deleg_trans.end_ts > deleg_trans.start_ts, deleg_trans


def _repeated_delegation_events() -> list[dict]:
    """Moderator delegates to schedule_agent 3 separate times in one run,
    each through its own delegation tool call. The native runtime reuses the
    exact same execution call_id (``schedule_agent-u1-exec``) for every turn
    of the same agent — real traces are not unique per delegation, only per
    (agent, turn-generation) — so all 3 peer AgentCall records arrive with an
    identical call_id before any dedup runs. Each turn's own execution_start
    still carries its OWN delegating tool's call_id as parent_call_id
    directly (the runtime resolves and threads a distinct caller_call_id per
    dispatch, keyed by that tool call's own correlation_id — see
    execute_engine_tool/ObservabilityOperator.call_id_for)."""
    mod, sched = "moderator", "schedule_agent"
    events: list[dict] = [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
    ]
    t = 0.0
    for i in range(3):
        events.append({"kind": "llm_call_start", "call_id": f"m-llm-{i}", "parent_call_id": mod,
                       "agent_id": mod, "correlation_id": i, "timestamp": t})
        t += 1.0
        events.append({"kind": "llm_call_end", "call_id": f"m-llm-{i}", "agent_id": mod,
                       "correlation_id": i, "timestamp": t})
        events.append({"kind": "tool_call_start", "call_id": f"deleg-{i}", "parent_call_id": mod,
                       "agent_id": mod, "correlation_id": 10 + i, "timestamp": t,
                       "tool_name": "delegate_to_schedule_agent"})
        # Same runtime call_id every time — the collision this test guards —
        # but each turn's OWN distinct delegating tool call_id as its parent.
        events.append({"kind": "execution_start", "call_id": "schedule_agent-u1-exec",
                       "parent_call_id": f"deleg-{i}", "agent_id": sched, "timestamp": t + 0.01,
                       "input": "check schedule"})
        t += 2.0
        events.append({"kind": "execution_end", "call_id": "schedule_agent-u1-exec",
                       "agent_id": sched, "timestamp": t})
    events.append({"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": t})
    events.append({"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": t})
    return events


def test_repeated_delegation_to_same_agent_gets_unique_branch_ids() -> None:
    """3 separate delegations to schedule_agent must produce 3 distinct,
    addressable call_ids and 3 distinct fork branches — not collapse onto
    one shared identity because the runtime reused the same execution
    call_id for every turn. A branch is only genuinely identifiable
    ("Fork branch node id = children ID") if its own call_id is unique."""
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records
    from mas.lab.plots.multilevel_trajectory.tree import (
        _build_call_tree,
        _detect_delegation_forks,
        _reset_branch_agent_call_ids,
    )

    events = _repeated_delegation_events()
    recs = _build_call_records(events)

    sched_execs = [r for r in recs if r["agent_id"] == "schedule_agent"]
    assert len(sched_execs) == 3
    assert len({r["call_id"] for r in sched_execs}) == 3, sched_execs

    children_of, parent_of = _build_call_tree(recs)
    rec_by_id = {r["call_id"]: r for r in recs}
    fork_groups = _detect_delegation_forks(recs, parent_of, rec_by_id)
    assert set(fork_groups) == {"moderator"}
    assert len(fork_groups["moderator"]) == 3

    reset_ids = _reset_branch_agent_call_ids(fork_groups)
    assert len(reset_ids) == 2  # every branch but the first needs a reset


def _repeated_delegation_events_never_closed() -> list[dict]:
    """Moderator delegates to schedule_agent 3 times through one shared
    delegation tool (never gets its own ``tool_call_end``, like every real
    delegation tool) — and, unlike ``_repeated_delegation_events`` above,
    none of the 3 ``schedule_agent-u1-exec`` turns ever receives its own
    ``execution_end`` either. records.py's ``_t_final`` extension (see
    ``_build_call_records``) then stretches all 3 turns' end_ts out to the
    trace's last known timestamp, so they heavily overlap each other in real
    wall-clock time — mirroring a live trace cut off mid-fork. Each turn does
    make one real LLM call of its own, parented (as the native runtime always
    does) to the literal, reused ``schedule_agent-u1-exec`` id rather than to
    whichever suffixed id records.py's collision handling minted for that
    particular turn."""
    mod, sched = "moderator", "schedule_agent"
    events: list[dict] = [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
        {"kind": "llm_call_start", "call_id": "m-llm-0", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 0, "timestamp": 0.0},
        {"kind": "llm_call_end", "call_id": "m-llm-0", "agent_id": mod,
         "correlation_id": 0, "timestamp": 1.0},
        # One shared delegation tool for all 3 turns — never gets tool_call_end.
        {"kind": "tool_call_start", "call_id": "deleg-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 1, "timestamp": 1.1,
         "tool_name": "delegate_to_schedule_agent"},
    ]
    t = 2.0
    for i in range(3):
        # Same runtime call_id every turn — the collision records.py suffixes
        # — but every turn correctly shares the ONE delegating tool's
        # call_id as its own parent_call_id (a single tool call legitimately
        # wrapping several sequential peer turns — see the DelegationContract
        # docstring/tests for why this must not be forced into a 1:1 shape).
        events.append({"kind": "execution_start", "call_id": "schedule_agent-u1-exec",
                       "parent_call_id": "deleg-1", "agent_id": sched, "timestamp": t,
                       "input": f"turn {i}"})
        # This turn's own real LLM call — parented to the reused base id, not
        # to whatever suffixed id records.py will later mint for this turn.
        events.append({"kind": "llm_call_start", "call_id": f"s-llm-{i}",
                       "parent_call_id": "schedule_agent-u1-exec", "agent_id": sched,
                       "correlation_id": i, "timestamp": t})
        events.append({"kind": "llm_call_end", "call_id": f"s-llm-{i}", "agent_id": sched,
                       "correlation_id": i, "timestamp": t + 0.5, "output": f"result {i}"})
        # No execution_end — this turn's AgentCall record never closes.
        t += 1.0
    events.append({"kind": "llm_call_start", "call_id": "m-llm-2", "parent_call_id": mod,
                   "agent_id": mod, "correlation_id": 2, "timestamp": t})
    events.append({"kind": "llm_call_end", "call_id": "m-llm-2", "agent_id": mod,
                   "correlation_id": 2, "timestamp": t + 1.0, "output": "here is your plan"})
    events.append({"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": t + 1.0})
    events.append({"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": t + 1.0})
    return events


def test_repeated_never_closed_delegation_every_branch_past_first_gets_reset() -> None:
    """Regression for a live trip-planner trace: when a delegate's AgentCall
    record never receives its own execution_end, ``_t_final`` extension makes
    it overlap its siblings almost entirely, and its real LLM/tool children
    are parented (by the runtime) to the shared, reused base call_id rather
    than to the suffixed id records.py minted for this particular turn. Every
    fork branch past the first must still get its call-level entry point
    tagged ``_branch_entry`` — not just the branches whose own suffixed id
    happens to already own its children in the call tree."""
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records
    from mas.lab.plots.multilevel_trajectory.tree import (
        _align_record_boundaries,
        _build_call_tree,
        _detect_delegation_forks,
        _make_agent_sequence,
        _make_call_sequence,
        _reset_branch_agent_call_ids,
    )

    events = _repeated_delegation_events_never_closed()
    records = _build_call_records(events)

    sched_execs = [
        r for r in records if r["agent_id"] == "schedule_agent" and r["level"] == "agent"
    ]
    assert len(sched_execs) == 3
    assert len({r["call_id"] for r in sched_execs}) == 3, sched_execs
    # All 3 never closed — the _t_final extension this bug hinges on.
    assert all(r.get("_end_missing") for r in sched_execs)

    children_of, parent_of = _build_call_tree(records)
    _align_record_boundaries(records, children_of)
    rec_by_id = {r["call_id"]: r for r in records}

    fork_groups = _detect_delegation_forks(records, parent_of, rec_by_id)
    assert set(fork_groups) == {"moderator"}
    assert len(fork_groups["moderator"]) == 3

    reset_ids = _reset_branch_agent_call_ids(fork_groups)
    assert len(reset_ids) == 2  # every branch but the first needs a reset

    agent_sequence = _make_agent_sequence(records, children_of, parent_of)
    call_sequence = _make_call_sequence(agent_sequence, children_of, reset_ids)

    tagged = {r["_branch_entry"] for r in call_sequence if r.get("_branch_entry")}
    assert tagged == reset_ids, (
        f"missing reset separator(s) for branch(es): {reset_ids - tagged}"
    )


def _three_way_fork_events() -> list[dict]:
    """moderator dispatches 3 DISTINCT agents (agent_a/b/c) in one sibling
    batch — a genuine 3-way fork, one rank beyond the 2-way case every other
    fixture in this file exercises. Regression for the Agents lane: rank>0
    branches (agent_b, agent_c) used to get no bracketing StateNode of their
    own at all (see dag.py's _bridge_to_state), which only ever produced a
    validator error for 3+-way forks — a 2-way fork's sole rank>0 branch
    happened to not need one inserted mid-sequence in the old code path."""
    mod = "moderator"
    return [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
        {"kind": "llm_call_start", "call_id": "m-llm-0", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 0, "timestamp": 0.0},
        {"kind": "llm_call_end", "call_id": "m-llm-0", "agent_id": mod,
         "correlation_id": 0, "timestamp": 1.0},
        # Sibling batch dispatched with a clear ordering (1.10 < 1.20 < 1.30)
        # so each peer unambiguously nearest-matches its own tool.
        {"kind": "tool_call_start", "call_id": "deleg-A", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 1, "timestamp": 1.10, "tool_name": "delegate_to_agent_a"},
        {"kind": "tool_call_start", "call_id": "deleg-B", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 2, "timestamp": 1.20, "tool_name": "delegate_to_agent_b"},
        {"kind": "tool_call_start", "call_id": "deleg-C", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 3, "timestamp": 1.30, "tool_name": "delegate_to_agent_c"},

        {"kind": "execution_start", "call_id": "agent_a-exec", "parent_call_id": "deleg-A",
         "agent_id": "agent_a", "timestamp": 1.11, "input": "task a"},
        {"kind": "llm_call_start", "call_id": "a-llm-1", "parent_call_id": "agent_a-exec",
         "agent_id": "agent_a", "correlation_id": 1, "timestamp": 1.15},
        {"kind": "llm_call_end", "call_id": "a-llm-1", "agent_id": "agent_a",
         "correlation_id": 1, "timestamp": 2.0, "output": "done a"},
        {"kind": "execution_end", "call_id": "agent_a-exec", "agent_id": "agent_a", "timestamp": 2.0},

        {"kind": "execution_start", "call_id": "agent_b-exec", "parent_call_id": "deleg-B",
         "agent_id": "agent_b", "timestamp": 1.21, "input": "task b"},
        {"kind": "llm_call_start", "call_id": "b-llm-1", "parent_call_id": "agent_b-exec",
         "agent_id": "agent_b", "correlation_id": 1, "timestamp": 1.25},
        {"kind": "llm_call_end", "call_id": "b-llm-1", "agent_id": "agent_b",
         "correlation_id": 1, "timestamp": 3.5, "output": "done b"},
        {"kind": "execution_end", "call_id": "agent_b-exec", "agent_id": "agent_b", "timestamp": 3.5},

        {"kind": "execution_start", "call_id": "agent_c-exec", "parent_call_id": "deleg-C",
         "agent_id": "agent_c", "timestamp": 1.31, "input": "task c"},
        {"kind": "llm_call_start", "call_id": "c-llm-1", "parent_call_id": "agent_c-exec",
         "agent_id": "agent_c", "correlation_id": 1, "timestamp": 1.35},
        {"kind": "llm_call_end", "call_id": "c-llm-1", "agent_id": "agent_c",
         "correlation_id": 1, "timestamp": 5.0, "output": "done c"},
        {"kind": "execution_end", "call_id": "agent_c-exec", "agent_id": "agent_c", "timestamp": 5.0},

        {"kind": "llm_call_start", "call_id": "m-llm-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 4, "timestamp": 5.1},
        {"kind": "llm_call_end", "call_id": "m-llm-1", "agent_id": mod,
         "correlation_id": 4, "timestamp": 5.5, "output": "final plan"},
        {"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": 5.5},
        {"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": 5.5},
    ]


def _three_way_fork_with_nested_fork_events() -> list[dict]:
    """Same outer 3-way fork as ``_three_way_fork_events``, but agent_c (the
    last-dispatched, longest-running outer branch) itself dispatches a
    nested 2-way fork (agent_d, agent_e) before finishing — mirroring a real
    trip-planner trace: a moderator forking to an itinerary agent + several
    schedule agents, one of which forks again to a peer + a concierge agent.
    agent_c's own execution_end lands within tolerance of the last nested
    branch's own end, so it gets no "resumes after the fork" tail fragment of
    its own — that narrower case (a fork whose branches finish in a
    different order than DFS visited them, so the *delegating* agent's own
    tail fragment lands on a timestamp no branch's dfs_pos reached) is a
    separate, deeper issue in how tree.py's _reserve_marker_width's single
    real-time cursor leaks across nested fork boundaries, not fixed here."""
    events = _three_way_fork_events()
    # Splice agent_c's nested fork in before its own execution_end/tail calls
    # (indices found by call_id — see _three_way_fork_events above).
    c_llm_end_idx = next(
        i for i, e in enumerate(events)
        if e["call_id"] == "c-llm-1" and e["kind"] == "llm_call_end"
    )
    events[c_llm_end_idx]["timestamp"] = 4.0
    c_exec_end_idx = next(
        i for i, e in enumerate(events)
        if e["call_id"] == "agent_c-exec" and e["kind"] == "execution_end"
    )
    nested: list[dict] = [
        {"kind": "tool_call_start", "call_id": "deleg-D", "parent_call_id": "agent_c-exec",
         "agent_id": "agent_c", "correlation_id": 2, "timestamp": 4.10, "tool_name": "delegate_to_agent_d"},
        {"kind": "tool_call_start", "call_id": "deleg-E", "parent_call_id": "agent_c-exec",
         "agent_id": "agent_c", "correlation_id": 3, "timestamp": 4.20, "tool_name": "delegate_to_agent_e"},
        {"kind": "execution_start", "call_id": "agent_d-exec", "parent_call_id": "deleg-D",
         "agent_id": "agent_d", "timestamp": 4.11, "input": "task d"},
        {"kind": "llm_call_start", "call_id": "d-llm-1", "parent_call_id": "agent_d-exec",
         "agent_id": "agent_d", "correlation_id": 1, "timestamp": 4.15},
        {"kind": "llm_call_end", "call_id": "d-llm-1", "agent_id": "agent_d",
         "correlation_id": 1, "timestamp": 5.0, "output": "done d"},
        {"kind": "execution_end", "call_id": "agent_d-exec", "agent_id": "agent_d", "timestamp": 5.0},
        {"kind": "execution_start", "call_id": "agent_e-exec", "parent_call_id": "deleg-E",
         "agent_id": "agent_e", "timestamp": 4.21, "input": "task e"},
        {"kind": "llm_call_start", "call_id": "e-llm-1", "parent_call_id": "agent_e-exec",
         "agent_id": "agent_e", "correlation_id": 1, "timestamp": 4.25},
        {"kind": "llm_call_end", "call_id": "e-llm-1", "agent_id": "agent_e",
         "correlation_id": 1, "timestamp": 6.0, "output": "done e"},
        {"kind": "execution_end", "call_id": "agent_e-exec", "agent_id": "agent_e", "timestamp": 6.0},
    ]
    events[c_exec_end_idx]["timestamp"] = 6.0
    events[c_exec_end_idx:c_exec_end_idx] = nested
    # Push the moderator's own resumption past the nested fork.
    m_llm_start_idx = next(
        i for i, e in enumerate(events)
        if e["call_id"] == "m-llm-1" and e["kind"] == "llm_call_start"
    )
    m_llm_end_idx = next(
        i for i, e in enumerate(events)
        if e["call_id"] == "m-llm-1" and e["kind"] == "llm_call_end"
    )
    events[m_llm_start_idx]["timestamp"] = 6.1
    events[m_llm_end_idx]["timestamp"] = 6.5
    events[next(i for i, e in enumerate(events)
                if e["call_id"] == "moderator" and e["kind"] == "execution_end")]["timestamp"] = 6.5
    events[next(i for i, e in enumerate(events)
                if e["call_id"] == "mas-1" and e["kind"] == "mas_call_end")]["timestamp"] = 6.5
    return events


def _agents_lane_dfs_positions(lanes) -> list[float]:
    """Flatten the Agents lane into its full sequence of dfs_pos values, in
    lane order — states contribute their own dfs_pos, transitions contribute
    both endpoints."""
    agents = next(l for l in lanes if l.lane_id == "agents")
    positions: list[float] = []
    for el in agents.sequence:
        if hasattr(el, "call_type"):
            positions.append(el.dfs_pos_start)
            positions.append(el.dfs_pos_end)
        else:
            positions.append(el.dfs_pos)
    return positions


def _mas_lane_inflated_t_max_events() -> list[dict]:
    """A short MAS session (mas_call_end at 0.024s) that also dispatches a
    delegation tool call with no matching ``tool_call_end`` — the norm for
    real traces (see `_delegation_events_no_tool_call_end`'s docstring). The
    tool's ``_end_missing`` placeholder (``start_ts + 1.0``, records.py) lands
    at ~1.02s, far past the MAS lane's own true end — this is exactly the
    shape golden-run fixtures like lifecycle-control/extensions hit in
    practice, where the global ``t_max = max(end_ts for all records)``
    (dag.py) is inflated by an unrelated lane's placeholder, well past this
    lane's last real state."""
    mod, sched = "moderator", "schedule_agent"
    return [
        {"kind": "mas_call_start", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.0},
        {"kind": "execution_start", "call_id": mod, "parent_call_id": "mas-1",
         "agent_id": mod, "timestamp": 0.0, "input": "plan a trip"},
        {"kind": "tool_call_start", "call_id": "deleg-1", "parent_call_id": mod,
         "agent_id": mod, "correlation_id": 1, "timestamp": 0.02, "tool_name": "delegate_to_schedule_agent"},
        {"kind": "execution_start", "call_id": "sched-exec", "parent_call_id": "deleg-1",
         "agent_id": sched, "timestamp": 0.021, "input": "check schedule"},
        {"kind": "execution_end", "call_id": "sched-exec", "agent_id": sched, "timestamp": 0.023},
        {"kind": "execution_end", "call_id": mod, "agent_id": mod, "timestamp": 0.024},
        {"kind": "mas_call_end", "call_id": "mas-1", "agent_id": "mas", "timestamp": 0.024},
    ]


def test_mas_lane_valid_when_global_t_max_is_inflated() -> None:
    """Regression: the MAS lane must never produce two consecutive StateNodes
    with no Trans between them when the global `t_max` sits well past this
    lane's own last real state (the validator error originally caught on the
    lifecycle-control/extensions golden fixtures).

    With the "impossible children" reparenting heuristic removed (parent_call_id
    is a real runtime fact, never overridden by a timestamp comparison),
    `deleg-1` correctly stays parented to `moderator` — so `_align_record_
    boundaries`'s bottom-up envelope pass (a parent must temporally contain its
    children) naturally extends `moderator`/`mas-1`'s own end_ts to cover
    deleg-1's inflated `_end_missing` placeholder, and the MAS lane's own
    MASCall transition already reaches `t_max` with no gap — dag.py's
    `_bridge_to_state` tail-check (still there as a general safety net for
    whatever record structure DOESN'T get naturally extended this way) simply
    isn't needed for this specific shape anymore. The invariant that matters —
    no consecutive states, no validator errors — is what's asserted here, not
    the specific mechanism that achieves it."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records
    from mas.lab.plots._trajectory_validator import validate_trajectory_dag

    events = _mas_lane_inflated_t_max_events()
    records = _build_call_records(events)
    state_reg, lanes = _build_dag(records, events)

    issues = validate_trajectory_dag(state_reg, lanes)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors

    mas_lane = next(l for l in lanes if l.lane_id == "mas")
    types = [
        el.call_type if hasattr(el, "call_type") else "state"
        for el in mas_lane.sequence
    ]
    # No two "state" entries ever sit back-to-back with nothing between them.
    for a, b in zip(types, types[1:]):
        assert not (a == "state" and b == "state"), types


def test_three_way_fork_agents_lane_is_structurally_valid() -> None:
    """A 3-way fork's Agents lane must pass every structural check the
    trajectory validator runs — no consecutive states, no consecutive
    transitions — and every rank>0 branch (agent_b, agent_c) must get its own
    bracketing StateNode via a genuine connecting element (a BranchLink
    TransNode), not appended bare next to the previous branch's own state."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records
    from mas.lab.plots._trajectory_validator import validate_trajectory_dag

    events = _three_way_fork_events()
    records = _build_call_records(events)
    state_reg, lanes = _build_dag(records, events)

    issues = validate_trajectory_dag(state_reg, lanes)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors

    agents = next(l for l in lanes if l.lane_id == "agents")
    labels_and_types = [
        (el.call_type, el.label) if hasattr(el, "call_type") else "state"
        for el in agents.sequence
    ]
    # Every rank>0 branch (agent_b, agent_c) is preceded by a BranchLink
    # connector bridging it in from the previous branch's own end state —
    # never a bare state-to-state jump.
    assert ("BranchLink", "") in labels_and_types
    assert labels_and_types.count(("BranchLink", "")) == 2  # agent_b, agent_c

    agent_labels = [
        el.label for el in agents.sequence
        if hasattr(el, "call_type") and el.call_type == "AgentCall"
    ]
    assert agent_labels == ["moderator", "agent_a", "agent_b", "agent_c", "moderator"]


def test_three_way_fork_agents_lane_dfs_pos_is_monotonic() -> None:
    """The whole point of the DFS virtual-position axis: dfs_pos must never
    decrease while walking the Agents lane in sequence order, even though
    agent_b's and agent_c's own real start_ts sit well BEFORE agent_a's
    subtree already advanced real time — a fork's rank>0 branches keep their
    own early dispatch instant, not a wall-clock-ordered one."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records

    events = _three_way_fork_events()
    records = _build_call_records(events)
    _state_reg, lanes = _build_dag(records, events)

    positions = _agents_lane_dfs_positions(lanes)
    assert positions == sorted(positions), positions


def test_nested_fork_agents_lane_is_structurally_valid() -> None:
    """A fork nested inside one branch of an outer fork (agent_c forking to
    agent_d/agent_e) must still pass every structural validator check — the
    strict State/Trans alternation the Agents lane loop maintains does not
    depend on nesting depth."""
    from mas.lab.plots.multilevel_trajectory.dag import _build_dag
    from mas.lab.plots.multilevel_trajectory.records import _build_call_records
    from mas.lab.plots._trajectory_validator import validate_trajectory_dag

    events = _three_way_fork_with_nested_fork_events()
    records = _build_call_records(events)
    state_reg, lanes = _build_dag(records, events)

    issues = validate_trajectory_dag(state_reg, lanes)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors

    agent_labels = [
        el.label for lane in lanes if lane.lane_id == "agents"
        for el in lane.sequence
        if hasattr(el, "call_type") and el.call_type == "AgentCall"
    ]
    assert agent_labels == [
        "moderator", "agent_a", "agent_b", "agent_c", "agent_d", "agent_e", "moderator",
    ]


def test_multilevel_cli_from_events_jsonl(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from mas.lab.cli import app as cli

    runner = CliRunner()
    out = tmp_path / "swimlane.html"
    result = runner.invoke(
        cli,
        [
            "plot",
            "multilevel-trajectory",
            str(EVENTS_FIXTURE),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert out.stat().st_size > 200
