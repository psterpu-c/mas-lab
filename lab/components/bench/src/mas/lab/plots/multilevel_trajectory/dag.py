#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""DAG assembly: sequences → StateNode registry + LaneDef list."""

from collections import defaultdict
from typing import Optional

from mas.lab.plots._trajectory_validator import validate_trajectory_dag
from mas.lab.plots.multilevel_trajectory.annotations import (
    _HOVER_PRIORITY,
    _collect_annotations,
    _collect_context_provenance,
    _format_cpr_hover,
    _source_category,
    _stagger_coinc_processing_calls,
)
from mas.lab.plots.multilevel_trajectory.constants import (
    _PROC_TYPE_LABEL,
    _TS_TOL,
    TYPE_LABEL,
)
from mas.lab.plots.multilevel_trajectory.governance import (
    _collect_blocked_actions,
    _collect_governance_decisions,
    _collect_hitl_exchanges,
    _collect_retry_chains,
    _governance_severity,
    notable_governance,
)
from mas.lab.plots.multilevel_trajectory.models import LaneDef, StateNode, TransNode
from mas.lab.plots.multilevel_trajectory.records import (
    _extract_final_output,
    _extract_user_input,
    _synthesize_thinking_records,
)
from mas.lab.plots.multilevel_trajectory.tree import (
    _align_record_boundaries,
    _assign_dfs_positions,
    _build_call_tree,
    _detect_delegation_forks,
    _make_agent_sequence,
    _make_call_sequence,
    _reset_branch_agent_call_ids,
)

#: Every optional annotation layer _build_dag can attach on top of the core
#: call-tree structure. Each is a pure function over (events, records) — see
#: governance.py's module docstring — so disabling one just skips computing
#: and attaching that data; nothing else in the DAG depends on it.
_ALL_FACETS = frozenset({"cpr", "governance", "annotations", "thinking"})


def _build_dag(
    records: list[dict],
    events:  list[dict],
    show_provenance: bool = True,
    enabled_facets: "set[str] | None" = None,
) -> tuple[dict[float, StateNode], list[LaneDef]]:
    """Assemble the DAG from the call tree.

    ``enabled_facets`` toggles the optional annotation layers overlaid on the
    core structural DAG (``_ALL_FACETS``); ``None`` (default) enables all of
    them. ``show_provenance=False`` is sugar for excluding ``"cpr"`` — kept
    for backward compatibility with existing callers.

    Returns
    -------
    state_reg : {ts: StateNode}     — shared state registry
    lanes     : [LaneDef, …]       — ordered swim-lane sequences

    Each lane's ``sequence`` is a strict alternation::

        [StateNode, TransNode, StateNode, TransNode, …, StateNode]

    States at the same ``ts`` in different lanes are the same object
    (shared reference), which the renderer uses to draw multi-lane connectors.
    """
    facets = set(enabled_facets) if enabled_facets is not None else set(_ALL_FACETS)
    if not show_provenance:
        facets.discard("cpr")
    all_ts = [float(e.get("timestamp") or 0) for e in events if e.get("timestamp")]
    # Use record start/end times as the authoritative bounds — they include
    # synthetic end_ts (+1 s) for calls whose *_end event was missing,
    # which is always the correct diagram extent.  Raw event timestamps are
    # only a fallback when no records were produced.
    if records:
        t_min = min(r["start_ts"] for r in records)
        t_max = max(r["end_ts"]   for r in records)
    else:
        t_min = min(all_ts) if all_ts else 0.0
        t_max = max(all_ts) if all_ts else 1.0
    if t_max <= t_min:
        t_max = t_min + 1.0

    state_reg: dict[float, StateNode] = {}
    hover_pri: dict[float, int]       = {}

    # L2/L3 XAI annotation map: call_id → list of short summary strings
    ann_map: dict[str, list[str]] = (
        _collect_annotations(events, records) if "annotations" in facets else {}
    )

    # L4 context provenance: context_part_contributed → per-call_id summary
    cpr_map: dict[str, list[dict]] = (
        _collect_context_provenance(events, records) if "cpr" in facets else {}
    )

    # Governance facet: decisions, HITL Q&A, blocked-action ghosts, retries.
    # See governance.py's module docstring for why this operates on the same
    # (events, records) pair the KG adapter already produces.
    if "governance" in facets:
        gov_map:    dict[str, list[dict]] = _collect_governance_decisions(events, records)
        hitl_map:   dict = _collect_hitl_exchanges(events)
        blocked:    list[dict] = _collect_blocked_actions(events, records)
        retry_map:  dict[str, dict] = _collect_retry_chains(events, records)
    else:
        gov_map, hitl_map, blocked, retry_map = {}, {}, [], {}

    def state(ts: float, hover: str = "", level: str = "") -> StateNode:
        """Fetch or create the StateNode at *ts* (exact timestamp key).

        Hover text is overwritten only by a level with equal or higher
        priority (call > agent > mas > session).
        """
        pri = _HOVER_PRIORITY.get(level, -1)
        if ts not in state_reg:
            state_reg[ts] = StateNode(ts=ts, hover=hover)
            hover_pri[ts] = pri if hover else -1
        elif hover and pri >= hover_pri.get(ts, -1):
            state_reg[ts].hover = hover
            hover_pri[ts]       = pri
        return state_reg[ts]

    s_in  = _extract_user_input(events)
    s_out = _extract_final_output(events)
    entry = state(t_min, s_in,  "session")
    exit_ = state(t_max, s_out, "session")
    entry.is_user_entry = True
    exit_.is_user_exit  = True

    # Build call tree, align boundaries structurally, then derive sequences.
    children_of, parent_of = _build_call_tree(records)
    _align_record_boundaries(records, children_of)
    rec_by_id: dict[str, dict] = {r["call_id"]: r for r in records}
    mas_records    = sorted([r for r in records if r["level"] == "mas"],
                            key=lambda r: r["start_ts"])
    agent_sequence = _make_agent_sequence(records, children_of, parent_of)

    # Fork/Branch detection (tree-structural — see _detect_delegation_forks):
    # computed before _make_call_sequence so it can tag each fork's 2nd..Nth
    # sibling's own first call as a branch entry (see
    # _reset_branch_agent_call_ids / _make_call_sequence's docstring).
    fork_groups = _detect_delegation_forks(records, parent_of, rec_by_id)
    _reset_branch_ids = _reset_branch_agent_call_ids(fork_groups)
    call_sequence = _stagger_coinc_processing_calls(
        _make_call_sequence(agent_sequence, children_of, _reset_branch_ids)
    )

    # DFS virtual-position axis: call_sequence is already DFS-ordered (no
    # wall-clock sort) and already tagged, so walking it once assigns every
    # boundary timestamp a monotonic position, with a fixed-tick reset
    # (instead of the real, often-negative gap) at each branch entry.
    _ts_to_dfs_pos, _reset_branch_id = _assign_dfs_positions(call_sequence)

    # Extend the DFS axis to each fork branch's OWN raw start_ts too — the
    # Agents lane's true fragment boundary (see _make_agent_sequence) is NOT
    # the same timestamp as the branch's first call-level record tagged
    # _branch_entry in call_sequence (_reserve_marker_width pushes that one
    # later to clear the delegating marker cluster). _assign_dfs_positions
    # above only covers call_sequence's own timestamps, so without this the
    # generic real-time interpolation fallback below (which assumes nearby
    # timestamps are nearby in DFS order — false for a fork's 2nd..Nth
    # sibling, dispatched long before DFS reaches its subtree) would place
    # this raw boundary in the middle of an already-passed DFS range instead
    # of just before the branch's own tagged entry. Every rank>0 fork member
    # gets its raw start_ts pinned to a hair before its own _branch_entry
    # position — both the Agents lane (_bridge_to_state, below) and the Calls
    # lane's own "inject agent-boundary timestamps" pass rely on this.
    _reset_ts_by_branch_id: dict[str, float] = {
        _bid: _ts for _ts, _bid in _reset_branch_id.items()
    }
    for _members in fork_groups.values():
        for _member in _members[1:]:
            _raw_ts   = _member["start_ts"]
            _entry_ts = _reset_ts_by_branch_id.get(_member["call_id"])
            if _entry_ts is not None and _raw_ts not in _ts_to_dfs_pos:
                _ts_to_dfs_pos[_raw_ts] = _ts_to_dfs_pos[_entry_ts] - 1e-3

    seq_ctr = [0]

    def next_seq() -> int:
        seq_ctr[0] += 1
        return seq_ctr[0]

    def _bridge_to_state(
        elems: list, target_ts: float, hover: str = "", level: str = "agent",
    ) -> StateNode:
        """Ensure *elems* (a lane's in-progress sequence) reaches *target_ts*
        as its own bracketing StateNode, inserting a zero-content connector
        TransNode first when the lane doesn't already end there.

        Needed wherever DFS order jumps — forward or backward in real time —
        without a real call of its own to anchor the reset boundary to (see
        the Agents lane's rank>0 fork-branch handling below): the strict
        State/Trans alternation ``_trajectory_validator.py`` enforces means a
        StateNode can never simply be appended next to another StateNode, no
        matter how the timestamps compare. The Calls lane doesn't need this —
        every branch already has a real call record to tag as `_branch_entry`
        (see tree.py) — but the Agents lane has no such record to synthesize
        one from, so a plain connector plays that role instead.
        """
        prev = elems[-1]
        node = state(target_ts, hover, level)
        if node is prev:
            return node
        # Guard against a DFS-position regression: target_ts may already be
        # registered from an EARLIER-in-DFS-order fragment that happened to
        # land on this exact real timestamp by coincidence — e.g. a fork's
        # rank-0 branch's own end, when a LATER sibling's merge-computed
        # "delegating agent resumes after every branch joins" boundary
        # (_make_agent_sequence) lands on that same wall-clock instant
        # because that earlier branch happened to be the last one to finish
        # in real time even though DFS visited it first. Reusing that node's
        # dfs_pos here would walk this lane backward. Mint a fresh, distinct
        # real ts a hair after prev's own instead of touching the pre-existing
        # node — other lanes may still reference it correctly — the same
        # technique tree.py's _MARKER_DUR/_reserve_marker_width already use
        # elsewhere to keep genuinely-different moments from colliding on one
        # timestamp.
        #
        # NOTE: this only catches the case where target_ts is ALREADY
        # registered in _ts_to_dfs_pos with a stale value at the time this
        # runs. A fork whose branches complete in a different order than DFS
        # visited them can still leave the delegating agent's own tail
        # fragment (resuming after every branch joins) with an imperfect
        # position when target_ts isn't covered yet either — the generic
        # real-time interpolation fallback (dag.py's later pass) has no DFS
        # awareness. That narrower case is not fully solved here; see
        # test_multilevel_plot_dfs_axis.py's module docstring for the
        # regression coverage this fix does guarantee.
        _existing_pos = _ts_to_dfs_pos.get(target_ts)
        _prev_pos      = _ts_to_dfs_pos.get(prev.ts, 0.0)
        if _existing_pos is not None and _existing_pos < _prev_pos:
            _fresh_ts = prev.ts + 1e-3
            while _fresh_ts in state_reg:
                _fresh_ts += 1e-3
            node = state(_fresh_ts, hover, level)
            _ts_to_dfs_pos[_fresh_ts] = _prev_pos + 1e-3
            target_ts = _fresh_ts
        elems.append(TransNode(
            node_id=f"tr-bridge-{next_seq()}",
            call_type="BranchLink",
            label="",
            start_ts=prev.ts,
            end_ts=target_ts,
            level=level,
            agent_id="",
            seq=next_seq(),
            is_instant=True,
        ))
        elems.append(node)
        return node

    def _provenance_block(rec: dict, ct: str) -> str:
        """Build a provenance triplet block for hover enrichment.

        Returns a compact multi-line string of ``(subject, predicate, object)``
        triplets that describe the structural graph context of the call.
        Empty string when nothing meaningful can be derived (e.g. Session).
        """
        lines: list[str] = []
        _cid = rec.get("call_id", "")
        _aid = rec.get("agent_id", "")

        # (call, type, CallType)
        lines.append(f"(call, type, {ct})")

        # (call, executedBy, agent)
        if _aid:
            lines.append(f"(call, executedBy, {_aid})")

        # (call, containedIn, parent)
        _pid = parent_of.get(_cid)
        if _pid:
            _prec = rec_by_id.get(_pid)
            if _prec:
                _pt = _prec.get("call_type", "?")
                _pa = _prec.get("agent_id") or _prec.get("label") or _pid[:12]
                lines.append(f"(call, containedIn, {_pa} [{_pt}])")

        # (call, usedModel, model) — LLMCall only
        _model = rec.get("model", "")
        if _model and ct in ("LLMCall", "MITMCall"):
            lines.append(f"(call, usedModel, {_model})")

        # (call, invoked, tool) — ToolCall only
        _tool = rec.get("tool_name", "")
        if _tool:
            lines.append(f"(call, invoked, {_tool})")

        # (call, children, N) — non-leaf calls
        _kids = children_of.get(_cid, [])
        if _kids:
            _child_types = defaultdict(int)
            for k in _kids:
                _child_types[k.get("call_type", "?")] += 1
            _parts = [f"{cnt}×{ctype}" for ctype, cnt in _child_types.items()]
            lines.append(f"(call, contains, {' '.join(_parts)})")

        if len(lines) <= 1:
            return ""
        return "\n\n🔗 Provenance:\n" + "\n".join(f"  {l}" for l in lines)

    def trans(
        rec: dict, *,
        lane_level: str,
        node_id: str,
        call_type: Optional[str] = None,
        label: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> TransNode:
        ct  = call_type or rec.get("call_type", "AgentCall")
        lbl = label or rec.get("label") or TYPE_LABEL.get(ct, ct[:12])
        s   = start_ts if start_ts is not None else rec["start_ts"]
        e   = end_ts   if end_ts   is not None else (rec["end_ts"] or t_max)
        # Enrich hover_in with L2/L3 annotation summary (XAI)
        base_in  = rec.get("input", "")
        ann_lines = ann_map.get(rec.get("call_id", ""), [])
        hint     = ("\n\n[" + " | ".join(ann_lines) + "]") if ann_lines else ""
        # For LLMCall, prepend the call_id so it's visible in the hover panel.
        _cid = rec.get("call_id", "")
        _model = rec.get("model", "")
        call_id_prefix = ""
        if ct == "LLMCall" and _cid:
            _meta_parts = [f"call_id: {_cid}"]
            if _model:
                _meta_parts.append(f"model: {_model}")
            call_id_prefix = "📋 " + "  ·  ".join(_meta_parts) + "\n\n"
        _raw_out = rec.get("output", "")
        _thinking_txt = rec.get("thinking", "") if ct == "LLMCall" else ""
        if _thinking_txt:
            _hover_out = f"🧠 Thinking:\n\n{_thinking_txt}\n\n---\n\n{_raw_out}".strip()
        else:
            _hover_out = _raw_out
        # ProcessingCall: enrich hoverIn with the operation name header and
        # mark hoverOut as sentinel so JS can hide the useless "assembled" line.
        if ct == "ProcessingCall":
            _proc_name = rec.get("processing_name", "")
            # Detect fallback case: input == raw processing_type name (not real task)
            _is_type_name_fallback = base_in in _PROC_TYPE_LABEL
            _task_in = "(task content not captured in this trace)" if _is_type_name_fallback else base_in
            _op_header = f"[{_proc_name}]\n" if _proc_name else ""
            base_in    = (_op_header + _task_in).strip()
            # Clear output if it's just a status marker or a processing type name
            if _hover_out in ("", "assembled") or _hover_out in _PROC_TYPE_LABEL:
                _hover_out = ""
        # Build structured CPR data for rich JS rendering.
        # For ProcessingCall records that carry their own ``segments`` (per-actor
        # spans emitted by ObservabilityPlugin), build CPR from those segments
        # directly instead of from the cpr_map (which maps LLMCall call_ids).
        _rec_segments = rec.get("segments") if (show_provenance and ct == "ProcessingCall") else None
        if _rec_segments:
            _cpr_structured: list[dict] = []
            for _s in _rec_segments:
                _prov = _s.get("provenance") or {}
                _psrc = _s.get("source", "?")
                _pmech = _prov.get("mechanism", "inject")
                _entry: dict = {
                    "source": _psrc,
                    "category": _source_category(_psrc, _pmech),
                    "mechanism": _pmech,
                    "retrieval": _prov.get("retrieval", ""),
                    "decision": _prov.get("decision", ""),
                    "causeType": _prov.get("cause_type", "deterministic"),
                    "cause": _prov.get("cause", "?"),
                    "tokens": _s.get("tokens") or 0,
                    "retained": True,
                    "placement": _s.get("placement", ""),
                    "content": _s.get("content", ""),
                    "sourceType": _prov.get("source_type", ""),
                    "sourceId": _prov.get("source_id", ""),
                    "trigger": _prov.get("trigger", ""),
                    "actor": _prov.get("actor", ""),
                    "via": _prov.get("via", ""),
                }
                _ann = _prov.get("annotations")
                if _ann:
                    _entry["annotations"] = _ann
                _meta = _s.get("metadata")
                if _meta:
                    _entry["metadata"] = _meta
                _enrich = _prov.get("enrichments")
                if _enrich:
                    _entry["enrichments"] = _enrich
                _chain = _prov.get("chain")
                if _chain:
                    _entry["chain"] = _chain
                _cpr_structured.append(_entry)
            _cpr_raw = []  # no legacy CPR hover for per-actor spans
        else:
            _cpr_raw = cpr_map.get(_cid, [])
            _cpr_structured = []
            for _p in _cpr_raw:
                _psrc = _p.get("source", "?")
                _pmech = _p.get("mechanism", "inject")
                _entry: dict = {
                    "source": _psrc,
                    "category": _source_category(_psrc, _pmech),
                    "mechanism": _pmech,
                    "retrieval": _p.get("retrieval", ""),
                    "decision": _p.get("decision", ""),
                    "causeType": _p.get("cause_type", "deterministic"),
                    "cause": _p.get("cause", "?"),
                    "tokens": _p.get("token_estimate", 0),
                    "retained": _p.get("retained", True),
                    "placement": _p.get("placement", ""),
                    "content": _p.get("content") or _p.get("content_preview") or "",
                    "sourceType": _p.get("source_type", ""),
                    "sourceId": _p.get("source_id", ""),
                    "trigger": _p.get("trigger", ""),
                    "actor": _p.get("actor", ""),
                    "via": _p.get("via", ""),
                }
                _ann = _p.get("annotations")
                if _ann:
                    _entry["annotations"] = _ann
                _meta = _p.get("metadata")
                if _meta:
                    _entry["metadata"] = _meta
                _enrich = _p.get("enrichments")
                if _enrich:
                    _entry["enrichments"] = _enrich
                _chain = _p.get("chain")
                if _chain:
                    _entry["chain"] = _chain
                _cpr_structured.append(_entry)
        return TransNode(
            node_id=node_id,
            call_type=ct,
            label=lbl,
            start_ts=s,
            end_ts=e,
            level=lane_level,
            agent_id=rec.get("agent_id", ""),
            seq=next_seq(),
            hover_in=(call_id_prefix + base_in + hint
                      + _format_cpr_hover(_cpr_raw)
                      + _provenance_block(rec, ct)).strip(),
            hover_out=_hover_out,
            is_instant=(ct not in ("AgentCall", "Session", "MASCall", "ProcessingCall") and abs(e - s) <= _TS_TOL),
            cpr_data=_cpr_structured,
            model=_model,
            # notable_governance: suppress the badge overlay for plain ALLOW/LOG
            # (nearly every call has one) — only surface it when a decision
            # actually changed the outcome. The dedicated Governance lane below
            # shows every decision unfiltered; this only gates the overlay badge.
            governance=notable_governance(gov_map.get(_cid, [])),
            retry_group_id=retry_map.get(_cid, {}).get("groupId", ""),
            retry_attempt=retry_map.get(_cid, {}).get("attempt", 0),
        )

    lanes: list[LaneDef] = []

    # ── Session lane: one wide transition covering the full session ──────────
    s0 = state(t_min, s_in,  "session")
    s1 = state(t_max, s_out, "session")
    sess_lane = LaneDef("session", "session", "Session")
    sess_lane.sequence = [
        s0,
        TransNode(
            node_id="tr-session",
            call_type="Session",
            label="Session",
            start_ts=s0.ts,
            end_ts=s1.ts,
            level="session",
            agent_id="",
            seq=next_seq(),
            hover_in=s_in,
            hover_out=s_out,
        ),
        s1,
    ]
    lanes.append(sess_lane)

    def _hitl_between(t0: float, t1: float) -> Optional[dict]:
        """Find the human's actual question and answer for a gap between two
        MAS invocations, so the HITL bar shows what was asked and decided
        instead of an empty box."""
        for ev in events:
            if ev.get("kind") != "hitl_request":
                continue
            ts = float(ev.get("timestamp") or 0)
            if t0 - _TS_TOL <= ts <= t1 + _TS_TOL:
                return hitl_map.get(ev.get("correlation_id"))
        return None

    # ── MAS lane: one transition per MAS invocation, HITL gaps between them ──
    mas_lane = LaneDef("mas", "mas", "MAS")
    if mas_records:
        elems: list = [state(t_min, s_in, "session")]
        for i, mrec in enumerate(mas_records):
            end_ts = mrec["end_ts"] if mrec.get("end_ts", 0) > 0 else t_max
            elems.append(trans(mrec, lane_level="mas", node_id=f"tr-mas-{i}",
                               call_type="MASCall", label="MAS",
                               end_ts=end_ts))
            elems.append(state(end_ts, mrec.get("output", ""), "mas"))
            if i + 1 < len(mas_records):
                nxt = mas_records[i + 1]
                hitl_rec: dict = {"input": "", "output": "", "agent_id": ""}
                _hx = _hitl_between(end_ts, nxt["start_ts"])
                if _hx:
                    hitl_rec["input"] = _hx.get("question", "")
                    _resolution = _hx.get("resolution", "")
                    _answer = _hx.get("answer", "")
                    if _resolution or _answer:
                        hitl_rec["output"] = f"[{_resolution}] {_answer}".strip()
                elems.append(trans(hitl_rec, lane_level="mas", node_id=f"tr-hitl-{i}",
                                   call_type="HITL", label="HITL",
                                   start_ts=end_ts, end_ts=nxt["start_ts"]))
                elems.append(state(nxt["start_ts"], "", "mas"))
        if not (isinstance(elems[-1], StateNode)
                and abs(elems[-1].ts - t_max) <= _TS_TOL):
            # A bare append here would leave two consecutive StateNodes with
            # no Trans between them (t_max can be inflated well past this
            # lane's last real state by an unrelated lane's synthetic
            # end-missing placeholder — see records.py's _end_missing
            # fallback) — bridge it instead, same as the Agent lane below.
            _bridge_to_state(elems, t_max, s_out, "session")
        mas_lane.sequence = elems
    else:
        empty: dict = {"input": s_in, "output": s_out, "agent_id": ""}
        mas_lane.sequence = [
            state(t_min),
            trans(empty, lane_level="mas", node_id="tr-mas-0",
                  call_type="MASCall", label="MAS",
                  start_ts=t_min, end_ts=t_max),
            state(t_max),
        ]
    lanes.append(mas_lane)

    # ── Agent lane: DFS-derived sequence, delegation splits included ─────────

    # Fork/Branch groups (see fork_groups above) render sequentially along the
    # DFS virtual-position axis with a reset separator between them — there
    # is no side-by-side slot layout for overlapping siblings anymore; every
    # fork's branches, overlapping or not, are drawn one after another.
    _parallel_info: dict[str, tuple[str, int, int]] = {}  # agent call_id → (group_id, rank, size)
    for _fork_parent_id, _members in fork_groups.items():
        _gid = f"fork-{_fork_parent_id}"
        for _rank, _r in enumerate(_members):
            _parallel_info[_r["call_id"]] = (_gid, _rank, len(_members))

    if agent_sequence:
        agent_lane = LaneDef("agents", "agent", "Agents")
        elems = [state(t_min, s_in, "session")]
        _agent_branch_started: set[str] = set()  # rank>0 call_ids already bridged
        for i, arec in enumerate(agent_sequence):
            short  = arec["agent_id"].split(".")[-1][:18]
            end_ts = arec["end_ts"] if arec.get("end_ts", 0) > 0 else t_max
            _par   = _parallel_info.get(arec["call_id"])  # (group_id, rank, size) or None

            # Populate the START state of each agent fragment with its input so
            # delegation handoff boundaries show the task being passed in.
            #
            # In the common case elems[-1] IS already this exact ts — the
            # previous fragment's st_end placed the same StateNode object
            # there — so _bridge_to_state's identity check makes this a no-op
            # (no new element appended). But that assumption can break in more
            # ways than just "rank > 0's first fragment" (a fork's 2nd..Nth
            # sibling, whose own subtree hasn't been explored yet — DFS went
            # back up the call tree and down into a new sibling, often a real
            # backward jump relative to elapsed time): a fork whose branches
            # finish in a DIFFERENT order than DFS visited them also leaves
            # the delegating agent's own tail fragment (resuming after ALL
            # branches join) starting at a timestamp elems[-1] never reached —
            # elems[-1] is whichever branch DFS happened to visit last, not
            # necessarily the one that ended last in real time. Bridging
            # unconditionally (instead of only for the rank > 0 case) makes
            # every fragment boundary robust to both, with no cost for the
            # (overwhelming majority of) fragments where the assumption holds.
            _target_ts = arec["start_ts"]
            if _par and _par[1] > 0 and arec["call_id"] not in _agent_branch_started:
                # rank > 0's first fragment — only the branch's FIRST
                # agent_sequence fragment needs this; a delegate that itself
                # further delegates gets split into pre/tail fragments, and
                # only the pre fragment is the true branch entry (the tail
                # fragment falls through to the plain arec["start_ts"] above
                # like any other continuation).
                #
                # Bridge to the branch's own _branch_entry timestamp
                # (_reset_ts_by_branch_id), NOT arec["start_ts"] — the two are
                # usually only ~1ms apart in principle (the delegate's own
                # execution starts just after its dispatching tool call, per
                # _align_record_boundaries), but both are computed from the
                # SAME "dispatch instant + _MARKER_DUR" arithmetic as a
                # sibling batch's *other* delegation markers, so
                # arec["start_ts"] can land exactly on some unrelated
                # marker's own timestamp — reusing that ts here would
                # silently steal the marker's already-correct (intentionally
                # clustered, not reset) dfs_pos instead of getting a reset
                # position of its own. The branch's own _branch_entry
                # timestamp has no such collision — it is unique to this
                # branch by construction — and already has a correct,
                # monotonic dfs_pos from _assign_dfs_positions.
                _target_ts = _reset_ts_by_branch_id.get(arec["call_id"], arec["start_ts"])
                _agent_branch_started.add(arec["call_id"])

            st_start = _bridge_to_state(elems, _target_ts, arec.get("input", ""))
            if _par and _par[1] == 0:
                # Fork state: mark the boundary where parallel branches
                # split. For rank 0 the fork state is (almost always)
                # already the previous iteration's st_end.
                _gid, _rank, _size = _par
                st_start.is_fork           = True
                st_start.parallel_group_id = _gid
                st_start.parallel_size     = _size
            st_start.hover_by_lane["agents"] = arec.get("input", "")

            # st_start.ts is the authoritative start for the real Trans below
            # too — _bridge_to_state may have minted a fresh ts distinct from
            # arec["start_ts"] (the branch-entry override above, or the
            # dfs_pos-regression guard inside _bridge_to_state itself), and
            # the Trans must agree with its own preceding StateNode so
            # dfs_pos_start resolves to the same, already-correct position
            # rather than falling through to a stale or interpolated one.
            _tr = trans(arec, lane_level="agent", node_id=f"tr-agent-{i}",
                        label=short, start_ts=st_start.ts, end_ts=end_ts)
            if _par:
                _gid, _rank, _size = _par
                _tr.parallel_group_id = _gid
                _tr.parallel_rank     = _rank
                _tr.parallel_size     = _size
            elems.append(_tr)
            _agent_output = arec.get("output", "")
            # Suppress bookkeeping strings that carry no semantic content.
            if _agent_output.startswith("Completed tool calls"):
                _agent_output = ""
            # Use "agent_end" level (priority 4 > call=3) so the agent's
            # returned output is the authoritative content for the state box
            # at this boundary — not a call-layer own_out/next_inp value.
            _end_level = "agent_end" if _agent_output else "agent"
            st_end = state(end_ts, _agent_output, _end_level)
            st_end.hover_by_lane["agents"] = _agent_output
            # Mark the state as interrupted only when the agent genuinely ran to
            # the trace cut-off (end_ts ≈ t_max).  Pre-delegation fragments of
            # the same parent record also carry _end_missing=True but have their
            # own explicit end_ts (child start), so they must NOT be flagged.
            if arec.get("_end_missing") and abs(end_ts - t_max) <= _TS_TOL:
                st_end.is_interrupted = True
            elif arec.get("_exec_status"):
                st_end.is_error = True
            if _par:
                _gid, _rank, _size = _par
                if _rank == _size - 1:
                    # Join state: last branch ends at the fork/join boundary
                    st_end.is_join           = True
                    st_end.parallel_group_id = _gid
                    st_end.parallel_size     = _size
            elems.append(st_end)
        if not (isinstance(elems[-1], StateNode)
                and abs(elems[-1].ts - t_max) <= _TS_TOL):
            # The DFS-last agent fragment isn't necessarily the real-time-last
            # one — e.g. a fork's earlier-ranked sibling can run long past the
            # point DFS order puts it, so this lane's own final element can
            # land well before t_max. A bare append here would leave two
            # consecutive StateNodes with no Trans between them (the same
            # class of bug rank>0 branches hit above); bridge it instead.
            _bridge_to_state(elems, t_max, s_out, "session")
        agent_lane.sequence = elems
        lanes.append(agent_lane)

    # ── Call lane: direct call children of each agent fragment ───────────────
    if call_sequence:
        call_lane = LaneDef("calls", "call", "Calls")
        elems = [state(t_min, s_in, "session")]
        for j, crec in enumerate(call_sequence):
            ct_j   = crec.get("call_type", "")
            end_ts = crec["end_ts"] if crec.get("end_ts", 0) > 0 else t_max
            # Every call type (including ProcessingCall) gets its own state pair.
            # ProcessingCall is now a real bar so the prompt-assembly step is
            # visible as a distinct chain node (S_k → ⚙prompt → S_{k+1} → llm → …).
            # Truly instant calls (ToolCall/MemoryCall/RAGQuery with duration≈0)
            # use is_instant=True and are rendered as icon badges by the HTML renderer.
            st_start = state(crec["start_ts"], crec.get("input", ""), "call")
            st_start.hover_by_lane["calls"] = crec.get("input", "")
            _tr = trans(crec, lane_level="call", node_id=f"tr-call-{j}",
                        end_ts=end_ts)
            # Context provenance (the "N parts · M tokens · System Prompt ×… ,
            # Context ×…" breakdown) describes the *assembled context* — a state,
            # not the LLM action.  Attach it to the LLM's start state (the node
            # between ⚙ context and the LLM bar) and leave the LLM bar to
            # represent the call itself.
            if ct_j == "LLMCall" and _tr.cpr_data:
                if not st_start.cpr_data:
                    st_start.cpr_data = _tr.cpr_data
                    st_start.model = _tr.model
                _tr.cpr_data = []
            elems.append(_tr)
            own_out  = crec.get("output", "")
            # ProcessingCall outputs are operation summaries ("3 injections · 51
            # tok") — they belong on the transition bar, not on the following
            # state.  Clear them so the state can show cumulative context instead.
            if ct_j == "ProcessingCall":
                own_out = ""
            next_rec  = call_sequence[j + 1] if j + 1 < len(call_sequence) else None
            next_inp  = next_rec.get("input", "") if next_rec else ""
            end_hover = own_out or next_inp
            # A tool-turn LLM call emits no text (the model returned a tool
            # call, captured as the next bar), so its end state would be empty.
            # Show what the call led to instead of a bare "(no content)".
            if not end_hover and next_rec:
                _nl = next_rec.get("label") or _PROC_TYPE_LABEL.get(
                    next_rec.get("call_type", ""), next_rec.get("call_type", "")
                )
                if _nl:
                    end_hover = f"→ {_nl}"
            st_end = state(end_ts, end_hover, "call")
            st_end.hover_by_lane["calls"] = end_hover
            elems.append(st_end)

        # ── Propagate the context breakdown to EVERY state ──────────────
        # Each state is the working memory at that instant.  Every action's
        # result enters working memory (appended to the end of context for
        # visualization), so the assembled context grows monotonically along the
        # lane.  Each LLM call's context snapshot already sits on its start state
        # (see the CPR move above); carry the latest snapshot forward to the
        # states that follow it, and back-fill the states before the first
        # snapshot with that first one, so no state is left without a breakdown.
        _model_for_states: str = ""
        for el in elems:
            if isinstance(el, TransNode) and el.call_type == "LLMCall" and el.model:
                _model_for_states = el.model
                break
        _latest_cpr: list[dict] = []
        for el in elems:
            if isinstance(el, StateNode):
                if el.cpr_data:
                    _latest_cpr = el.cpr_data
                elif _latest_cpr:
                    el.cpr_data = list(_latest_cpr)
                    if not el.model:
                        el.model = _model_for_states
        # Back-fill leading states (before the first snapshot).
        _first_cpr: list[dict] = next(
            (el.cpr_data for el in elems if isinstance(el, StateNode) and el.cpr_data),
            [],
        )
        if _first_cpr:
            for el in elems:
                if isinstance(el, StateNode):
                    if el.cpr_data:
                        break
                    el.cpr_data = list(_first_cpr)
                    if not el.model:
                        el.model = _model_for_states

        # NOTE: agent-boundary transitions in this lane no longer need their
        # own marker — the delegation tool call is now a visible node here
        # (see tree.py's _reserve_marker_width) and the DFS branch-reset
        # separator (is_branch_reset, stamped in the finalization pass below)
        # already marks every "enter a new sibling" boundary. A "return to
        # the parent and continue" transition (delegate's last call ->
        # delegating agent's next call) isn't itself entering a new branch,
        # so it renders plainly — the Agents lane already shows which agent
        # owns which span.

        call_lane.sequence = elems
        lanes.append(call_lane)

        # Inject agent-boundary timestamps that are not yet present in the call
        # lane.  This ensures delegation splits and fragment boundaries are
        # visible in the Calls lane and that cross-lane connector lines appear
        # at every shared state (not just call-end timestamps).
        call_ts: set[float] = {el.ts for el in elems if isinstance(el, StateNode)}
        for arec in agent_sequence:
            for bts in (arec["start_ts"], arec["end_ts"]):
                # Skip if a call state already sits at (or within a hair of) this
                # boundary.  Agent execution boundaries can fall a millisecond off
                # the nearest call boundary (e.g. execution_end fires just before
                # the final llm_call_end); injecting a near-duplicate state there
                # creates an empty connector column between two states with no
                # transition (the "S9 → S10 with nothing between" glitch).
                if any(abs(bts - ex) <= _TS_TOL for ex in call_ts):
                    continue
                injected = state(bts)
                call_lane.connector_only_ts.add(bts)
                call_lane.sequence.append(injected)
                call_ts.add(bts)

        # Blocked-action ghost markers: a BLOCK/TERMINATE/SKIP/BLACKLIST
        # decision stops the call before the engine ever runs it, so there is
        # no execution record to hang a bar on. WHOSE ghost this is (agent_id,
        # decision, reason, policy — see governance.py's _collect_blocked_actions)
        # is already fully known from the governance_decision event's own real
        # ids; the tolerance search below only decides which EXISTING visual
        # state to hang that already-identified ghost's hover/badge on, in the
        # absence of any execution record to attach it to directly — it never
        # guesses which call was blocked. Injects a connector-only marker
        # state when nothing already sits at that boundary — same technique
        # used above for agent-boundary timestamps.
        for ghost in blocked:
            gts = ghost["ts"]
            close_ts = next((ex for ex in call_ts if abs(gts - ex) <= _TS_TOL), None)
            if close_ts is not None:
                gstate = state_reg[close_ts]
            else:
                gstate = state(gts)
                call_lane.connector_only_ts.add(gts)
                call_lane.sequence.append(gstate)
                call_ts.add(gts)
            gstate.governance.append({
                "decision":   ghost["decision"],
                "reason":     ghost["reason"],
                "policyName": ghost["policyName"],
            })

    # ── Thinking lane: one ThinkingCall per LLM call that has thinking content
    thinking_records = _synthesize_thinking_records(records) if "thinking" in facets else []
    thinking_sub_tss: list[float] = []   # intermediate state ts values (for letter labels)

    # Build a map from LLM call_id → staggered start_ts so the thinking lane
    # anchors at the correct (post-stagger) timestamp.
    _staggered_llm_starts: dict[str, float] = {
        r["call_id"]: r["start_ts"]
        for r in call_sequence
        if r.get("call_type") == "LLMCall"
    }

    if thinking_records:
        thinking_lane = LaneDef("thinking", "thinking", "Thinking")
        telems: list = []
        prev_ts: float | None = None
        prev_agent_id: str | None = None

        for k, trec in enumerate(sorted(thinking_records, key=lambda r: r["start_ts"])):
            # Use staggered LLM call start_ts (parent_call_id points to the
            # LLMCall) so the thinking lane aligns with the call lane visually.
            _parent_cid = trec.get("parent_call_id", "")
            ts_start = _staggered_llm_starts.get(_parent_cid, trec["start_ts"])
            # Adjust thinking end proportionally to the staggered start.
            _orig_start = trec["start_ts"]
            _offset = ts_start - _orig_start
            ts_think_end = (trec["end_ts"] if trec.get("end_ts", 0) > 0 else t_max) + _offset
            ts_llm_end   = (trec.get("_llm_end_ts") or t_max)

            if prev_ts is None:
                # First thinking record — lane anchors at the LLM call (ts_start)
                _st_anchor = state(ts_start, trec.get("input", ""), "call")
                _st_anchor.hover_by_lane["thinking"] = trec.get("input", "")
                telems.append(_st_anchor)
            elif ts_start > prev_ts + _TS_TOL:
                _is_inter_agent = (prev_agent_id is not None and prev_agent_id != trec["agent_id"])
                # Add the anchor state; mark as laneRestart when agents differ
                # (breaks lifeline — thinking is per-agent, not continuous).
                # For same-agent gaps the lifeline connects them naturally.
                _st_gap_end = state(ts_start, trec.get("input", ""), "call")
                _st_gap_end.hover_by_lane["thinking"] = trec.get("input", "")
                if _is_inter_agent:
                    _st_gap_end.is_lane_restart = True
                telems.append(_st_gap_end)
            else:
                # The boundary state at ts_start already exists in state_reg
                # (registered by the call lane). Update its thinking hover and
                # append it to telems so the thinking lane renders a box there.
                _anch = state(ts_start)
                _anch.hover_by_lane["thinking"] = trec.get("input", "")
                telems.append(_anch)

            # ThinkingCall transition: from LLM call start → thinking end (85%)
            telems.append(trans(trec, lane_level="thinking",
                                node_id=f"tr-think-{k}",
                                call_type="ThinkingCall", label="Think",
                                start_ts=ts_start, end_ts=ts_think_end))
            think_st = state(ts_think_end, trec.get("output", ""), "thinking")
            think_st.hover_by_lane["thinking"] = trec.get("output", "")
            telems.append(think_st)
            thinking_sub_tss.append(ts_think_end)   # mark as letter-suffix candidate

            # Output passthrough: thinking end → LLM call end (remaining 15%)
            # Rendered as ThinkingEmit (indigo-900, label "Emit") so users can
            # distinguish it from the idle gap before thinking starts.
            if ts_llm_end > ts_think_end + _TS_TOL:
                emit_rec: dict = {"input": trec.get("output", ""), "output": trec.get("llm_output", ""), "agent_id": trec["agent_id"]}
                telems.append(trans(emit_rec, lane_level="thinking",
                                    node_id=f"tr-think-out-{k}",
                                    call_type="ThinkingEmit", label="Emit",
                                    start_ts=ts_think_end, end_ts=ts_llm_end))
                telems.append(state(ts_llm_end))
                prev_ts = ts_llm_end
            else:
                prev_ts = ts_think_end
            prev_agent_id = trec["agent_id"]

        # Lane ends at the state after the last Emit — no leading gap from t_min,
        # no trailing gap to t_max.

        thinking_lane.sequence = telems
        lanes.append(thinking_lane)

    # ── Governance lane: one bar per governed call + one instant marker per
    # blocked action, all reusing the exact state boundaries already
    # registered by the calls lane — so this lane adds no new timestamps of
    # its own and cannot violate the shared-state alignment other lanes rely
    # on. Hidden by default in the renderer to avoid clutter; toggled on to
    # see governance impact consolidated in one place instead of per-call
    # badges. Built from the same facet data as the Phase 2 badges/markers.
    _gov_intervals: list[tuple[float, float, str, str, list[dict]]] = []
    for _gcid, _gdata in gov_map.items():
        _grec = rec_by_id.get(_gcid)
        if _grec is None:
            continue
        _gdecision, _gct, _ = _governance_severity(_gdata)
        _gov_intervals.append((_grec["start_ts"], _grec["end_ts"], _gct, _gdecision, _gdata))
    for _ghost in blocked:
        _gov_intervals.append((_ghost["ts"], _ghost["ts"], "GovernanceBlock", "BLOCK", [{
            "hook": "egress", "checkpoint": "after",
            "decision": _ghost["decision"], "reason": _ghost["reason"],
            "policyName": _ghost["policyName"],
        }]))

    if _gov_intervals:
        _gov_intervals.sort(key=lambda iv: iv[0])
        gov_lane = LaneDef("governance", "governance", "Governance")
        gelems: list = []
        _gov_cursor = -1.0
        for _gi, (_gs, _ge, _gct, _glabel, _gdata) in enumerate(_gov_intervals):
            if _gs < _gov_cursor - _TS_TOL:
                # Overlaps the previous bar — skip rather than corrupt ordering.
                # _align_record_boundaries (tree.py) sometimes widens a call's
                # rendered window to fill a visual gap to its neighbour; when
                # that widened window swallows an earlier ghost's timestamp,
                # the ghost is dropped from this lane only — it still renders
                # on its own Calls-lane state boundary (see the block above).
                continue
            if not gelems:
                gelems.append(state(_gs))
            gelems.append(trans(
                {"agent_id": "", "input": "", "output": "", "call_id": f"gov-{_gi}"},
                lane_level="governance", node_id=f"tr-gov-{_gi}",
                call_type=_gct, label=_glabel,
                start_ts=_gs, end_ts=_ge,
            ))
            gelems[-1].governance = _gdata
            gelems[-1].is_instant = abs(_ge - _gs) <= _TS_TOL
            gelems.append(state(_ge))
            _gov_cursor = _ge
        if gelems:
            gov_lane.sequence = gelems
            lanes.append(gov_lane)

    # ── DFS virtual-position axis: stamp dfs_pos/branch_id onto every node ──
    # _ts_to_dfs_pos (from _assign_dfs_positions) only covers timestamps that
    # appear in call_sequence (the Calls lane) — extend it to every timestamp
    # in state_reg (session/MAS boundaries, HITL gaps, etc. share most of the
    # same timestamps via _align_record_boundaries, but not necessarily all)
    # by interpolating between the nearest covered neighbors in real time.
    _all_ts = sorted(state_reg.keys())
    _covered = sorted(_ts_to_dfs_pos)
    if _covered:
        for _ts in _all_ts:
            if _ts in _ts_to_dfs_pos:
                continue
            _lo = next((c for c in reversed(_covered) if c <= _ts), None)
            _hi = next((c for c in _covered if c >= _ts), None)
            if _lo is None:
                _ts_to_dfs_pos[_ts] = _ts_to_dfs_pos[_covered[0]] - (_covered[0] - _ts)
            elif _hi is None:
                _ts_to_dfs_pos[_ts] = _ts_to_dfs_pos[_covered[-1]] + (_ts - _covered[-1])
            elif _lo == _hi:
                _ts_to_dfs_pos[_ts] = _ts_to_dfs_pos[_lo]
            else:
                _frac = (_ts - _lo) / (_hi - _lo)
                _ts_to_dfs_pos[_ts] = (
                    _ts_to_dfs_pos[_lo] + _frac * (_ts_to_dfs_pos[_hi] - _ts_to_dfs_pos[_lo])
                )
    else:
        _ts_to_dfs_pos = {_ts: float(_i) for _i, _ts in enumerate(_all_ts)}

    # branch_id: the real call_id of the branch's own child (see
    # _detect_delegation_forks — "Fork branch node id = children ID", not a
    # synthetic counter), stamped on the reset boundary itself. _reset_branch_id
    # (from _assign_dfs_positions) is keyed by the ts the reset actually landed
    # on — strictly after the delegating tool call's own marker, not at it.
    for _ts, _node in state_reg.items():
        _node.dfs_pos = _ts_to_dfs_pos.get(_ts, 0.0)
        _node.is_branch_reset = _ts in _reset_branch_id
        if _ts in _reset_branch_id:
            _node.branch_id = _reset_branch_id[_ts]

    for _lane in lanes:
        for _el in _lane.sequence:
            if isinstance(_el, TransNode):
                _el.dfs_pos_start = _ts_to_dfs_pos.get(_el.start_ts, 0.0)
                _el.dfs_pos_end   = _ts_to_dfs_pos.get(_el.end_ts, 0.0)
                if _el.start_ts in _reset_branch_id:
                    _el.branch_id = _reset_branch_id[_el.start_ts]

    # ── Assign letter-suffix labels to thinking sub-states (S2 → S2a, S2b …)
    if thinking_sub_tss:
        from collections import defaultdict as _defdict
        all_bkts_tmp = sorted(state_reg.keys())
        _state_num_tmp = {b: i + 1 for i, b in enumerate(all_bkts_tmp)}
        # Group sub-states by their preceding numbered state
        _pred_to_subs: dict[int, list[float]] = _defdict(list)
        for _ts in sorted(thinking_sub_tss):
            _preds = [b for b in all_bkts_tmp if b < _ts - _TS_TOL]
            if _preds:
                _pn = _state_num_tmp[max(_preds)]
                _pred_to_subs[_pn].append(_ts)
        for _pn, _subs in _pred_to_subs.items():
            for _idx, _ts in enumerate(sorted(_subs)):
                state_reg[_ts].label_override = f"S{_pn}{chr(ord('a') + _idx)}"

    # ── Reassign sequential IDs globally (DFS order, level as tiebreaker)
    _LEVEL_ORDER = {"session": 0, "mas": 1, "agent": 2, "call": 3, "thinking": 4, "governance": 5}
    all_trans = [el for lane in lanes for el in lane.sequence
                 if isinstance(el, TransNode)]
    all_trans.sort(key=lambda t: (t.dfs_pos_start, _LEVEL_ORDER.get(t.level, 9)))
    for i, t in enumerate(all_trans):
        t.seq = i + 1

    # ── Structural sanity check ──────────────────────────────────────────
    validate_trajectory_dag(state_reg, lanes)

    return state_reg, lanes
