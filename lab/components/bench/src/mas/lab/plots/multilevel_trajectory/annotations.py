#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Annotation and context-provenance hover enrichment."""

from collections import defaultdict
from typing import Optional

from mas.lab.plots.multilevel_trajectory.constants import _TS_TOL

# State hover priority: the most specific (deepest) level's output wins when
# multiple transitions end at the same timestamp.
# ``agent_end`` (=4) is used at agent-result boundaries so the agent's
# authoritative output takes precedence over call-layer bookkeeping values
# (e.g. ``next_inp`` of the next call that happens to share the same ts).
_HOVER_PRIORITY: dict[str, int] = {
    "session":   0,
    "mas":       1,
    "agent":     2,
    "call":      3,
    "agent_end": 4,
}


# ---------------------------------------------------------------------------
# Annotation collection — L2/L3 XAI context attached to hover text
# ---------------------------------------------------------------------------

_ANNOTATION_KINDS: frozenset[str] = frozenset({
    "routing",
    "routing_result",
    "context_assembled",
    "governance_authorize_start",
    "governance_authorize_end",
    "governance_validate_start",
    "governance_validate_end",
    "obs_wrap_gov_authorize_start",
    "obs_wrap_gov_authorize_end",
    "obs_wrap_gov_validate_start",
    "obs_wrap_gov_validate_end",
    "state_update_start",
    "state_update_end",
    "agent_communication_start",
    "agent_communication_end",
})

_ANNOTATION_LABEL: dict[str, str] = {
    "routing":                        "→ routing",
    "routing_result":                 "→ routed",
    "context_assembled":              "ctx",
    "state_update_start":             "state↑",
    "state_update_end":               "state↓",
    "agent_communication_start":      "agent-remote↑",
    "agent_communication_end":        "agent-remote↓",
    # Governance: shown as ⛨ prefix/suffix on the enclosing call hover — not
    # rendered as a separate bar so the calls lane stays clean.
    "governance_authorize_start":     "⛨ gov-auth start",
    "governance_authorize_end":       "⛨ gov-auth end",
    "governance_validate_start":      "⛨ gov-val start",
    "governance_validate_end":        "⛨ gov-val end",
    "obs_wrap_gov_authorize_start":   "⛨ gov-auth",
    "obs_wrap_gov_authorize_end":     "⛨ gov-auth ✓",
    "obs_wrap_gov_validate_start":    "⛨ gov-val",
    "obs_wrap_gov_validate_end":      "⛨ gov-val ✓",
}


def _collect_annotations(
    events:  list[dict],
    records: list[dict],
) -> dict[str, list[str]]:
    """Return ``call_id \u2192 [annotation_summary_line, ...]`` for L2/L3 XAI hover enrichment.

    Annotation events are matched to the call record they annotate by a real
    id, never timestamp containment. Two shapes, both already resolved by the
    runtime (see ObservabilityOperator._resolve_transition_ids):

    - governance_authorize/validate_*, obs_wrap_gov_*, and context_assembled
      are recorded AT the call they annotate \u2014 the event's own ``call_id`` IS
      that call's ``call_id``.
    - state_update_start/end is a CHILD of the call whose result it records
      (record_context_mutation's op threading) \u2014 its own ``call_id`` is a
      distinct, self-paired synthetic id, so the real link is its
      ``parent_call_id``.

    Events with neither a matching ``call_id`` nor ``parent_call_id`` (e.g.
    turn/session-scoped state_update, or an annotation kind with no producer
    at all \u2014 routing/agent_communication today) are silently skipped: there is
    no real id to attach them to, and none is guessed.
    """
    ann_events = [e for e in events if e.get("kind") in _ANNOTATION_KINDS]
    if not ann_events:
        return {}

    rec_by_id = {r["call_id"]: r for r in records}
    result: dict[str, list[str]] = defaultdict(list)

    for ann in ann_events:
        kind = ann.get("kind", "")

        _own_id = ann.get("call_id")
        best_rec: Optional[dict] = rec_by_id.get(_own_id) if _own_id in rec_by_id else None
        if best_rec is None:
            _parent_id = ann.get("parent_call_id")
            best_rec = rec_by_id.get(_parent_id) if _parent_id else None

        if best_rec is None:
            continue

        label = _ANNOTATION_LABEL.get(kind, kind)
        parts = [label]
        # Extract the most meaningful payload field for the short summary
        for _f in ("target_agent_id", "target", "to", "task"):
            val = ann.get(_f)
            if val:
                parts.append(str(val))
                break
        if kind == "context_assembled":
            segs   = ann.get("segments")
            tokens = ann.get("total_tokens")
            if segs   is not None:
                parts.append(f"{segs} seg")
            if tokens is not None:
                parts.append(f"{tokens} tok")
        if kind in ("state_update_start", "state_update_end"):
            action = ann.get("update_type")
            if action:
                parts.append(str(action))
            preview = ann.get("content_preview")
            if preview:
                parts.append(str(preview)[:40])
        result[best_rec["call_id"]].append(" ".join(parts))

    return dict(result)


# ── Mechanism colour badges ──────────────────────────────────────────────

_MECH_BADGE: dict[str, str] = {
    "inject":    "🔵",
    "rag":       "🟣",
    "tool_call": "🟠",
}

_CAUSE_BADGE_MAP: dict[str, str] = {
    "deterministic": "⚙",
    "stochastic":    "🎲",
    "explicit":      "👤",
}


def _source_category(source: str, mechanism: str = "inject") -> str:
    """Derive a human-readable category from a CPR source name.

    Category is determined by the **source** (where the data comes from),
    not the mechanism (how it was retrieved).  RAG is a mechanism, not a source.
    """
    sl = source.lower()
    if sl.startswith("memory:") or sl.startswith("mem:"):
        return "MEMORY"
    if "skill" in sl or sl.startswith("facet:skill"):
        return "SKILL"
    if sl.startswith("context/role"):
        return "SYSTEM"
    if sl.startswith("context/intent"):
        return "SYSTEM"
    if sl.startswith("context/"):
        return "SYSTEM"
    if sl.startswith("tool:") or sl.startswith("tool_result"):
        return "TOOL"
    if mechanism == "tool_call":
        return "TOOL"
    return "CONTEXT"


def _collect_context_provenance(
    events:  list[dict],
    records: list[dict],
) -> dict[str, list[dict]]:
    """Return ``call_id → [cpr_event, …]`` for L4 context provenance hover enrichment.

    ``context_part_contributed`` events are matched to LLM call records by a
    direct, real id match on ``llm_call_id`` — the runtime now resolves this
    to the SAME call_id the LLM call's own llm_call_start/end use (see
    ObservabilityOperator.record_context_assembled's op="LLM_CALL" threading
    and _resolve_transition_ids' CONTEXT_ASSEMBLED branch), never a synthetic
    placeholder needing timestamp-proximity reconstruction.
    """
    cpr_events = [e for e in events if e.get("kind") == "context_part_contributed"]
    if not cpr_events:
        return {}

    llm_records_by_id = {
        r["call_id"]: r for r in records if r.get("call_type") == "LLMCall"
    }

    result: dict[str, list[dict]] = defaultdict(list)

    for ev in cpr_events:
        raw_cid = ev.get("llm_call_id") or ""
        matched = llm_records_by_id.get(raw_cid) if raw_cid else None
        if matched:
            result[matched["call_id"]].append(ev)

    return dict(result)


def _format_cpr_hover(cpr_parts: list[dict]) -> str:
    """Format context provenance parts into a readable hover-text block.

    Shows each context part with its source, mechanism badge, and the actual
    content text (up to 300 chars per part).
    """
    if not cpr_parts:
        return ""

    lines: list[str] = ["\n\n📦 Context Assembly:"]
    total_tokens = 0
    for part in cpr_parts:
        source = part.get("source", "?")
        mechanism = part.get("mechanism", "inject")
        cause_type = part.get("cause_type", "deterministic")
        tokens = part.get("token_estimate", 0)
        retained = part.get("retained", True)
        content = part.get("content") or part.get("content_preview") or ""
        placement = part.get("placement", "")
        mech_badge = _MECH_BADGE.get(mechanism, "⚪")
        cause_badge = _CAUSE_BADGE_MAP.get(cause_type, "")

        # Source category for clearer labeling
        cat = _source_category(source, mechanism)
        status = "" if retained else " ❌evicted"

        lines.append(f"\n  {mech_badge} [{cat}] {source} {cause_badge} {tokens}tok{status}")
        if placement:
            lines.append(f"     placement: {placement}")

        # Show actual content — up to 300 chars
        if content:
            preview = content[:300].replace("\n", "\n     ")
            if len(content) > 300:
                preview += "…"
            lines.append(f"     ───\n     {preview}")
        else:
            lines.append("     (no content captured)")

        total_tokens += tokens

    lines.append(f"\n  ── {len(cpr_parts)} parts · {total_tokens} tokens total")
    return "\n".join(lines)


def _stagger_coinc_processing_calls(seq: list[dict]) -> list[dict]:
    """Stagger coincident point-in-time ProcessingCall records with small offsets.

    When the runtime emits multiple per-actor ProcessingCall spans at the same
    wall-clock instant (start_ts == end_ts == ts_now), they would share the
    same StateNode boundaries and render as overlapping bars.  This helper
    gives each coincident ProcessingCall a tiny sequential offset so that each
    occupies its own visual slot in the Calls lane:

        PC1: (ts, ts+δ), PC2: (ts+δ, ts+2δ), …

    The stagger increment (1ms) is well below human perception but large
    enough to create distinct StateNode keys.
    """
    _STAGGER_DUR = 0.001  # 1ms per processing call
    if not seq:
        return seq
    result: list[dict] = []
    i = 0
    while i < len(seq):
        crec = seq[i]
        # Only stagger point-in-time ProcessingCalls (start ≈ end).
        if (
            crec.get("call_type") == "ProcessingCall"
            and abs(crec["end_ts"] - crec["start_ts"]) <= _TS_TOL
        ):
            ts    = crec["start_ts"]
            group: list[dict] = [crec]
            j     = i + 1
            while j < len(seq):
                nxt = seq[j]
                if (
                    nxt.get("call_type") == "ProcessingCall"
                    and abs(nxt["start_ts"] - ts) <= _TS_TOL
                    and abs(nxt["end_ts"]   - ts) <= _TS_TOL
                ):
                    group.append(nxt)
                    j += 1
                else:
                    break
            # A lone point-in-time ProcessingCall (e.g. a synthesized context
            # -assembly node before an LLM call) needs no staggering — leave its
            # boundaries intact so it keeps sharing the state with the following
            # call.  Rewriting its end to ts+δ here would break that shared
            # boundary and open an empty connector gap before the LLM.
            if len(group) < 2:
                result.append(crec)
                i += 1
                continue
            # Stagger each member with a small time offset.
            for idx, rec in enumerate(group):
                staggered = dict(rec)
                staggered["start_ts"] = ts + idx * _STAGGER_DUR
                staggered["end_ts"]   = ts + (idx + 1) * _STAGGER_DUR
                result.append(staggered)
            # Snap the next record's start_ts forward if it falls inside the
            # staggered group.  Only advance — never move it backward — and
            # shift end_ts by the same delta so a real (non-zero) duration is
            # never corrupted (e.g. shrunk, or inverted into end_ts <
            # start_ts) — the same rule _reserve_marker_width in tree.py
            # applies for its own analogous forward-bump.
            _group_end = ts + len(group) * _STAGGER_DUR
            if j < len(seq):
                nxt_copy = dict(seq[j])
                _delta = _group_end - nxt_copy.get("start_ts", _group_end)
                if _delta > 0:
                    nxt_copy["start_ts"] = _group_end
                    nxt_copy["end_ts"] = nxt_copy.get("end_ts", _group_end) + _delta
                result.append(nxt_copy)
                i = j + 1
            else:
                i = j
        else:
            result.append(crec)
            i += 1
    return result
