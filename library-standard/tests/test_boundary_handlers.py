#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Native boundary-event handlers — one test per ObsEventKind dispatch.

None of these had a dedicated test before: they were only exercised
indirectly (if at all) through end-to-end plot/pipeline tests, which would
not catch a wrong dict key or a dropped field in any single handler.
"""

from __future__ import annotations

from mas.library.standard.lib.observability.native.boundary_handlers import dispatch_boundary
from mas.library.standard.lib.observability.native.transform import TransformContext


def _ctx() -> TransformContext:
    return TransformContext(agent_id="moderator", run_id="run-1")


def test_hitl_request_surfaces_policy_name_regression() -> None:
    """Regression: EMIT_HITL_REQUEST's policy_name was only ever set as the
    top-level ObservabilityEvent field, never duplicated into payload like
    every other emit site that needs to survive the live TransitionEvent
    pipeline (boundary_dict_from_transition only forwards payload/attributes)
    — so it silently dropped through that pipeline while governance_decision
    and boundary_error (already fixed) didn't."""
    record = {
        "kind": "hitl.request",
        "correlation_id": 5,
        "payload": {
            "question": "Proceed with booking?",
            "policy_name": "sample_governance",
            "pending_schedule": [],
            "offered_actions": [],
        },
    }
    out = dispatch_boundary(record, ctx=_ctx())
    assert len(out) == 1
    assert out[0]["kind"] == "hitl_gate"
    assert out[0]["question"] == "Proceed with booking?"
    assert out[0]["policy_name"] == "sample_governance"


def test_hitl_resolve_carries_resolution_and_answer() -> None:
    record = {
        "kind": "hitl.resolve",
        "correlation_id": 5,
        "payload": {"resolution": "approved", "answer": "yes, proceed"},
    }
    out = dispatch_boundary(record, ctx=_ctx())
    assert out == [{
        "kind": "hitl_resolve",
        "agent_id": "moderator", "run_id": "run-1", "correlation_id": 5,
        "timestamp": out[0]["timestamp"],
        "resolution": "approved", "answer": "yes, proceed",
    }]


def test_governance_decision_carries_reason_and_policy_name() -> None:
    record = {
        "kind": "governance.decision",
        "correlation_id": 2,
        "payload": {
            "hook": "egress", "checkpoint": "after", "decision": "BLOCK",
            "reason": "Restricted destination: Shadowmere", "policy_name": "forbidden-destination",
        },
    }
    out = dispatch_boundary(record, ctx=_ctx())
    assert len(out) == 1
    assert out[0]["kind"] == "governance_decision"
    assert out[0]["decision"] == "BLOCK"
    assert out[0]["reason"] == "Restricted destination: Shadowmere"
    assert out[0]["policy_name"] == "forbidden-destination"
    # checkpoint must reach the native trace: multilevel_trajectory/governance.py's
    # _collect_blocked_actions only recognizes a BLOCK as a ghost marker when
    # hook=="egress" and checkpoint=="after" — dropping this field here silently
    # made every real BLOCK/TERMINATE/SKIP/BLACKLIST decision invisible to the plot.
    assert out[0]["hook"] == "egress"
    assert out[0]["checkpoint"] == "after"


def test_governance_decision_with_no_decision_is_dropped() -> None:
    """The "before" checkpoint carries no decision yet (see GovEnvelopeMachine)
    — dispatch_boundary must not emit an empty/meaningless record for it."""
    record = {
        "kind": "governance.decision",
        "correlation_id": 2,
        "payload": {"hook": "egress", "checkpoint": "before", "decision": ""},
    }
    assert dispatch_boundary(record, ctx=_ctx()) == []


def test_boundary_error_carries_code_and_message() -> None:
    record = {
        "kind": "boundary.error",
        "correlation_id": 3,
        "payload": {
            "code": "RETRY_BUDGET_EXHAUSTED", "recoverable": False,
            "message": "max retries exceeded", "parent_call_id": "call-1",
        },
    }
    out = dispatch_boundary(record, ctx=_ctx())
    assert len(out) == 1
    assert out[0]["kind"] == "boundary_error"
    assert out[0]["code"] == "RETRY_BUDGET_EXHAUSTED"
    assert out[0]["recoverable"] is False
    assert out[0]["message"] == "max retries exceeded"
    assert out[0]["parent_call_id"] == "call-1"


def test_context_steer_carries_collect_id() -> None:
    record = {"kind": "context.steer", "correlation_id": 0, "payload": {"collect_id": "abc-123"}}
    out = dispatch_boundary(record, ctx=_ctx())
    assert out[0]["kind"] == "context_steer"
    assert out[0]["collect_id"] == "abc-123"


def test_boundary_egress_ingress_catch_alls() -> None:
    egress = dispatch_boundary(
        {"kind": "boundary.egress", "correlation_id": 0, "payload": {"egress_kind": "NO_OP"}},
        ctx=_ctx(),
    )
    assert egress[0]["kind"] == "boundary_egress"
    assert egress[0]["egress_kind"] == "NO_OP"

    ingress = dispatch_boundary(
        {"kind": "boundary.ingress", "correlation_id": 0, "payload": {"ingress_kind": "TOOL_RESULT"}},
        ctx=_ctx(),
    )
    assert ingress[0]["kind"] == "boundary_ingress"
    assert ingress[0]["ingress_kind"] == "TOOL_RESULT"


def test_unknown_kind_is_dropped() -> None:
    assert dispatch_boundary({"kind": "not.a.real.kind", "payload": {}}, ctx=_ctx()) == []


def test_dispatch_boundary_uses_real_timestamp_not_export_time() -> None:
    """Regression: dispatch_boundary used to unconditionally stamp every
    boundary-sourced record with `time.time()` at whatever moment the async
    export pipeline happened to process it, discarding the real occurrence-
    time timestamp boundary_dict_from_transition threads through as
    record["timestamp"] (TransitionEvent.timestamp, captured synchronously
    when the event actually fired). That export-time lag scrambled real
    occurrence order across concurrent per-agent async workers — e.g. an
    agent's own tool_call_end could appear to land after that same agent's
    own execution_end, even though it truly happened first."""
    record = {
        "kind": "engine.io",
        "correlation_id": 7,
        "call_id": "tool-abc",
        "timestamp": 12345.6789,
        "payload": {"op": "TOOL_CALL", "tool_name": "lookup_schedule"},
    }
    out = dispatch_boundary(record, ctx=_ctx())
    assert out
    assert all(rec["timestamp"] == 12345.6789 for rec in out), out
