#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Kernel driver loop integration."""

from dataclasses import dataclass, field

from mas.runtime.boundary.obs.observability_plugin import ObservabilityPlugin
from mas.runtime.boundary.obs.transition import TransitionEvent
from mas.runtime.driver.instance import RuntimeInstance
from mas.runtime.engine.simulated import SimMode, SimulatedEngine
from mas.runtime.kernel.config import KernelConfig
from mas.runtime.schema.ingress import EngineIoReturn, LifecyclePause, LifecycleResume


@dataclass
class _CapturePlugin(ObservabilityPlugin):
    seen: list[TransitionEvent] = field(default_factory=list)

    def on_transition(self, event: TransitionEvent) -> None:
        self.seen.append(event)


def test_runtime_instance_run_user_text_produces_client_response():
    engine = SimulatedEngine(
        script={
            1: EngineIoReturn(
                correlation_id=1,
                response_kind="MODEL_TEXT",
                next_step="STOP",
                text="hello back",
            )
        }
    )
    inst = RuntimeInstance.from_parts(engine=engine)
    inst.capture_session_baseline()
    trace = inst.run_user_text("hello")
    assert trace.client_responses or trace.steps


def test_runtime_instance_pause_and_resume():
    inst = RuntimeInstance.from_parts()
    inst.capture_session_baseline()
    pause_trace = inst.pause(reason="operator")
    assert pause_trace is not None
    resume_trace = inst.resume()
    assert resume_trace is not None


def test_runtime_instance_reset_after_baseline():
    inst = RuntimeInstance.from_parts()
    inst.capture_session_baseline()
    inst.run_user_text("turn one")
    inst.reset_session()
    assert inst.kernel.q.dp in {"IDLE", "AWAITING_INGRESS", "EGRESS_PENDING", "EVALUATING"}


def test_driver_attaches_own_call_id_to_tool_call_invocation():
    """driver.py resolves this TOOL_CALL's own call_id (via
    ObservabilityOperator.call_id_for — already assigned by the
    CONTRACT_EXECUTE observability step just before the engine is invoked)
    and attaches it onto the InvokeEngineIo passed to engine.invoke(). This
    is the mechanism a delegate_to_X tool call relies on to forward its own
    identity as the delegate's caller_call_id (see InvokeEngineIo.call_id
    and execute_engine_tool) — no closure capture, no timestamp heuristics,
    a real field resolved once at the true origin."""
    seen_ios: list = []

    class _RecordingEngine(SimulatedEngine):
        def invoke(self, io):
            seen_ios.append(io)
            return super().invoke(io)

    engine = _RecordingEngine(
        llm_next_step=lambda cid: "TOOL_CALL" if cid == 1 else "STOP",
        llm_tool_intent=lambda cid: ("some_tool", {"x": 1}),
    )
    inst = RuntimeInstance.from_parts(engine=engine)
    inst.capture_session_baseline()
    inst.run_user_text("trigger a tool call")

    tool_call_ios = [io for io in seen_ios if io.op == "TOOL_CALL"]
    assert tool_call_ios, "expected at least one TOOL_CALL dispatch"
    assert all(io.call_id for io in tool_call_ios), tool_call_ios


def test_driver_records_real_tool_call_end_matching_start_call_id():
    """Regression: ObsEnvelopeMachine's OBSERVABILITY_POST_EXECUTE branch used
    to be permanently shadowed by an earlier elif matching the same symbol
    (see obs_envelope.py), so record_engine_io_return never fired for
    TOOL_CALL — tool_call_end was never emitted by any real trace, for any
    tool call. This drives a full kernel/driver turn end-to-end and asserts
    the tool call's own ENGINE_IO_RETURN now fires, with the SAME call_id its
    own ENGINE_IO start used."""
    engine = SimulatedEngine(
        llm_next_step=lambda cid: "TOOL_CALL" if cid == 1 else "STOP",
        llm_tool_intent=lambda cid: ("some_tool", {"x": 1}),
    )
    inst = RuntimeInstance.from_parts(engine=engine)
    cap = _CapturePlugin()
    inst.driver.observability.subscribe(cap)
    inst.capture_session_baseline()
    inst.run_user_text("trigger a tool call")

    starts = [e for e in cap.seen if e.contract_id == "tool" and e.phase == "start"]
    ends = [e for e in cap.seen if e.contract_id == "tool" and e.phase == "end"]
    assert starts, "expected at least one tool call start"
    assert ends, "expected at least one tool call end (tool_call_end) — was never emitted before the fix"
    assert {e.call_id for e in starts} == {e.call_id for e in ends}


def test_driver_llm_call_gets_exactly_one_end_event():
    """Regression guard: fixing the ObsEnvelopeMachine shadowing bug must not
    double-record llm_call_end. driver.py used to have its own hand-written
    special case (record_engine_llm_return) that fired for every LLM_CALL
    return, independent of the (permanently shadowed) envelope-machine path —
    once that path actually runs, both would fire unless the hand-written
    special case is removed."""
    inst = RuntimeInstance.from_parts()
    cap = _CapturePlugin()
    inst.driver.observability.subscribe(cap)
    inst.capture_session_baseline()
    inst.run_user_text("hello")

    llm_ends = [e for e in cap.seen if e.contract_id == "model" and e.phase == "end"]
    by_call_id: dict = {}
    for e in llm_ends:
        by_call_id.setdefault(e.call_id, 0)
        by_call_id[e.call_id] += 1
    duplicated = {cid: n for cid, n in by_call_id.items() if n > 1}
    assert not duplicated, duplicated


def test_driver_records_real_memory_call_end_matching_start_call_id():
    """Regression: apply_engine_io_return (ingress_step.py) only distinguished
    TOOL_CALL vs LLM_CALL — a MEMORY_OP return fell through to "LLM_CALL",
    resolving via a DIFFERENT (correlation_id, op)-keyed call_id than its own
    ENGINE_IO start used, permanently orphaning memory_call_end from
    memory_call_start. Drives a full turn through the react design pattern's
    memory-egress path (SimMode.MEMORY_PROBE's "DELEGATE" next_step)."""
    engine = SimulatedEngine(sim_mode=SimMode.MEMORY_PROBE)
    inst = RuntimeInstance.from_parts(
        engine=engine, config=KernelConfig(enable_memory_egress=True),
    )
    cap = _CapturePlugin()
    inst.driver.observability.subscribe(cap)
    inst.capture_session_baseline()
    inst.run_user_text("trigger a memory op")

    starts = [e for e in cap.seen if e.contract_id == "memory" and e.phase == "start"]
    ends = [e for e in cap.seen if e.contract_id == "memory" and e.phase == "end"]
    assert starts, "expected at least one memory op start"
    assert ends, "expected at least one memory_call_end — was misclassified as LLM_CALL before the fix"
    assert {e.call_id for e in starts} == {e.call_id for e in ends}
