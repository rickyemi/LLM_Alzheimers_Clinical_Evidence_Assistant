import json
from types import SimpleNamespace

from ad_rag.agent import ClaudeToolAgent, ToolContext, ToolRegistry, build_agent, default_tools
from ad_rag.monitoring import AuditLog


def scenarios(settings):
    return [json.loads(l) for l in (settings.root / "data/eval/agent_scenarios.jsonl").read_text().splitlines()]


def test_offline_agent_tool_selection_and_escalation(pipeline, settings):
    for sc in scenarios(settings):
        agent = build_agent(settings, pipeline)
        res = agent.run(sc["task"])
        assert res.tools_used == sc["expected_tools"], (sc["id"], res.tools_used)
        assert res.escalated == sc["expect_escalation"], sc["id"]


def test_agent_answer_is_cited_and_numeric_logic_is_deterministic(pipeline, settings):
    res = build_agent(settings, pipeline).run("The plasma assay ratio was 0.45. How should an intermediate result be interpreted?")
    assert res.citations_ok and "intermediate" in res.final_answer.lower()
    interp = next(s for s in res.steps if s["tool"] == "interpret_biomarker")
    assert interp["observation"] == "band=intermediate"


def test_write_tool_denied_without_approval_and_allowed_with_it(pipeline, settings):
    task = "Is blood testing appropriate for a symptomatic adult evaluated at a memory clinic?"
    denied = build_agent(settings, pipeline).run(task, save_brief=True)
    assert not next(s for s in denied.steps if s["tool"] == "save_evidence_brief")["ok"]
    allowed = build_agent(settings, pipeline, approver=lambda t, a: True).run(task, save_brief=True)
    assert next(s for s in allowed.steps if s["tool"] == "save_evidence_brief")["ok"]


def test_registry_enforces_allowlist_validation_and_call_cap(pipeline, settings):
    reg = ToolRegistry(default_tools(), ToolContext(settings, pipeline), max_calls_per_tool=1)
    assert not reg.execute("rm_rf_everything", {})["ok"]                    # unknown tool
    assert not reg.execute("interpret_biomarker", {"value": -1})["ok"]      # schema violation
    assert reg.execute("interpret_biomarker", {"value": 0.1})["ok"]
    assert "cap" in reg.execute("interpret_biomarker", {"value": 0.2})["error"].lower()


def test_biomarker_bands(pipeline, settings):
    reg = ToolRegistry(default_tools(), ToolContext(settings, pipeline), 10)
    assert reg.execute("interpret_biomarker", {"value": 0.1})["band"] == "below_lower_cutoff"
    assert reg.execute("interpret_biomarker", {"value": 0.45})["band"] == "intermediate"
    assert reg.execute("interpret_biomarker", {"value": 0.60})["band"] == "above_upper_cutoff"


def test_tool_calls_are_audited_and_chain_is_intact(pipeline, settings):
    build_agent(settings, pipeline).run("Are there recruiting clinical trials for Alzheimer's disease?")
    ok, n, _ = AuditLog.verify(settings.path("audit_log"))
    assert ok and n > 0


# ---- Claude driver, tested with a scripted stub client (no network, no key) ----
class StubClient:
    """Returns a tool_use turn, then a final text turn, mimicking the Messages API shape."""
    def __init__(self):
        self.turn = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        if self.turn == 0:
            self.turn += 1
            blk = SimpleNamespace(type="tool_use", id="tu_1", name="search_knowledge_base",
                                  input={"query": "two cutoffs", "k": 3})
            return SimpleNamespace(content=[blk], stop_reason="tool_use", usage=usage)
        txt = SimpleNamespace(type="text", text="Two cutoffs reduce misclassification. [source: two_cutoff_interpretation.md]")
        return SimpleNamespace(content=[txt], stop_reason="end_turn", usage=usage)


def test_claude_agent_loop_with_stub(pipeline, settings):
    s = settings.model_copy(deep=True); s.llm.backend = "anthropic"
    agent = build_agent(s, pipeline, client=StubClient())
    assert isinstance(agent, ClaudeToolAgent)
    res = agent.run("Why two cutoffs?")
    assert res.tools_used == ["search_knowledge_base"] and res.citations_ok and res.stop_reason == "completed"
    assert res.tokens == {"input": 20, "output": 10}


def test_claude_agent_stops_at_step_cap(pipeline, settings):
    class Loop(StubClient):
        def _create(self, **kw):
            blk = SimpleNamespace(type="tool_use", id="x", name="search_knowledge_base", input={"query": "abc def"})
            return SimpleNamespace(content=[blk], stop_reason="tool_use",
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    s = settings.model_copy(deep=True); s.llm.backend = "anthropic"; s.agent.max_steps = 3
    res = build_agent(s, pipeline, client=Loop()).run("loop forever please")
    assert res.stop_reason == "max_steps" and res.escalated
