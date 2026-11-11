import json

from ad_rag.monitoring import AuditLog, DriftMonitor, MetricsRegistry


def test_audit_chain_verifies_and_detects_tampering(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    for i in range(5):
        log.append("evt", i=i)
    assert AuditLog.verify(p) == (True, 5, None)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[2]); rec["i"] = 999
    lines[2] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    ok, _, bad = AuditLog.verify(p)
    assert not ok and bad == 2


def test_audit_chain_continues_across_restarts(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p).append("one")
    AuditLog(p).append("two")
    assert AuditLog.verify(p)[:2] == (True, 2)


def test_drift_monitor_alerts_on_ungrounded_spike_and_score_drop():
    d = DriftMonitor(window=50, ungrounded_rate_alert=0.25, score_drop_alert=0.15)
    d.set_baseline(0.4)
    for _ in range(30):
        d.record(0.40, True)
    assert d.status()["alerts"] == []
    for _ in range(30):
        d.record(0.05, False)
    alerts = " ".join(d.status()["alerts"])
    assert "ungrounded_rate" in alerts and "mean_best_score" in alerts


def test_prometheus_rendering():
    m = MetricsRegistry()
    m.inc("req_total", status="ok"); m.observe("lat", 0.2)
    text = m.render_prometheus()
    assert 'req_total{status="ok"} 1.0' in text and "lat_count 1" in text
