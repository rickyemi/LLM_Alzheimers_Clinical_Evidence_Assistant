"""Observability and governance plumbing: metrics, drift monitor, tamper-evident audit log.

Why:
  * Metrics (Prometheus text format) feed dashboards and alerts: latency, error
    rate, refusal/ungrounded rate, token spend.
  * DriftMonitor watches retrieval health. A falling mean similarity or a rising
    ungrounded rate means users are asking about things the corpus does not
    cover (or the corpus went stale): the cue to refresh the knowledge base.
  * AuditLog is an append-only JSONL file where each record embeds the hash of the
    previous one (a hash chain). Editing or deleting any record breaks
    verification, giving tamper evidence for regulated settings.
Only REDACTED text and hashes are logged, never raw user input.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class MetricsRegistry:
    """Tiny Prometheus-compatible registry (counters + histograms)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, Tuple], float] = defaultdict(float)
        self._hist: Dict[str, List[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._hist[name].append(value)

    def summary(self) -> Dict:
        import numpy as np
        with self._lock:
            out: Dict = {"counters": {}, "latency": {}}
            for (name, labels), v in self._counters.items():
                key = name + ("{" + ",".join(f'{k}="{x}"' for k, x in labels) + "}" if labels else "")
                out["counters"][key] = v
            for name, vals in self._hist.items():
                if vals:
                    out["latency"][name] = {
                        "count": len(vals),
                        "p50": float(np.percentile(vals, 50)),
                        "p95": float(np.percentile(vals, 95)),
                    }
            return out

    def render_prometheus(self) -> str:
        lines: List[str] = []
        with self._lock:
            for (name, labels), v in sorted(self._counters.items()):
                lab = "{" + ",".join(f'{k}="{x}"' for k, x in labels) + "}" if labels else ""
                lines.append(f"{name}{lab} {v}")
            for name, vals in sorted(self._hist.items()):
                for b in _BUCKETS:
                    lines.append(f'{name}_bucket{{le="{b}"}} {sum(1 for x in vals if x <= b)}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {len(vals)}')
                lines.append(f"{name}_sum {sum(vals)}")
                lines.append(f"{name}_count {len(vals)}")
        return "\n".join(lines) + "\n"


class DriftMonitor:
    """Rolling-window retrieval health with threshold alerts."""

    def __init__(self, window: int = 200, ungrounded_rate_alert: float = 0.25,
                 score_drop_alert: float = 0.15):
        self.scores: Deque[float] = deque(maxlen=window)
        self.grounded: Deque[bool] = deque(maxlen=window)
        self.ungrounded_rate_alert = ungrounded_rate_alert
        self.score_drop_alert = score_drop_alert
        self.baseline_mean: Optional[float] = None

    def set_baseline(self, mean_score: float) -> None:
        self.baseline_mean = mean_score

    def record(self, best_score: float, grounded: bool) -> None:
        self.scores.append(best_score)
        self.grounded.append(grounded)

    def status(self) -> Dict:
        n = len(self.scores)
        if n == 0:
            return {"n": 0, "alerts": []}
        mean = sum(self.scores) / n
        ungrounded = 1 - sum(self.grounded) / n
        alerts: List[str] = []
        if n >= 20 and ungrounded > self.ungrounded_rate_alert:
            alerts.append(f"ungrounded_rate {ungrounded:.2f} > {self.ungrounded_rate_alert}")
        if n >= 20 and self.baseline_mean and mean < self.baseline_mean * (1 - self.score_drop_alert):
            alerts.append(f"mean_best_score {mean:.3f} dropped >{self.score_drop_alert:.0%} "
                          f"vs baseline {self.baseline_mean:.3f}")
        return {"n": n, "mean_best_score": mean, "ungrounded_rate": ungrounded,
                "baseline_mean": self.baseline_mean, "alerts": alerts}


class AuditLog:
    """Append-only, hash-chained JSONL audit trail."""

    GENESIS = "0" * 64

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS
        last = self.path.read_text().strip().splitlines()[-1]
        return json.loads(last)["hash"]

    @staticmethod
    def _digest(prev: str, body: Dict) -> str:
        return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()

    def append(self, event: str, **fields) -> Dict:
        with self._lock:
            body = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": event, **fields}
            rec = {**body, "prev": self._last, "hash": self._digest(self._last, body)}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            self._last = rec["hash"]
            return rec

    @classmethod
    def verify(cls, path: Path) -> Tuple[bool, int, Optional[int]]:
        """Return (ok, n_records, index_of_first_bad_record)."""
        prev, n = cls.GENESIS, 0
        p = Path(path)
        if not p.exists():
            return True, 0, None
        for i, line in enumerate(p.read_text().splitlines()):
            rec = json.loads(line)
            body = {k: v for k, v in rec.items() if k not in {"prev", "hash"}}
            if rec["prev"] != prev or rec["hash"] != cls._digest(prev, body):
                return False, n, i
            prev, n = rec["hash"], n + 1
        return True, n, None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]
