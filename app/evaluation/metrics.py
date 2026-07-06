"""
metrics.py — Runtime query metrics aggregation.

Tracks per-query statistics and exposes aggregated summaries via /api/metrics.

Metrics collected:
  - Total query count
  - Cache hit rate
  - Average / P95 end-to-end latency
  - Average faithfulness score
  - Strategy distribution (which retrieval strategies were selected)
  - Guardrail trigger counts
  - Per-node latency averages
"""

import logging
import threading
from collections import defaultdict
from statistics import mean, quantiles
from typing import List

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Thread-safe metrics accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with threading.Lock():
            self._total_queries     = 0
            self._cache_hits        = 0
            self._latencies_ms: List[int]   = []
            self._faithfulness: List[float] = []
            self._strategy_counts: dict     = defaultdict(int)
            self._guardrail_counts: dict    = defaultdict(int)
            self._node_latencies: dict      = defaultdict(list)  # node -> [ms, ...]

    def record_query(self, result: dict, cached: bool = False) -> None:
        """Record metrics for a completed query."""
        with self._lock:
            self._total_queries += 1
            if cached:
                self._cache_hits += 1

            lat = result.get("latency_ms", 0)
            if lat:
                self._latencies_ms.append(lat)

            faith = result.get("faithfulness_score")
            if faith is not None:
                self._faithfulness.append(float(faith))

            strategy = result.get("retrieval_strategy", "unknown")
            self._strategy_counts[strategy] += 1

            for flag in result.get("guardrail_flags", []):
                self._guardrail_counts[flag] += 1

            for node, ms in (result.get("node_timings") or {}).items():
                self._node_latencies[node].append(ms)

    def summary(self) -> dict:
        """Return a serialisable summary of all collected metrics."""
        with self._lock:
            total = self._total_queries
            if total == 0:
                return {"message": "No queries recorded yet."}

            latencies = self._latencies_ms or [0]
            faith     = self._faithfulness or [1.0]

            # P95 latency
            try:
                p95 = int(quantiles(latencies, n=20)[18]) if len(latencies) >= 2 else latencies[-1]
            except Exception:
                p95 = max(latencies)

            # Per-node average latencies
            node_avgs = {
                node: round(mean(ms_list), 1)
                for node, ms_list in self._node_latencies.items()
                if ms_list
            }

            return {
                "total_queries":        total,
                "cache_hit_rate":       round(self._cache_hits / total, 3),
                "cache_hits":           self._cache_hits,
                "avg_latency_ms":       round(mean(latencies), 1),
                "p95_latency_ms":       p95,
                "avg_faithfulness":     round(mean(faith), 3),
                "min_faithfulness":     round(min(faith), 3),
                "strategy_distribution": dict(self._strategy_counts),
                "guardrail_triggers":   dict(self._guardrail_counts),
                "node_avg_latency_ms":  node_avgs,
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
metrics_tracker = MetricsTracker()
