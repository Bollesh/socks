# drain3/log_anomaly.py
"""
Log anomaly detection using real Drain3 template clustering
combined with per-template Isolation Forest scoring.
"""
from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Any

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest

log = logging.getLogger("log_anomaly")

# Minimum number of events seen for a template before the IF model is fitted.
MIN_SAMPLES_FOR_MODEL = 30
# sklearn IsolationForest contamination assumption.
CONTAMINATION = 0.05
# Score threshold: IsolationForest returns negative scores; more negative = more anomalous.
# -0.1 is a reasonable starting point — tune after observing real score distributions.
SCORE_THRESHOLD = float("-0.1")


class LogAnomalyDetector:
    def __init__(self):
        cfg = TemplateMinerConfig()
        # Drain3 config tweaks for HTTP log lines
        cfg.drain_sim_th = 0.4        # lower = more aggressive clustering
        cfg.drain_depth = 4
        cfg.drain_max_children = 100
        self._miner = TemplateMiner(config=cfg)

        # template_id → list of feature vectors (numpy rows)
        self._history: dict[int, list[np.ndarray]] = defaultdict(list)
        # template_id → fitted IsolationForest (or None if not enough samples yet)
        self._models: dict[int, IsolationForest | None] = defaultdict(lambda: None)
        # template_ids seen so far
        self._known: set[int] = set()

    def process(self, entry: dict[str, Any], feature_vec: np.ndarray) -> tuple[bool, str]:
        """
        Process one log entry. Returns (is_anomaly, reason).
        feature_vec must be the same 8-D vector already computed by _featurize().
        """
        message = str(entry.get("message") or entry.get("line") or "")
        result = self._miner.add_log_message(message)
        if result is None:
            return False, ""

        # Handle both dict and object return types from drain3
        if isinstance(result, dict):
            tid = result.get("cluster_id")
            template = result.get("template_mined", "")
        else:
            cluster = getattr(result, "cluster", None)
            if cluster is None:
                return False, ""
            tid = cluster.cluster_id
            template = cluster.get_template()

        if tid is None:
            return False, ""
        is_new = tid not in self._known
        self._known.add(tid)

        if is_new:
            log.info("New log template #%s: %s", tid, template)
            return True, f"drain3:new_template:{template[:80]}"

        # Accumulate history for this template
        self._history[tid].append(feature_vec)
        n = len(self._history[tid])

        # Refit IsolationForest periodically
        if n >= MIN_SAMPLES_FOR_MODEL and n % 20 == 0:
            X = np.stack(self._history[tid], axis=0)
            self._models[tid] = IsolationForest(
                contamination=CONTAMINATION, random_state=42, n_estimators=50
            )
            self._models[tid].fit(X)
            log.debug("Refitted IF for template #%s (n=%s)", tid, n)

        model = self._models[tid]
        if model is None:
            return False, ""

        score = float(model.score_samples(feature_vec.reshape(1, -1))[0])
        if score < SCORE_THRESHOLD:
            return True, (
                f"isolation_forest:template#{tid}:"
                f"score={score:.4f}<{SCORE_THRESHOLD}"
            )

        return False, ""
