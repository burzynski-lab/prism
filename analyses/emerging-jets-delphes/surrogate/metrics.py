"""Event-level signal-region probabilities from per-jet probabilities."""

from __future__ import annotations

import numpy as np


def efficiency(mask: np.ndarray) -> tuple[float, float]:
    """(efficiency, binomial standard error)."""
    n = len(mask)
    if n == 0:
        return float("nan"), float("nan")
    k = float(np.sum(mask))
    eff = k / n
    return eff, float(np.sqrt(max(eff * (1 - eff), 1.0 / n) / n))


def prob_at_least(p: np.ndarray, k: int = 2) -> float:
    """P(at least k successes) for independent Bernoulli(p_i) (Poisson binomial)."""
    p = np.asarray(p, dtype=float)
    # distribution of the number of successes, truncated at k
    dist = np.zeros(k + 1)
    dist[0] = 1.0
    for pi in p:
        new = dist * (1 - pi)
        new[1:] += dist[:-1] * pi
        new[k] += dist[k] * pi        # mass at >= k stays at >= k
        dist = new
    return float(dist[k])


def event_pass_counts(event: np.ndarray, passed: np.ndarray, n_events: int) -> np.ndarray:
    """Number of passing jets per event (length n_events)."""
    return np.bincount(event[passed], minlength=n_events)


def sr_efficiency(event: np.ndarray, passed: np.ndarray, n_events: int, n_min: int = 2):
    """Fraction of events with >= n_min passing jets, with binomial error."""
    return efficiency(event_pass_counts(event, passed, n_events) >= n_min)


def predicted_sr_efficiency(event: np.ndarray, prob: np.ndarray, n_events: int, n_min: int = 2):
    """Surrogate prediction: mean over events of P(>= n_min jets pass) from
    per-jet pass probabilities. Returns (mean, std error of the mean)."""
    order = np.argsort(event, kind="stable")
    ev, pr = event[order], prob[order]
    starts = np.searchsorted(ev, np.arange(n_events), side="left")
    stops = np.searchsorted(ev, np.arange(n_events), side="right")
    per_event = np.array([prob_at_least(pr[a:b], n_min) if b > a else 0.0
                          for a, b in zip(starts, stops)])
    return float(per_event.mean()), float(per_event.std(ddof=1) / np.sqrt(n_events)) if n_events > 1 else float("nan")
