"""Predictor of the emerging-jets example analysis: generator particles are
clustered into truth large-R jets, a transformer surrogate returns per jet
the probability that the detector-level tagger would select it, and the
per-event probability of the two-jet signal region follows from the Poisson
binomial. All the physics lives in ``surrogate/`` next to this file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.predictors import Predictor, finish_summary, register

from .surrogate import features as F
from .surrogate.inference import load_checkpoint, score_padded
from .surrogate.jets import associate, jet_table, recluster_large_r, small_r_jets_from_particles
from .surrogate.metrics import predicted_sr_efficiency, prob_at_least, sr_efficiency

TRUTH_PT_MIN = 150.0        # truth jets are kept below the 200 GeV analysis cut so the reco turn-on is modelled


def truth_tables(part) -> tuple[np.ndarray, np.ndarray]:
    """Generator particles of one batch -> (truth_jets, padded truth_parts)."""
    truth = recluster_large_r(small_r_jets_from_particles(part), pt_min=TRUTH_PT_MIN)
    part_jet = associate(part, truth)
    jets = jet_table(truth, np.zeros(len(truth), np.int32))
    cols, part_jet = F.particle_columns(part, part_jet, jets)
    jets["n_assoc"] = np.bincount(part_jet[part_jet >= 0], minlength=len(truth))
    parts = F.pad_groups(part_jet, len(truth), cols["pt"], cols, F.MAX_PART, F.PART_FLOATS, F.PART_CATS)
    return jets, parts


@register("emerging_jets_transformer")
class EmergingJetsPredictor(Predictor):
    def __init__(self, analysis, device: str = "cpu"):
        super().__init__(analysis, device)
        self.device = device
        self.model, self.pre = load_checkpoint(analysis.model_path, device)

    def run(self, hepmc: Path, max_events: int, progress=None, options=None, chunk: int = 1000):
        from prism.hepmc import read_hepmc
        jets, parts, n_ev = [], [], 0
        for batch in read_hepmc(hepmc, max_events=max_events, chunk=chunk):
            tj, tp = truth_tables(batch.part)
            tj["event"] += n_ev
            jets.append(tj); parts.append(tp); n_ev += len(batch)
            if progress:
                progress(f"{n_ev} events read, {sum(len(j) for j in jets)} truth jets")
        if n_ev == 0:
            raise ValueError("no events could be read from the input (is it HepMC2/3 ASCII?)")
        jets, parts = np.concatenate(jets), np.concatenate(parts)
        logit = score_padded(self.model, self.pre, parts, self.device)
        prob = 1.0 / (1.0 + np.exp(-logit))
        eff, err = predicted_sr_efficiency(jets["event"], prob, n_ev)
        eff_thr, err_thr = sr_efficiency(jets["event"], prob > 0.5, n_ev)
        order = np.argsort(jets["event"], kind="stable")
        ev, pr = jets["event"][order], prob[order]
        starts = np.searchsorted(ev, np.arange(n_ev), side="left")
        stops = np.searchsorted(ev, np.arange(n_ev), side="right")
        per_event = np.array([prob_at_least(pr[a:b], 2) if b > a else 0.0 for a, b in zip(starts, stops)])
        summary = {"n_events": int(n_ev), "n_truth_jets": int(len(jets)),
                   "sr_efficiency": eff, "sr_efficiency_err": err,
                   "sr_efficiency_threshold05": eff_thr, "sr_efficiency_threshold05_err": err_thr,
                   "mean_jet_probability": float(prob.mean()) if len(prob) else None}
        summary = finish_summary(self.analysis, summary, per_event,
                                 objects={"label": "truth jets", "count": int(len(jets)),
                                          "mean_probability": float(prob.mean()) if len(prob) else None})
        return summary, per_event, {"jet_probability": prob, "truth_jets": jets}
