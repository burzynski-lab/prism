"""Per-particle features of the surrogate and the padded (jet, slot) arrays.

Generator particles (stable, or decayed physical particles: SM hadrons,
taus, dark hadrons) inside a truth large-R jet -> PART_FLOATS (+ PART_CATS,
embedded PDG ids). Raw values are built here; ``model_space()`` applies the
fixed transforms (logs, symlogs) that the network consumes, exactly as in
training.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .jets import delta_phi, delta_r

MAX_PART = 150
PART_PT_MIN = 0.5

# ---------------------------------------------------------------- truth particles
PART_FLOATS = [
    "pt", "ptrel", "energy", "mass", "deta", "dphi", "dr", "charge", "status",
    "prod_lxy", "prod_z", "decay_lxy", "decay_z", "decay_len",
    "decay_dphi", "decay_deta", "has_decay", "n_children",
    "c0_pt", "c0_deta", "c0_dphi", "c0_mass",
    "c1_pt", "c1_deta", "c1_dphi", "c1_mass",
]
PART_CATS = ["pdgid", "c0_pdgid", "c1_pdgid"]   # embedded via a vocabulary

# transform applied before normalization: name -> (kind, scale)
_LOG = ("log", 1.0)
_SYMLOG_UM = ("symlog", 0.01)     # mm quantities: sign(x) log1p(|x| / 10 um)
_SYMLOG_MM = ("symlog", 1.0)
TRANSFORMS = {
    "pt": _LOG, "ptrel": _LOG, "energy": _LOG, "c0_pt": ("log0", 1.0), "c1_pt": ("log0", 1.0),
    "d0": _SYMLOG_UM, "z0": _SYMLOG_MM, "sigma_d0": _LOG, "sigma_z0": _LOG,
    "prod_lxy": _SYMLOG_UM, "prod_z": _SYMLOG_MM, "decay_lxy": _SYMLOG_UM,
    "decay_z": _SYMLOG_MM, "decay_len": _SYMLOG_UM,
}


def model_space(name: str, x: np.ndarray) -> np.ndarray:
    kind, scale = TRANSFORMS.get(name, ("lin", 1.0))
    x = np.asarray(x, dtype=np.float32)
    if kind == "log":
        return np.log(np.maximum(x, 1e-6))
    if kind == "log0":                      # 0 stays 0 (absent child)
        return np.where(x > 0, np.log(np.maximum(x, 1e-6)), 0.0).astype(np.float32)
    if kind == "symlog":
        return (np.sign(x) * np.log1p(np.abs(x) / scale)).astype(np.float32)
    return x


def primary_vertex_z(part) -> np.ndarray:
    """Per-event z of the primary interaction, the median production z of the
    prompt particles (those made within 1 mm of the beamline).

    Every longitudinal quantity is measured from this point, so the features do
    not depend on where along the beamline the event was placed. The training
    samples were simulated with pile-up, which moves the hard scatter along z;
    without this reference an uploaded generator file would be scored out of
    distribution.
    """
    n = ak.to_numpy(ak.num(part.pt))
    p = ak.flatten(part)
    z = ak.to_numpy(p.z)
    prompt = np.hypot(ak.to_numpy(p.x), ak.to_numpy(p.y)) < 1.0
    ev = np.repeat(np.arange(len(n)), n)
    pv = np.zeros(len(n), dtype=np.float32)
    if prompt.any():
        order = np.lexsort((z[prompt], ev[prompt]))
        e, zz = ev[prompt][order], z[prompt][order]
        _, start, cnt = np.unique(e, return_index=True, return_counts=True)
        pv[np.unique(e)] = zz[start + cnt // 2]
    return pv


def padded_dtype(floats, cats):
    return np.dtype([(n, "f4") for n in floats] + [(n, "i4") for n in cats] + [("valid", "?")])


def pad_groups(jet_id: np.ndarray, n_jets: int, sort_key: np.ndarray, columns: dict,
               max_n: int, floats, cats) -> np.ndarray:
    """Scatter flat per-object columns into a (n_jets, max_n) structured array,
    ordering objects within a jet by descending ``sort_key`` and truncating."""
    out = np.zeros((n_jets, max_n), dtype=padded_dtype(floats, cats))
    sel = jet_id >= 0
    jid, key = jet_id[sel], sort_key[sel]
    order = np.lexsort((-key, jid))
    jid = jid[order]
    starts = np.searchsorted(jid, jid, side="left")
    pos = np.arange(len(jid)) - starts
    keep = pos < max_n
    rows, cols = jid[keep], pos[keep]
    src = np.flatnonzero(sel)[order][keep]
    for name in list(floats) + list(cats):
        out[name][rows, cols] = columns[name][src]
    out["valid"][rows, cols] = True
    return out


# ---------------------------------------------------------------- truth particles
def particle_columns(part: ak.Array, jet_id: np.ndarray, jets: np.ndarray,
                     pv_z: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Flat per-particle feature columns (status 1 or 2, pT > PART_PT_MIN, not
    a neutrino) relative to the associated truth large-R jet. ``pv_z`` is the
    per-event primary vertex; every z is referred to it, so that production and
    decay vertices, and hence flight lengths, do not depend on where the event
    sits along the beamline."""
    n = ak.to_numpy(ak.num(part.pt))
    off = np.concatenate([[0], np.cumsum(n)])
    p = ak.flatten(part)
    pid = ak.to_numpy(p.pid); apid = np.abs(pid)
    status = ak.to_numpy(p.status)
    pt = ak.to_numpy(p.pt); eta = ak.to_numpy(p.eta); phi = ak.to_numpy(p.phi)
    e = ak.to_numpy(p.e); m = ak.to_numpy(p.mass)
    q = ak.to_numpy(p.charge).astype(np.float32)   # from the PDG tables, filled by the HepMC reader
    x = ak.to_numpy(p.x); y = ak.to_numpy(p.y); z = ak.to_numpy(p.z)
    d1 = ak.to_numpy(p.d1); d2 = ak.to_numpy(p.d2)
    ev = np.repeat(np.arange(len(n)), n)
    if pv_z is not None:
        z = z - pv_z[ev]
    # Status 1 stable, 2 decayed hadron or tau (HepMC convention); Pythia keeps
    # its own codes for dark hadrons: 81-89 from fragmentation, 91-99 from
    # decays. Keep every stable particle and every decayed physical particle,
    # SM or dark.
    dark = apid >= 4900000
    decayed = (status == 2) | (dark & (status >= 81) & (status <= 99))
    ok = ((status == 1) | decayed) & (pt > PART_PT_MIN) & (jet_id >= 0)
    ok &= ~np.isin(apid, (12, 14, 16))
    jid = np.where(ok, jet_id, -1)
    j = jets[np.maximum(jid, 0)]

    # children: D1..D2 are event-local indices into the Particle list
    has = decayed & (d1 >= 0)
    g1 = np.where(has, off[ev] + d1, 0)
    nchild = np.where(has, np.where(d2 >= d1, d2 - d1 + 1, 1), 0)
    has2 = has & (nchild >= 2)
    g2 = np.where(has2, g1 + 1, 0)
    # decay vertex = production vertex of the first child
    dx = np.where(has, x[g1], 0.0); dy = np.where(has, y[g1], 0.0); dz = np.where(has, z[g1], 0.0)
    fx, fy, fz = dx - x, dy - y, dz - z
    flight_t = np.hypot(fx, fy)
    dec_phi = np.where(has & (flight_t > 0), delta_phi(np.arctan2(fy, fx), phi), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec_eta = np.where(has & (flight_t > 0), np.arcsinh(fz / np.maximum(flight_t, 1e-9)) - eta, 0.0)

    def child(g, ok_c):
        return {
            "pt": np.where(ok_c, pt[g], 0.0),
            "deta": np.where(ok_c, eta[g] - eta, 0.0),
            "dphi": np.where(ok_c, delta_phi(phi[g], phi), 0.0),
            "mass": np.where(ok_c, m[g], 0.0),
            "pdgid": np.where(ok_c, pid[g], 0),
        }
    c0, c1 = child(g1, has), child(g2, has2)
    cols = {
        "pt": pt, "ptrel": pt / np.maximum(j["pt"], 1e-6), "energy": e, "mass": m,
        "deta": eta - j["eta"], "dphi": delta_phi(phi, j["phi"]), "dr": delta_r(eta, phi, j["eta"], j["phi"]),
        "charge": q, "status": status.astype(np.float32),
        "prod_lxy": np.hypot(x, y), "prod_z": z,
        "decay_lxy": np.hypot(dx, dy), "decay_z": dz, "decay_len": np.sqrt(fx**2 + fy**2 + fz**2),
        "decay_dphi": dec_phi, "decay_deta": dec_eta, "has_decay": has.astype(np.float32),
        "n_children": nchild.astype(np.float32),
        "c0_pt": c0["pt"], "c0_deta": c0["deta"], "c0_dphi": c0["dphi"], "c0_mass": c0["mass"],
        "c1_pt": c1["pt"], "c1_deta": c1["deta"], "c1_dphi": c1["dphi"], "c1_mass": c1["mass"],
        "pdgid": pid, "c0_pdgid": c0["pdgid"], "c1_pdgid": c1["pdgid"],
    }
    return {k: (np.asarray(v, dtype=np.float32) if k not in PART_CATS else np.asarray(v, dtype=np.int32))
            for k, v in cols.items()}, jid
