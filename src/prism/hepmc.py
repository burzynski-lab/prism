"""HepMC2/3 input for the predictors.

``read_hepmc`` turns a HepMC file (ASCII, optionally gzipped) into awkward
batches of per-event particle records: pid, status, mother and daughter
index ranges (m1, m2, d1, d2), charge, mass, e, pt, eta, phi and the
production vertex x, y, z in mm. Particles are renumbered so that the
daughters of one vertex are contiguous (d1..d2 is a range). Every predictor
is free to select whatever it needs from this record.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator

import awkward as ak
import numpy as np

FIELDS = ("pid", "status", "m1", "m2", "d1", "d2", "charge", "mass", "e", "pt", "eta", "phi", "x", "y", "z")


_QUARK_THIRDS = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2}      # charge in units of e/3


def _charge_from_digits(pid: int) -> float:
    """Charge from the PDG numbering scheme (quark content), for hadrons and
    diquarks the tables do not list. Fundamental particles are looked up,
    anything else (hidden-sector ids, generator-specific codes) is neutral."""
    a = abs(pid); sign = 1.0 if pid > 0 else -1.0
    if a in _QUARK_THIRDS:
        return sign * _QUARK_THIRDS[a] / 3.0
    if a in (11, 13, 15):
        return -sign
    if a in (24, 34, 37):
        return sign
    if a < 100 or a >= 10_000_000 or 4_900_000 <= a < 5_000_000:
        return 0.0
    nq1, nq2, nq3 = (a // 1000) % 10, (a // 100) % 10, (a // 10) % 10
    if nq2 and nq3 and not nq1:                 # meson: by convention the positive id has charge >= 0
        thirds = abs(_QUARK_THIRDS.get(nq2, 0) - _QUARK_THIRDS.get(nq3, 0))
    elif nq1 and nq2 and nq3:                   # baryon
        thirds = sum(_QUARK_THIRDS.get(q, 0) for q in (nq1, nq2, nq3))
    elif nq1 and nq2 and not nq3:               # diquark
        thirds = _QUARK_THIRDS.get(nq1, 0) + _QUARK_THIRDS.get(nq2, 0)
    else:
        return 0.0
    return sign * thirds / 3.0


@lru_cache(maxsize=None)
def pdg_charge(pid: int) -> float:
    """Electric charge in units of e from the PDG numbering scheme (quark
    content), which reproduces Pythia's particle table for every hadron,
    lepton and boson. Hidden-sector states (49xxxxx) are neutral, as in
    Pythia's Hidden Valley tables; other unknown ids are neutral too."""
    return _charge_from_digits(int(pid))


def charges_from_pid(pid: np.ndarray) -> np.ndarray:
    ids, inv = np.unique(pid, return_inverse=True)
    return np.array([pdg_charge(int(i)) for i in ids], dtype=np.float32)[inv]


def _hm():
    from pyHepMC3 import HepMC3
    return HepMC3


def _open_reader(path: str | Path):
    """Reader for HepMC2 or HepMC3 ASCII; gzipped files are decompressed to a
    temporary file first (pyHepMC3 reads plain files only)."""
    hm = _hm()
    path = str(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        import gzip
        import shutil
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".hepmc", delete=False)
        with gzip.open(path, "rb") as fi:
            shutil.copyfileobj(fi, tmp)
        tmp.close()
        path = tmp.name
    head = Path(path).open("rb").read(200)
    if b"HepMC::Version 2" in head or b"HepMC::IO_GenEvent" in head:
        return hm.ReaderAsciiHepMC2(path)
    return hm.ReaderAscii(path)


def _event_record(evt) -> dict:
    """One HepMC GenEvent -> dict of numpy columns (FIELDS), daughters contiguous."""
    hm = _hm()
    lscale = 1.0 if evt.length_unit() == hm.Units.MM else 10.0          # cm -> mm
    mscale = 1.0 if evt.momentum_unit() == hm.Units.GEV else 1e-3       # MeV -> GeV
    parts = list(evt.particles())
    # order: by production vertex (so siblings are contiguous), then original id
    def key(p):
        v = p.production_vertex()
        return (v.id() if v is not None else 0, p.id())
    parts.sort(key=key)
    new_index = {p.id(): i for i, p in enumerate(parts)}
    n = len(parts)
    cols = {f: np.zeros(n, dtype=np.int32 if f in ("pid", "status", "m1", "m2", "d1", "d2") else np.float32)
            for f in FIELDS}
    cols["m1"][:] = -1; cols["m2"][:] = -1; cols["d1"][:] = -1; cols["d2"][:] = -1
    for i, p in enumerate(parts):
        mom = p.momentum()
        px, py, pz, e = mom.px() * mscale, mom.py() * mscale, mom.pz() * mscale, mom.e() * mscale
        pt = np.hypot(px, py)
        cols["pid"][i] = p.pid(); cols["status"][i] = p.status()
        cols["charge"][i] = pdg_charge(p.pid())
        gm = p.generated_mass()
        if gm > 0 or p.is_generated_mass_set():
            cols["mass"][i] = gm
        else:
            m2 = e * e - px * px - py * py - pz * pz
            cols["mass"][i] = np.sqrt(m2) if m2 > 0 else 0.0
        cols["e"][i] = e; cols["pt"][i] = pt
        cols["eta"][i] = np.arcsinh(pz / pt) if pt > 0 else 0.0
        cols["phi"][i] = np.arctan2(py, px)
        pv = p.production_vertex()
        if pv is not None:
            pos = pv.position()
            cols["x"][i] = pos.x() * lscale; cols["y"][i] = pos.y() * lscale; cols["z"][i] = pos.z() * lscale
            ins = [new_index[q.id()] for q in pv.particles_in()]
            if ins:
                cols["m1"][i] = min(ins); cols["m2"][i] = max(ins)
        ev = p.end_vertex()
        if ev is not None:
            outs = [new_index[q.id()] for q in ev.particles_out()]
            if outs:
                cols["d1"][i] = min(outs); cols["d2"][i] = max(outs)
    return cols


def read_hepmc(path: str | Path, max_events: int | None = None, chunk: int = 2000) -> Iterator[ak.Array]:
    """Yield awkward batches of up to ``chunk`` events, each with a ``part`` field."""
    hm = _hm()
    reader = _open_reader(path)
    batch, n = [], 0
    try:
        while max_events is None or n < max_events:
            evt = hm.GenEvent()
            reader.read_event(evt)
            if reader.failed():
                break
            batch.append(_event_record(evt)); n += 1
            if len(batch) == chunk:
                yield _to_awkward(batch); batch = []
    finally:
        reader.close()
    if batch:
        yield _to_awkward(batch)


def _to_awkward(batch: list[dict]) -> ak.Array:
    counts = np.array([len(b["pid"]) for b in batch])
    part = ak.zip({f: ak.unflatten(np.concatenate([b[f] for b in batch]), counts) for f in FIELDS})
    return ak.zip({"part": part}, depth_limit=1)
