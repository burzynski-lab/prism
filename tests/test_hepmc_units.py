import numpy as np
from pyHepMC3 import HepMC3 as hm

from prism.hepmc import _event_record


def test_mev_and_gev_events_produce_same_features():
    def record(unit, scale):
        event = hm.GenEvent(unit, hm.Units.MM)
        vertex = hm.GenVertex()
        particle = hm.GenParticle(hm.FourVector(30 * scale, 40 * scale, 0, 100 * scale), 35, 2)
        particle.set_generated_mass(50 * scale)
        vertex.add_particle_out(particle)
        event.add_vertex(vertex)
        return _event_record(event)

    gev = record(hm.Units.GEV, 1)
    mev = record(hm.Units.MEV, 1000)
    for field in ('mass', 'pt', 'e', 'eta'):
        np.testing.assert_allclose(mev[field], gev[field])
    np.testing.assert_allclose(mev['mass'], [50])
