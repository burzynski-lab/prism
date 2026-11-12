# Emerging jets with a track transformer tagger (Delphes ATLAS example)

This is the example analysis of the surrogate method: a fully controlled
chain in which every element is available, so the surrogate can be
confronted with the detector-level answer on models it never saw.

## The analysis being approximated

Pythia 8.312 events are passed through Delphes 3.5.1 with the ATLAS card
and ATLAS track impact-parameter resolutions, with an average of 60
minimum-bias interactions overlaid on every event and jet-area pile-up
subtraction. Anti-kt R = 0.4 particle-flow
jets are reclustered into R = 1.0 jets. A transformer tagger acting on the
reconstructed tracks of each large-radius jet separates dark showers from
QCD, with the working point fixed at a QCD jet rejection of 1000. The
signal region requires two tagged jets with pT > 200 GeV.

## The surrogate

A transformer with the same architecture acting on the generator particles
inside the truth large-radius jet (stable particles and decayed hadrons,
with kinematics, production and decay vertices, decay products and PDG
identities), trained to predict the tagger decision. Per-jet probabilities
are combined into the event-level signal-region probability with the
Poisson binomial. The surrogate reads the hard scatter only, with every
longitudinal position measured from the primary vertex, so uploads need no
pile-up overlay. File: `surrogate.pt` (architecture, weights,
normalization and PDG vocabulary).

## Validation

See the figures on the analysis page. On held-out events of the training
model the predicted signal-region efficiency is within 4% of the
detector-level value for lifetimes of 0.05 mm and above, and 13% high at
0.01 mm, where the dark-pion tracks are nearly prompt. Models outside the
training set close within 8% when the dark-shower fragmentation
(m_pi/Lambda from 0.2 to 1.4), the number of dark flavours or the Z' mass
(2 to 3 TeV) is changed, except at 0.01 mm, where the overestimate grows
with the Z' mass to 33% at 3 TeV. Changing the dark-pion mass is the known
weakness: at 0.1 mm the prediction is 18% low at 10 GeV and 28% high at
2 GeV, and the errors shrink at longer lifetimes.

## Provenance

| field | value |
|---|---|
| training code | https://github.com/jburzy/jet-surrogate (generation, tagger and surrogate training) |
| training data | nominal signal (6 lifetimes) plus m_pi/Lambda = 0.2 and 1.0, 3.5M truth jets, all with 60 pile-up interactions in the detector simulation |
| tagger | trained on disjoint seeds of the same pile-up samples, AUC 0.979, working point at a QCD jet rejection of 1000 |
| checkpoint | `logs/surrogate_20260902-T134116/ckpts/epoch=025-val_loss=0.19479.ckpt` in the training repository (version 0.5) |
| inference check | per-jet probabilities identical to the training repository's `predict` (max difference 0) on the example file and on 2000 exported pile-up events |
