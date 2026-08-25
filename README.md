# quantum-future-feedback-hardware

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22099471.svg)](https://doi.org/10.5281/zenodo.22099471)

Code, count data, pre-registered plans and machine-checkable gates for the
hardware campaigns of the future-referential feedback series.

> **Paper VII — One round is enough: certified entanglement-breaking indices
> from single-round tomography**
> Eran Kopel (Tel Aviv University) — submitted to *Physical Review A*

Companion to [`quantum-future-feedback`](https://github.com/erankopel/quantum-future-feedback),
which carries the theory and machine-verified certificates of Papers I and II.

The entanglement-breaking index of a thermal collision feedback loop is an
integer near 90 at the certified targets of the series, and no device holds
coherence for 90 rounds. It does not have to: the round map is a fixed channel
and the composite is its power, so one round measured once determines the whole
trajectory, provided the measurement uncertainty is carried as an interval
rather than a point estimate. This repository is the hardware side of that
claim: three pre-registered campaigns on one superconducting device, 1704
seconds of quantum processor time in total.

## Layout

| Path | Contents |
|---|---|
| `src/` | Analysis core: the round unitary and ideal affine pair, the tomographic read-off, the explorer bundle (contraction, implemented index, polar rotation, anisotropy), the shared submission core and the fitted cost model |
| `hardware/` | Submission and analysis scripts for rungs 1 to 15, one pair per experiment: baseline, decoupling, delay scaling, echo, segmented idle, one- and two-qubit gate-set tomography, the virtual-Z scan and its wrong-link control, and the stability-epoch driver |
| `gates/` | **The gates.** Executable assertions that ran before any submission, at zero device cost, plus the synthetic unit tests |
| `plans/` | Pre-registered plans, the amendment record with each amendment dated against the data it had or had not seen, and the window-3 results summary |
| `data/` | One JSON per hardware job and per analysis, the stability ledger, gauge-invariant gate-set outputs and bootstraps, and the fitted cost models |
| `calibration/` | Device calibration snapshot and the QPU-second ledgers for all three windows |
| `figures/` | Figure-generation scripts for the paper |
| `tools/` | Cost-model refits, the power and sensitivity studies, the relaxation fit and the report generator |

## Reproducing the paper's numbers

Nothing below needs a quantum processor. Every published number is re-derived
from the stored counts.

```bash
pip install numpy scipy matplotlib          # analysis only
python gates/gate_validate.py               # GATE GCORE: re-derives the paper's own
                                            # recorded numbers from stored counts and
                                            # exits non-zero if they do not reproduce
python gates/w3_gate_unittest.py            # the decision gate, 6 synthetic cases x 8 seeds
python tools/w3_relaxation_fit.py           # the post-recalibration relaxation, tau = 13.8 min
python tools/w3_v3_forensics.py             # the eta / eta-prime decomposition, 31 per cent rotation
python tools/w3_report.py                   # regenerates plans/window3_results.txt
python figures/w3_figures.py                # Figs. 2 to 4
```

`gate_validate.py` is the entry point. It is the gate the campaign itself had
to pass before any new data was trusted, and it is the fastest way for a reader
to confirm that this repository reproduces the manuscript.

Certified-given-shots quantities additionally need `python-flint>=0.9`.
Re-submitting to hardware needs `qiskit`, `qiskit-ibm-runtime` and, for the
gate-set tomography, `pygsti>=0.10.2`; none of that is required to reproduce a
published number.

## The gates

The campaigns are pre-registered, and the plans in `plans/` are the record.
Gates are machine-checkable assertions that must pass on the live backend
before anything is submitted:

| Gate | Asserts |
|---|---|
| `GCORE` | the analysis pipeline re-derives the series' own published numbers from stored counts |
| `GVZ` | the inserted `rz` count equals the compiled two-qubit gate count on the targeted link, the link is non-empty, no other operation count changes, and `rz` is zero-duration on the live target |
| `GGST2` | every gate-set-tomography circuit preserves its logical gate counts through transpilation |
| `G-VZ` | the decision gate: the registered predictions and the signed-arm asymmetry against a bootstrap floor, a magnitude-tracking artefact by odd/even symmetry decomposition, and a guarded hand-off of the fitted angle |

Budget guards refuse any submission projecting past the window cap and count
jobs already in flight, since execution seconds bill asynchronously.

## Scope and honesty notes

Hardware numbers are labelled **explorer** throughout: float64, SPAM-inclusive,
single calibration epoch, no error mitigation. Certified brackets are interval
arithmetic and live in the companion repository.

Raw provider-format job dumps (about 24 MB) are omitted; every quantity the
paper uses is distilled into the `p0` click probabilities of
`data/rung*_result.json`. Available from the author on request.

Four amendments were made to the window-3 plan during the window. Three
preceded any submission; one, the plateau rule for the angle hand-off, was made
post hoc and is labelled as such in `plans/window3_plan_v2.txt`. Two
pre-registered predictions failed, one on the sign of a correction and one on
its mechanism, and both are recorded as failures rather than rewritten.

**No data were excluded.**

### Redaction

The cloud account and instance identifiers have been redacted from the
QPU-second ledgers and the stability ledger, replaced by stable labels
(`REDACTED-ACCOUNT-1..3`, `REDACTED-INSTANCE-1..3`). The substitution is
one-to-one, so the three separate 600-second allocations the paper describes
remain distinguishable from one another; only the identifiers are removed. No
timestamp, workload id, status or second of billed usage was altered. The
release gate refuses to tag a release if any identifier reappears.

## Citation

See `CITATION.cff`. License: MIT.
