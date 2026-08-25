#!/usr/bin/env python3
"""
Shared hardware-submission core for the Paper VII continuation runs.

Self-contained: reuses paper7_core (verified against the series by GATE GCORE)
for U8/PREP/BASIS, so it does not need the sibling pulse_engine. The triple is
LOCKED to (21, 22, 36) exactly as rungs 3-6, which the stability study
requires (same physical qubits every epoch) and the segmented study inherits.

QPU-second cost model (series heuristic, matched by rung2's actual 80 s):
    cost_s(n_circuits, shots) = 80 * n_circuits * shots / (72 * 4096)
"""
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService
import paper7_core as C

TRIPLE = (21, 22, 36)
SEED = 7
BASE_SHOTS_REF = 72 * 4096
BASE_SECONDS_REF = 80.0


def cost_seconds(n_circuits, shots):
    return BASE_SECONDS_REF * n_circuits * shots / BASE_SHOTS_REF


def service():
    return QiskitRuntimeService()      # saved account (container only)


def build_tomo(prep, meas, f, l):
    """Baseline message-qubit tomography circuit (identical to rung2 build_hw)."""
    qc = QuantumCircuit(3, 1)
    for g in C.PREP[prep]:
        getattr(qc, g)(0)
    if f:
        qc.x(1)
    if l:
        qc.x(2)
    qc.unitary(Operator(C.U8), [2, 1, 0], label="round")   # G0-checked order
    for g in C.BASIS[meas]:
        getattr(qc, g)(0)
    qc.measure(0, 0)
    return qc


def build_cpmg(prep, meas, f, l, total_dt, n_pulses, gran):
    """Tomography circuit with a fixed total idle of total_dt (dt units) on the
    message qubit, refocused by n_pulses evenly-spaced X gates (CPMG). n=0 is a
    single contiguous bare delay. Barriers isolate the idle from the transpiler.
    Returns (circuit, x_added, total_idle_dt)."""
    qc = QuantumCircuit(3, 1)
    for g in C.PREP[prep]:
        getattr(qc, g)(0)
    if f:
        qc.x(1)
    if l:
        qc.x(2)
    qc.unitary(Operator(C.U8), [2, 1, 0], label="round")
    qc.barrier(0)
    total = 0
    if n_pulses == 0:
        if total_dt > 0:
            qc.delay(total_dt, 0, unit="dt"); total = total_dt
    else:
        tau = int(round((total_dt / (2 * n_pulses)) / gran)) * gran
        qc.delay(tau, 0, unit="dt"); total += tau
        for _ in range(n_pulses - 1):
            qc.x(0)
            qc.delay(2 * tau, 0, unit="dt"); total += 2 * tau
        qc.x(0)
        qc.delay(tau, 0, unit="dt"); total += tau
    qc.barrier(0)
    for g in C.BASIS[meas]:
        getattr(qc, g)(0)
    qc.measure(0, 0)
    return qc, n_pulses, total


def all_settings():
    for prep in C.PREP:
        for meas in C.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    yield prep, meas, f, l
