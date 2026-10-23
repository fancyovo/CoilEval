"""Pure invariant-mode enumeration, extracted from the frozen psi module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolyMode:
    a: int
    b: int
    m: int
    kind: str


def build_modes(poly_degree: int, m_tor: int) -> list[PolyMode]:
    """Keep the frozen native column order and its fixed X-squared gauge."""
    modes: list[PolyMode] = []
    for deg in range(2, poly_degree + 1):
        for ax in range(deg, -1, -1):
            bz = deg - ax
            for m in range(m_tor + 1):
                if ax == 2 and bz == 0 and m == 0:
                    continue
                if m == 0:
                    modes.append(PolyMode(ax, bz, m, "cos"))
                else:
                    modes.append(PolyMode(ax, bz, m, "cos"))
                    modes.append(PolyMode(ax, bz, m, "sin"))
    return modes


def native_mode_arrays() -> tuple[np.ndarray, ...]:
    """Return the existing degree-10, toroidal-order-12 native basis."""
    modes = build_modes(10, 12)
    return (
        np.asarray([mode.a for mode in modes], dtype=np.int32),
        np.asarray([mode.b for mode in modes], dtype=np.int32),
        np.asarray([mode.m for mode in modes], dtype=np.int32),
        np.asarray([0 if mode.kind == "cos" else 1 for mode in modes], dtype=np.int32),
    )
