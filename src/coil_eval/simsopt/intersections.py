"""Bounded geometric-section search for a confirmed surface self-intersection.

Derived from the independent QUASR geometric-plane audit. This inexpensive
acceptance guard stops at its first original-surface witness. A negative search
is only a sampled result; nonmonotone toroidal maps and unresolved candidates
remain explicitly inconclusive. No source magnetic axis is required.
"""
from __future__ import annotations

import time
import numpy as np


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _candidates(rz):
    """Nonadjacent closed-polygon intersections, including collinear overlap."""
    following = np.roll(rz, -1, axis=0)
    eps = 1e-12 * max(float(np.ptp(rz, axis=0).max()), 1e-8) ** 2
    for i in range(len(rz) - 2):
        indices = np.arange(i + 2, len(rz))
        if i == 0:
            indices = indices[indices != len(rz) - 1]
        a, da = rz[i], following[i] - rz[i]
        b, db = rz[indices], following[indices] - rz[indices]
        delta, determinant = b - a, _cross(da, db)
        safe = np.abs(determinant) > eps
        t = np.divide(_cross(delta, db), determinant, out=np.full(len(indices), np.nan), where=safe)
        u = np.divide(_cross(delta, da), determinant, out=np.full(len(indices), np.nan), where=safe)
        hits = safe & (t >= -1e-10) & (t <= 1 + 1e-10) & (u >= -1e-10) & (u <= 1 + 1e-10)
        for k in np.flatnonzero(hits):
            yield i, int(indices[k]), float(np.clip(t[k], 0, 1)), float(np.clip(u[k], 0, 1)), "transverse"
        norm2 = float(np.dot(da, da))
        if norm2 <= eps:
            continue
        collinear = (~safe) & (np.abs(_cross(delta, da)) <= eps)
        t0 = np.sum(delta * da, axis=1) / norm2
        t1 = np.sum((following[indices] - a) * da, axis=1) / norm2
        low, high = np.maximum(np.minimum(t0, t1), 0), np.minimum(np.maximum(t0, t1), 1)
        for k in np.flatnonzero(collinear & (high - low > 1e-8)):
            ti, db2 = float((low[k] + high[k]) / 2), float(np.dot(db[k], db[k]))
            if db2 > eps:
                yield i, int(indices[k]), ti, float(np.dot(a + ti * da - b[k], db[k]) / db2), "collinear_overlap"


def detect_self_intersection(surface, *, n_sections=8, theta_samples=256,
                             phi_samples=None, xyz_tolerance_m=1e-9,
                             min_parameter_separation=1e-4,
                             maximum_refinements_per_section=32):
    """Return diagnostic status and the first confirmed distinct-parameter root.

    Defaults inspect eight true geometric-phi sections over one field period.
    A hard rejection requires ``confirmed_self_intersection=True``. An
    inconclusive status must remain a quality warning rather than being called
    an embedding pass. Ordinary calls need NumPy; SciPy is imported only when
    polygon crossings require root refinement.
    """
    started = time.perf_counter()
    result = {
        "method": "geometric-sections-original-surface-root-v1",
        "status": "inconclusive", "checked": False,
        "confirmed_self_intersection": False, "absence_is_not_a_certificate": True,
        "n_sections": int(n_sections), "theta_samples": int(theta_samples),
        "witness_xyz_tolerance_m": float(xyz_tolerance_m),
        "minimum_normalized_parameter_separation": float(min_parameter_separation),
        "maximum_refinements_per_section": int(maximum_refinements_per_section),
        "sampled_sections": 0, "candidate_count": 0, "refinement_attempts": 0,
        "unresolved_candidates": 0, "surface_point_evaluations": 0, "witness": None,
    }

    def points(phis, thetas):
        phis, thetas = np.broadcast_arrays(phis, thetas)
        xyz = np.empty((phis.size, 3), dtype=np.float64)
        surface.gamma_lin(xyz, np.ascontiguousarray(phis.ravel(), dtype=np.float64),
                          np.ascontiguousarray(thetas.ravel(), dtype=np.float64))
        result["surface_point_evaluations"] += phis.size
        return xyz.reshape((*phis.shape, 3))

    try:
        nfp = int(surface.nfp)
        phi_samples = int(phi_samples if phi_samples is not None else max(65, 4 * int(getattr(surface, "ntor", 0)) + 1))
        if (nfp <= 0 or n_sections < 1 or theta_samples < 8 or phi_samples < 9 or
                xyz_tolerance_m <= 0 or min_parameter_separation <= 0 or maximum_refinements_per_section < 1):
            raise ValueError("invalid geometric-section settings")
        result.update(nfp=nfp, monotonicity_phi_samples=phi_samples)
        period = 2 * np.pi / nfp
        theta = np.arange(theta_samples, dtype=float) / theta_samples
        parameter_phi = np.linspace(0, 1 / nfp, phi_samples)
        coarse = points(parameter_phi[:, None], theta[None, :])
        if not np.isfinite(coarse).all() or np.hypot(coarse[..., 0], coarse[..., 1]).min() <= 1e-8:
            raise ValueError("nonfinite surface or undefined geometric angle near R=0")
        geometric = np.unwrap(np.arctan2(coarse[..., 1], coarse[..., 0]), axis=0)
        increments = np.diff(geometric, axis=0)
        result["sampled_toroidal_increment_min_rad"] = float(increments.min())
        if increments.min() <= 1e-10 or increments.max() >= np.pi / 2:
            result["status"] = "inconclusive_toroidal_map"
            raise ValueError("geometric toroidal map is nonmonotone or unresolved")
        if np.max(np.abs(geometric[-1] - geometric[0] - period)) > 1e-8:
            result["status"] = "inconclusive_toroidal_map"
            raise ValueError("geometric toroidal winding differs from one field period")
        result["checked"] = True
        columns = np.arange(theta_samples)
        for section in range(n_sections):
            desired = period * section / n_sections
            er = np.array([np.cos(desired), np.sin(desired), 0])
            normal = np.array([-np.sin(desired), np.cos(desired), 0])
            shifts = np.floor((desired - geometric[0]) / period)
            base_target = desired - shifts * period
            hi_index = np.clip(np.sum(geometric <= base_target[None, :], axis=0), 1, phi_samples - 1)
            lo = parameter_phi[hi_index - 1] + shifts / nfp
            hi = parameter_phi[hi_index] + shifts / nfp
            flo = geometric[hi_index - 1, columns] - base_target
            fhi = geometric[hi_index, columns] - base_target
            roots, xyz = np.full(theta_samples, np.nan), np.full((theta_samples, 3), np.nan)
            active = np.ones(theta_samples, dtype=bool)
            # Safeguarded bracketed secant, with bisection for slow brackets.
            # Completed theta rays are removed from subsequent Fourier calls.
            for iteration in range(42):
                indices = np.flatnonzero(active)
                if not len(indices):
                    break
                fraction = np.clip(-flo[indices] / (fhi[indices] - flo[indices]), 0, 1)
                if iteration >= 18 or iteration % 6 == 5:
                    fraction[:] = .5
                mid = lo[indices] + fraction * (hi[indices] - lo[indices])
                value = points(mid, theta[indices])
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite section root geometry")
                angle = np.arctan2(value[:, 1], value[:, 0])
                angle += 2 * np.pi * np.rint((desired - angle) / (2 * np.pi))
                residual = angle - desired
                done = (np.abs(value @ normal) < xyz_tolerance_m * .1) & (value @ er > 0)
                roots[indices[done]], xyz[indices[done]] = mid[done], value[done]
                active[indices[done]] = False
                below = residual < 0
                lo[indices[below]], flo[indices[below]] = mid[below], residual[below]
                hi[indices[~below]], fhi[indices[~below]] = mid[~below], residual[~below]
            if active.any():
                raise ValueError("geometric section root inversion did not converge")
            eps_phi = 1e-6 / nfp
            minus, plus = points(roots - eps_phi, theta), points(roots + eps_phi, theta)
            dangle = np.angle((plus[:, 0] + 1j * plus[:, 1]) / (minus[:, 0] + 1j * minus[:, 1]))
            if not np.isfinite(dangle).all() or dangle.min() <= 0:
                result["status"] = "inconclusive_toroidal_map"
                raise ValueError("local toroidal map is not increasing at section roots")
            result["sampled_sections"] += 1
            rz = np.column_stack((xyz @ er, xyz[:, 2]))
            candidates = list(_candidates(rz))
            result["candidate_count"] += len(candidates)
            result["unresolved_candidates"] += max(0, len(candidates) - maximum_refinements_per_section)
            if not candidates:
                continue
            from scipy.optimize import root

            def interpolate(index, fraction):
                nxt = (index + 1) % theta_samples
                delta = (roots[nxt] - roots[index] + .5) % 1 - .5
                return [roots[index] + fraction * delta, theta[index] + fraction / theta_samples]

            def equations(parameters):
                pair = points(parameters[[0, 2]], parameters[[1, 3]])
                delta = pair[0] - pair[1]
                return np.array([pair[0] @ normal, pair[1] @ normal, delta @ er, delta[2]])

            for i, j, ti, tj, kind in candidates[:maximum_refinements_per_section]:
                initial = np.r_[interpolate(i, ti), interpolate(j, tj)]
                solved = root(equations, initial, method="hybr", options={"xtol": 1e-10, "maxfev": 180})
                result["refinement_attempts"] += 1
                parameters = np.asarray(solved.x)
                if not np.isfinite(parameters).all():
                    result["unresolved_candidates"] += 1
                    continue
                pair = points(parameters[[0, 2]], parameters[[1, 3]])
                delta_phi = abs((parameters[0] - parameters[2] + .5) % 1 - .5) * nfp
                delta_theta = abs((parameters[1] - parameters[3] + .5) % 1 - .5)
                separation = float(np.hypot(delta_phi, delta_theta))
                error = float(np.linalg.norm(pair[0] - pair[1]))
                residual = float(np.max(np.abs(equations(parameters))))
                if (np.isfinite(pair).all() and error < xyz_tolerance_m and residual < xyz_tolerance_m
                        and separation > min_parameter_separation and np.min(pair @ er) > 0):
                    result.update(status="confirmed_self_intersection", confirmed_self_intersection=True,
                        witness={"geometric_phi_rad": desired, "section_index": section,
                                 "parameter_phi_cycles": parameters[[0, 2]].tolist(),
                                 "parameter_theta_cycles": parameters[[1, 3]].tolist(),
                                 "normalized_parameter_separation": separation,
                                 "xyz_m": np.mean(pair, axis=0).tolist(),
                                 "xyz_disagreement_m": error, "equation_residual_max_m": residual,
                                 "candidate_kind": kind, "root_success": bool(solved.success)})
                    return result
                result["unresolved_candidates"] += 1
        result["status"] = ("inconclusive_candidates" if result["unresolved_candidates"] else
                            "not_detected_in_sampled_sections")
    except Exception as exc:
        if not result["status"].startswith("inconclusive"):
            result["status"] = "inconclusive_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_s"] = time.perf_counter() - started
    return result
