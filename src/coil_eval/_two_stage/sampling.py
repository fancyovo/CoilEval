"""Deterministic Cartesian cross-section sampling inside a frozen boundary.

This module deliberately knows nothing about the original s fit or magnetic
field.  A caller freezes one polygon at each *geometric* phi and passes those
polygons here.  All subsequent point decisions depend only on these arrays.

The lattice is uniform in physical (R, Z) at uniformly spaced phi.  Equal rows
therefore represent dR dZ dphi, not physical volume R dR dZ dphi.  Both weights
are returned explicitly; choosing either fitting objective is the caller's job.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack("<" + "Q" * value.ndim, *value.shape))
        digest.update(value.tobytes())
    return digest.hexdigest()


def polygon_area_and_first_R_moment(polygon: np.ndarray) -> tuple[float, float]:
    """Return positive area and integral R dR dZ, independent of orientation."""
    polygon = np.asarray(polygon, dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * np.sum(cross)
    signed_moment = np.sum((polygon[:, 0] + following[:, 0]) * cross) / 6.0
    return float(abs(signed_area)), float(np.sign(signed_area) * signed_moment)


def polygon_contains_strict(
    polygon: np.ndarray, points: np.ndarray, *, boundary_tolerance: float | None = None,
    max_pairs: int = 1_000_000,
) -> np.ndarray:
    """Even/odd containment for a simple polygon, explicitly excluding edges.

    Works for concave polygons of either orientation.  The boundary tolerance
    is a physical distance.  Chunking bounds temporary point-by-edge storage.
    Input polygons must be non-self-intersecting; this is not a topology test.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must have shape (vertices >= 3, 2)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.all(np.isfinite(polygon)) or not np.all(np.isfinite(points)):
        raise ValueError("polygon and points must be finite")
    if boundary_tolerance is None:
        boundary_tolerance = 64 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(polygon))), float(np.max(np.ptp(polygon, axis=0))))
    if boundary_tolerance < 0 or not np.isfinite(boundary_tolerance):
        raise ValueError("boundary_tolerance must be finite and nonnegative")
    start = polygon
    stop = np.roll(polygon, -1, axis=0)
    edge = stop - start
    length = np.linalg.norm(edge, axis=1)
    nonzero = length > 0
    start, stop, edge, length = start[nonzero], stop[nonzero], edge[nonzero], length[nonzero]
    if len(start) < 3:
        raise ValueError("polygon needs at least three nonzero edges")
    chunk_size = max(1, int(max_pairs) // len(start))
    result = np.zeros(len(points), dtype=bool)
    for begin in range(0, len(points), chunk_size):
        current = points[begin:begin + chunk_size]
        dx = current[:, None, 0] - start[None, :, 0]
        dz = current[:, None, 1] - start[None, :, 1]
        vertical_hit = ((start[None, :, 1] > current[:, None, 1]) !=
                        (stop[None, :, 1] > current[:, None, 1]))
        xhit = np.zeros_like(dx)
        np.divide(dz * edge[None, :, 0], edge[None, :, 1], out=xhit,
                  where=edge[None, :, 1] != 0)
        xhit += start[None, :, 0]
        inside = np.sum(vertical_hit & (current[:, None, 0] < xhit), axis=1) % 2 == 1
        cross_distance_numerator = dx * edge[None, :, 1] - dz * edge[None, :, 0]
        along = dx * edge[None, :, 0] + dz * edge[None, :, 1]
        on_edge = ((np.abs(cross_distance_numerator) <= boundary_tolerance * length) &
                   (along >= -boundary_tolerance * length) &
                   (along <= length**2 + boundary_tolerance * length)).any(axis=1)
        result[begin:begin + len(current)] = inside & ~on_edge
    return result


def _freeze_sections(polygons_rz: Sequence[np.ndarray], phis: np.ndarray, nfp: int):
    if isinstance(nfp, bool) or int(nfp) != nfp or nfp < 1:
        raise ValueError("nfp must be a positive integer")
    phis = np.array(phis, dtype=np.float64, copy=True)
    if phis.ndim != 1 or len(phis) < 2 or not np.all(np.isfinite(phis)):
        raise ValueError("phis must be a finite vector with >= 2 sections")
    if len(polygons_rz) != len(phis):
        raise ValueError("one frozen polygon is required per phi")
    period = 2 * np.pi / int(nfp)
    dphi = period / len(phis)
    # A physical-phi lattice over one complete field period.  Do not silently
    # interpret the caller's Boozer angle or repeated endpoint as geometric phi.
    if not np.allclose(np.diff(phis), dphi, rtol=1e-9, atol=1e-12):
        raise ValueError("phis must be uniformly spaced over exactly one field period, endpoint excluded")
    polygons = []
    areas, moments = [], []
    for polygon in polygons_rz:
        polygon = np.array(polygon, dtype=np.float64, copy=True)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("each polygon must have shape (vertices >= 3, 2)")
        if not np.all(np.isfinite(polygon)) or np.min(polygon[:, 0]) <= 0:
            raise ValueError("frozen polygons must be finite and strictly at R > 0")
        if np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        area, moment = polygon_area_and_first_R_moment(polygon)
        if area <= np.finfo(float).eps or moment <= 0:
            raise ValueError("frozen polygon has zero or numerically degenerate area")
        polygons.append(polygon)
        areas.append(area)
        moments.append(moment)
    return polygons, phis, dphi, np.asarray(areas), np.asarray(moments)


def _lattice_in_polygon(polygon: np.ndarray, spacing: float, shift: np.ndarray) -> np.ndarray:
    low = np.min(polygon, axis=0)
    high = np.max(polygon, axis=0)
    first = np.ceil(low / spacing - shift).astype(np.int64)
    last = np.floor(high / spacing - shift).astype(np.int64)
    r = spacing * (np.arange(first[0], last[0] + 1) + shift[0])
    z = spacing * (np.arange(first[1], last[1] + 1) + shift[1])
    rr, zz = np.meshgrid(r, z, indexing="ij")
    points = np.column_stack((rr.ravel(), zz.ravel()))
    start=polygon
    stop=np.roll(polygon,-1,axis=0)
    edge=stop-start
    length=np.linalg.norm(edge,axis=1)
    nonzero=length>0
    start,stop,edge,length=start[nonzero],stop[nonzero],edge[nonzero],length[nonzero]
    if len(start)<3:
        raise ValueError('polygon needs at least three nonzero edges')
    tolerance=64*np.finfo(float).eps*max(1.,float(np.max(abs(polygon))),float(np.max(np.ptp(polygon,axis=0))))
    # A complete scanline shares its edge intersections. Sorting these once
    # replaces the old points-by-edges matrices while preserving even/odd rules.
    mask=np.zeros((len(r),len(z)),dtype=bool)
    for iz,value in enumerate(z):
        hit=(start[:,1]>value)!=(stop[:,1]>value)
        intersection=start[hit,0]+(value-start[hit,1])*edge[hit,0]/edge[hit,1]
        intersection.sort()
        mask[:,iz]=((len(intersection)-np.searchsorted(intersection,r,side='right'))%2)==1
    # Recheck only lattice points close to a supporting edge. The final cross
    # and projection tests are exactly those used by polygon_contains_strict.
    # A two-tolerance bounding box contains its rotated tolerance rectangle.
    eps=np.finfo(float).eps
    for origin,finish,delta,size in zip(start,stop,edge,length):
        first_z=np.searchsorted(z,min(origin[1],finish[1])-2*tolerance,side='left')
        last_z=np.searchsorted(z,max(origin[1],finish[1])+2*tolerance,side='right')
        for iz in range(first_z,last_z):
            dz=z[iz]-origin[1]
            if delta[1]!=0:
                center=origin[0]+dz*delta[0]/delta[1]
                halfwidth=tolerance*size/abs(delta[1])
                halfwidth+=16*eps*(abs(center)+abs(origin[0])+1.)
                left=np.searchsorted(r,center-halfwidth,side='left')
                right=np.searchsorted(r,center+halfwidth,side='right')
            else:
                if abs(dz)>tolerance:
                    continue
                left=np.searchsorted(r,min(origin[0],finish[0])-2*tolerance,side='left')
                right=np.searchsorted(r,max(origin[0],finish[0])+2*tolerance,side='right')
            if left==right:
                continue
            dx=r[left:right]-origin[0]
            cross=dx*delta[1]-dz*delta[0]
            along=dx*delta[0]+dz*delta[1]
            on_edge=(abs(cross)<=tolerance*size)&(along>=-tolerance*size)&(along<=size*size+tolerance*size)
            mask[left:right,iz]&=~on_edge
    return np.ascontiguousarray(points[mask.ravel()])


def _evaluate_lattice(polygons, spacing, shift):
    sections = [_lattice_in_polygon(polygon, spacing, shift) for polygon in polygons]
    return sections, sum(len(section) for section in sections)


def mask_frozen_sections(polygons_rz, rz: np.ndarray, section_index: np.ndarray) -> np.ndarray:
    """Classify a fixed point superset against another frozen section boundary.

    This enables common-point comparisons between a target-V boundary and a
    1.2V fit domain without resampling either fitted s.  Section indices must
    refer to the same geometric-phi planes as the supplied polygon sequence.
    """
    rz = np.asarray(rz, dtype=np.float64)
    section_index = np.asarray(section_index)
    if rz.ndim != 2 or rz.shape[1] != 2 or section_index.shape != (len(rz),):
        raise ValueError("rz and section_index must have aligned shapes (N,2) and (N,)")
    if not np.issubdtype(section_index.dtype, np.integer):
        raise ValueError("section_index must contain integers")
    if np.any(section_index < 0) or np.any(section_index >= len(polygons_rz)):
        raise ValueError("section_index is outside the frozen polygon sequence")
    mask = np.zeros(len(rz), dtype=bool)
    for index, polygon in enumerate(polygons_rz):
        select = section_index == index
        mask[select] = polygon_contains_strict(polygon, rz[select])
    return mask


def sample_frozen_sections(
    polygons_rz: Sequence[np.ndarray], phis: np.ndarray, target_count: int = 82368,
    *, nfp: int = 4, count_tolerance: float = 0.05,
    lattice_shift: tuple[float, float] = (0.5, 0.5), max_refinements: int = 10,
    spacing: float | None = None,
) -> dict:
    """Sample an isotropic physical R-Z lattice masked by frozen section polygons.

    A common spacing starts at sqrt(sum(section areas) / target_count).  If
    necessary, deterministic density adjustments retain a complete lattice at
    the chosen spacing; points are never randomly thinned.  A caller supplying
    ``spacing`` fixes it exactly (for an independent shifted holdout) and opts
    out of count adjustment.  Failure to reach the requested count tolerance in
    automatic mode raises rather than silently returning a sparse sample.

    Returns contiguous arrays ``xyz``, ``rz``, ``phi``, ``section_index``,
    ``weights_equal``, ``weights_volume`` (= R), ``cell_volume_weights``
    (= R h^2 dphi for one field period), ``cell_volume_weights_full``
    (= nfp times one-period weights), plus ``spacing`` and JSON metadata.
    Boundary arrays and point hashes are included for frozen-run provenance.
    No s values, fitted labels, or original fit parameters are consulted here.
    """
    polygons, phis, dphi, areas, moments = _freeze_sections(polygons_rz, phis, nfp)
    if isinstance(target_count, bool) or int(target_count) != target_count or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    target_count = int(target_count)
    if not np.isfinite(count_tolerance) or not 0 < count_tolerance < 1:
        raise ValueError("count_tolerance must lie strictly between zero and one")
    if isinstance(max_refinements, bool) or int(max_refinements) != max_refinements or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    shift = np.asarray(lattice_shift, dtype=np.float64)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)) or np.any((shift < 0) | (shift >= 1)):
        raise ValueError("lattice_shift must contain two finite phases in [0, 1)")
    fixed_spacing = spacing is not None
    h = float(spacing) if fixed_spacing else float(np.sqrt(np.sum(areas) / target_count))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("spacing must be finite and positive")
    history = []
    best = None
    for iteration in range(1 if fixed_spacing else int(max_refinements) + 1):
        section_points, count = _evaluate_lattice(polygons, h, shift)
        error = abs(count / target_count - 1)
        history.append({"spacing_m": h, "point_count": count, "relative_count_error": error})
        if best is None or error < best[0]:
            best = (error, h, section_points)
        if fixed_spacing or (error <= count_tolerance and all(len(p) for p in section_points)):
            break
        h *= np.sqrt(max(count, 1) / target_count)
    error, h, section_points = best
    counts = np.asarray([len(points) for points in section_points], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("selected spacing leaves a frozen section without points")
    if not fixed_spacing and error > count_tolerance:
        raise ValueError(f"could not reach count tolerance: target={target_count}, actual={counts.sum()}, relative_error={error:.6g}")
    section_index = np.repeat(np.arange(len(phis), dtype=np.int64), counts)
    rz = np.ascontiguousarray(np.concatenate(section_points))
    lattice_indices=np.rint(rz/h-shift).astype(np.int64)
    point_phi = np.ascontiguousarray(phis[section_index])
    xyz = np.ascontiguousarray(np.column_stack((
        rz[:, 0] * np.cos(point_phi), rz[:, 0] * np.sin(point_phi), rz[:, 1])))
    weights_equal = np.ones(len(rz), dtype=np.float64)
    weights_volume = np.ascontiguousarray(rz[:, 0].copy())
    cell_volume_weights = np.ascontiguousarray(weights_volume * h * h * dphi)
    estimated_section_areas = counts * h * h
    estimated_section_moments = np.asarray([np.sum(points[:, 0]) * h * h for points in section_points])
    frozen_hash = _hash_arrays(phis, *polygons)
    metadata = {
        "schema": "frozen-irregular-section-lattice-v1",
        "coordinate_convention": "R-Z metres; uniformly spaced geometric phi radians over one field period",
        "nfp": int(nfp), "section_count": len(phis), "dphi_rad": float(dphi),
        "target_count": target_count, "actual_count": len(rz),
        "relative_count_error": float(error), "count_tolerance": float(count_tolerance),
        "spacing_m": h, "fixed_spacing": fixed_spacing, "lattice_shift": shift.tolist(),
        "spacing_area_factor": float(h/np.sqrt(np.sum(areas)/target_count)),
        "lattice_origin_RZ_m": [0.0, 0.0],
        "sampling_measure": "uniform R-Z lattice at uniform geometric phi; no thinning",
        "equal_row_measure": "dR dZ dphi", "physical_volume_measure": "R dR dZ dphi",
        "frozen_boundary_sha256": frozen_hash,
        "points_sha256": _hash_arrays(xyz, point_phi, section_index),
        "section_point_counts": counts.tolist(), "section_areas_m2": areas.tolist(),
        "section_first_R_moments_m3": moments.tolist(),
        "section_lattice_area_m2": estimated_section_areas.tolist(),
        "section_lattice_area_relative_error": (estimated_section_areas / areas - 1).tolist(),
        "polygon_field_period_volume_m3": float(dphi * np.sum(moments)),
        "lattice_field_period_volume_m3": float(np.sum(cell_volume_weights)),
        "polygon_full_device_volume_m3": float(nfp * dphi * np.sum(moments)),
        "lattice_full_device_volume_m3": float(nfp * np.sum(cell_volume_weights)),
        "cell_volume_weights_scope": "one field period; cell_volume_weights_full multiplies by nfp",
        "lattice_volume_relative_error": float(np.sum(estimated_section_moments) / np.sum(moments) - 1),
        "strict_inside_fraction": 1.0,
        "within_section_minimum_spacing_lower_bound_m": h,
        "refinement_history": history,
        "topology_scope": "simple frozen section polygons assumed; no global surface topology certificate",
    }
    return {
        "xyz": xyz, "rz": rz, "phi": point_phi, "section_index": section_index,
        "lattice_indices": lattice_indices, "lattice_shift": shift.copy(),
        "weights_equal": weights_equal, "weights_volume": weights_volume,
        "cell_volume_weights": cell_volume_weights,
        "cell_volume_weights_full": np.ascontiguousarray(nfp * cell_volume_weights),
        "spacing": h, "metadata": metadata,
    }


def shifted_holdout(polygons_rz, phis, training_sample: dict, *, nfp: int = 4) -> dict:
    """Independent interlaced lattice at the training spacing, with no old s use.

    Both R and Z phases are shifted by h/2.  This is an independent collocation
    set inside the *same* frozen polygons, not a new fit or a radial contraction.
    It shares the geometric-phi planes; that limitation is recorded explicitly.
    """
    old_shift = np.asarray(training_sample["metadata"]["lattice_shift"])
    new_shift = tuple((old_shift + 0.5) % 1)
    result = sample_frozen_sections(
        polygons_rz, phis, target_count=training_sample["metadata"]["target_count"],
        nfp=nfp, spacing=training_sample["spacing"], lattice_shift=new_shift)
    if result["metadata"]["frozen_boundary_sha256"] != training_sample["metadata"]["frozen_boundary_sha256"]:
        raise ValueError("holdout boundary differs from the frozen training boundary")
    result["metadata"]["holdout"] = {
        "kind": "half-cell shifted R-Z lattice", "shares_geometric_phi_planes": True,
        "training_points_sha256": training_sample["metadata"]["points_sha256"],
    }
    return result
