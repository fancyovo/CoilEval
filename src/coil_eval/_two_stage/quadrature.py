"""Fixed polar nodes with area weights for a moving star-shaped fit domain."""
from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=16)
def _radial_gauss_rule(count):
    """Reuse the fixed rule; each call still owns its returned radial arrays."""
    gauss,weights=np.polynomial.legendre.leggauss(count)
    gauss.setflags(write=False);weights.setflags(write=False)
    return gauss,weights


def sample_polar_quadrature(boundary, target_count=82368, *, nfp=4, theta_count=128):
    """Integrate uniform Cartesian section area using fixed polar nodes.

    A tensor product of uniform geometric phi/theta and Gauss-Legendre nodes
    in ``u=(r/r_boundary)**2`` maps every node into the domain. The Jacobian
    contributes ``r_boundary**2`` to the least-squares row weight. Weights are
    normalized across all sections, retaining their relative areas. No node
    enters or leaves under a continuous boundary displacement.

    ``target_count`` selects the nearest positive integer radial order. With
    48 phi sections and 128 rays, the default is 13 nodes/ray, or 79,872 rows.
    Row order is phi, theta, radial node. ``spacing`` is None because this is
    not a Cartesian lattice; ``equivalent_spacing_m`` is only a diagnostic.
    """
    if isinstance(nfp, bool) or int(nfp) != nfp or nfp < 1:
        raise ValueError('nfp must be a positive integer')
    if isinstance(target_count, bool) or int(target_count) != target_count or target_count < 1:
        raise ValueError('target_count must be a positive integer')
    if isinstance(theta_count, bool) or int(theta_count) != theta_count or theta_count < 8:
        raise ValueError('theta_count must be an integer >= 8')
    phi = np.asarray(boundary['phis'], dtype=np.float64)
    theta = np.asarray(boundary['theta'], dtype=np.float64)
    radii = np.asarray(boundary['radii'], dtype=np.float64)
    polygons = np.asarray(boundary['polygons_rz'], dtype=np.float64)
    if (phi.ndim != 1 or len(phi) < 2 or theta.ndim != 1 or len(theta) % theta_count or
            radii.shape != (len(phi), len(theta)) or polygons.shape != (*radii.shape, 2)):
        raise ValueError('polar quadrature requires aligned boundary sections and a theta-count divisor')
    if (not np.isfinite(phi).all() or not np.isfinite(theta).all() or
            not np.isfinite(radii).all() or not np.isfinite(polygons).all() or np.any(radii <= 0)):
        raise ValueError('polar quadrature boundary must be finite with positive radii')
    dphi = 2*np.pi/(int(nfp)*len(phi))
    if (not np.allclose(np.diff(phi), dphi, rtol=1e-9, atol=1e-12) or
            not np.allclose(theta, np.arange(len(theta))*2*np.pi/len(theta), rtol=1e-9, atol=1e-12)):
        raise ValueError('polar quadrature requires uniform complete phi/theta periods without repeated endpoints')
    theta_indices = np.arange(0, len(theta), len(theta)//int(theta_count), dtype=np.int64)
    angle = theta[theta_indices]
    edge = radii[:, theta_indices]
    axis_R = np.mean(polygons[..., 0]-radii*np.cos(theta), axis=1)
    axis_Z = np.mean(polygons[..., 1]-radii*np.sin(theta), axis=1)
    radial_count = max(1, int(np.floor(int(target_count)/(len(phi)*int(theta_count))+.5)))
    gauss, weights = _radial_gauss_rule(radial_count)
    radial_u = (gauss+1)/2
    radial_weights = weights/2
    radius = edge[..., None]*np.sqrt(radial_u)
    R = axis_R[:, None, None]+radius*np.cos(angle)[None, :, None]
    Z = axis_Z[:, None, None]+radius*np.sin(angle)[None, :, None]
    rz = np.ascontiguousarray(np.stack((R, Z), axis=-1).reshape(-1, 2))
    if not np.isfinite(rz).all() or np.any(rz[:, 0] <= 0):
        raise ValueError('polar quadrature nodes require finite positive cylindrical R')
    count = len(rz)
    section = np.repeat(np.arange(len(phi), dtype=np.int64), len(angle)*radial_count)
    point_phi = np.repeat(phi, len(angle)*radial_count)
    raw = np.broadcast_to(edge[..., None]**2*radial_weights, radius.shape).reshape(-1)
    row_weights = np.ascontiguousarray(raw/np.mean(raw))
    mass = raw*(.5*2*np.pi/int(theta_count)*dphi)
    xyz = np.column_stack((rz[:, 0]*np.cos(point_phi), rz[:, 0]*np.sin(point_phi), rz[:, 1]))
    area = .5*(polygons[..., 0]*np.roll(polygons[..., 1], -1, axis=1)-
               np.roll(polygons[..., 0], -1, axis=1)*polygons[..., 1]).sum(axis=1)
    if np.any(area <= 0):
        raise ValueError('polar quadrature requires positive oriented section areas')
    result = dict(kind='polar_quadrature', rz=rz, xyz=xyz, phi=point_phi,
        section_index=section, theta_indices=theta_indices, radial_u=radial_u,
        radial_weights=radial_weights, row_weights=row_weights,
        cell_area_phi_weights=mass, cell_volume_weights=rz[:, 0]*mass,
        cell_volume_weights_full=int(nfp)*rz[:, 0]*mass,
        spacing=None,equivalent_spacing_m=float(np.sqrt(area.sum()/count)),
        lattice_indices=np.empty((0,2),dtype=np.int64),
        metadata=dict(kind='polar_quadrature', target_count=int(target_count), point_count=count,
            geometric_phi_count=len(phi), theta_count=int(theta_count), radial_count=radial_count,
            radial_coordinate='u=(r/r_boundary)^2', radial_rule='Gauss-Legendre on (0,1)',
            row_weight_measure='dR dZ dphi; normalized to arithmetic mean one over all sections',
            node_order='phi,theta,radial', spacing_kind='not applicable to polar quadrature',
            discrete_choices='fixed angular/radial quadrature nodes; active capacity ray remains a local branch'))
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result
