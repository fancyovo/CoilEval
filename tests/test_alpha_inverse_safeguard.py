"""Monotone-map regressions for target-angle inversion, independent of CUDA."""
import math
import pytest

torch = pytest.importorskip('torch')

from coil_eval._autograd.boozer import invert_alpha, fourier_values


def test_high_frequency_monotone_map_inverts_without_newton_cycle():
    # The former clipped Newton retains O(0.1 rad) residual after 30 steps at
    # this target, despite the analytic lower bound d(theta+lambda)/dtheta=.01.
    target = torch.tensor([3.0564640055423147, 0., 2*math.pi], dtype=torch.float64)
    data = tuple(target.new_tensor([x]) for x in (32., 0., 0., .99/32))
    root, info = invert_alpha(data, target, target*0)
    lam, derivative, _ = fourier_values(data, root, target*0)
    torch.testing.assert_close(root+lam, target, atol=1e-10, rtol=0.)
    assert info['alpha_iterations'] <= 30
    assert float((1+derivative).min()) >= .01-1e-13


def test_inverse_keeps_winding_for_large_constant_correction():
    target = torch.tensor([-19., .2, 50.], dtype=torch.float64)
    data = tuple(target.new_tensor([x]) for x in (0., 0., 20., 0.))
    root, _ = invert_alpha(data, target, target*0)
    torch.testing.assert_close(root, target-20., atol=1e-11, rtol=0.)


def test_safeguarded_root_keeps_exact_coefficient_and_angle_derivatives():
    parameters = torch.tensor([.99/32, 3.0564640055423147, 0.],
                              dtype=torch.float64, requires_grad=True)
    amplitude, target, phi = parameters.unbind()
    data = (target.new_tensor([32.]), target.new_tensor([-4.]),
            amplitude[None]*0, amplitude[None])
    root, _ = invert_alpha(data, target, phi)
    gradient, = torch.autograd.grad(root, parameters)
    phase = 32*root.detach()-4*phi.detach()
    jacobian = 1+32*amplitude.detach()*phase.cos()
    expected = torch.stack((-phase.sin()/jacobian, 1/jacobian,
                            4*amplitude.detach()*phase.cos()/jacobian))
    torch.testing.assert_close(gradient, expected, atol=1e-11, rtol=1e-10)


def test_safeguard_preserves_noninvertible_map_rejection():
    target = torch.tensor(math.pi, dtype=torch.float64)
    data = tuple(target.new_tensor([x]) for x in (1., 0., 0., 1.1))
    with pytest.raises(ValueError, match='noninvertible alpha coordinate map'):
        invert_alpha(data, target, target*0)


def test_broadcast_angle_gradients_reach_original_inputs():
    target = torch.tensor([[.3, .6, 1.2]], dtype=torch.float64, requires_grad=True)
    phi = torch.tensor([[.1], [.2]], dtype=torch.float64, requires_grad=True)
    data = tuple(target.new_tensor([x]) for x in (2., -3., 0., .1))
    root, _ = invert_alpha(data, target, phi)
    dt, dp = torch.autograd.grad(root.sum(), (target, phi))
    phase = 2*root.detach()-3*phi.detach()
    jacobian = 1+.2*phase.cos()
    torch.testing.assert_close(dt, (1/jacobian).sum(0, keepdim=True))
    torch.testing.assert_close(dp, (.3*phase.cos()/jacobian).sum(1, keepdim=True))
