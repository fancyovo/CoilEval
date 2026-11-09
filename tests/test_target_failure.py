from types import SimpleNamespace

import numpy as np

from coil_eval._two_stage.boundary import MaxClosedBoundaryError
from coil_eval._two_stage.model import PsiModel, build_modes
from coil_eval._two_stage.target_failure import explain_target_failure, training_domain_error


def model(terms):
    coeff=[terms.get((m.a,m.b),0.) for m in build_modes(4,0)]
    axis=np.tile([.5,0.,0.,0.],(16,1))
    return PsiModel(coeff,axis,nfp=2,a=.1,degree=4,mtor=0)


def explain(terms, stage=2, maximum=.01):
    config=SimpleNamespace(psi_m_tor=0,psi_n_phi=4,flux_phi_count=4,flux_theta_count=32,
                           surface_max_radius_scale=1.,target_volume=.2)
    diag=dict(error_message='old failure',target_volume_maximum=maximum)
    explain_target_failure(diag,model(terms),config,stage)
    return diag


def test_only_the_genuine_limiting_fold_is_reported_as_nonmonotone():
    # s = u²-u⁴, versus the monotone s = u² at the same physical cap.
    fold=explain({(0,2):1,(4,0):-1,(2,2):-2,(0,4):-1})
    cap=explain({(0,2):1})
    assert fold['failure_reason']=='s_radial_fold_capacity'
    assert fold['target_branch_diagnostic']['capacity_source']=='radial_fold'
    assert 'radially nonmonotone' in fold['error_message']
    assert cap['failure_reason']=='s_radius_cap_capacity'
    assert cap['target_branch_diagnostic']['capacity_source']=='configured_radius'
    assert cap['native_error_message']=='old failure'


def test_domain_failure_names_first_fit_and_retains_witness():
    error=MaxClosedBoundaryError('positive quadratic required',
        dict(critical_root_diagnostics=dict(failure_reason='invalid_near_axis_radial_polynomial',quadratic=-.01)))
    updated=training_domain_error(error,1)
    assert updated.diagnostics['failure_reason']=='s_near_axis_nonmonotone'
    assert updated.diagnostics['psi_fit_stage']==1
    assert 'positive quadratic required' in str(updated)


def test_early_target_branch_and_inverse_failure_stay_distinct():
    assert explain({(0,2):-1})['failure_reason']=='s_near_axis_nonmonotone'
    assert explain({(0,2):1},maximum=.3)['failure_reason']=='target_volume_inverse_failed_within_capacity'
