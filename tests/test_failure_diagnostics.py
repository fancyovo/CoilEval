"""Failure explanations retain numerical status and native evidence."""
import numpy as np
from coil_eval._binding import _SgpuScoreResult, _score_result_dict


def test_missing_axis_message_does_not_claim_global_absence():
    raw = _SgpuScoreResult();raw.status = 1;raw.axis_candidate_count = 16
    result = _score_result_dict(raw)
    assert result['status'] == 'no_axis'
    assert result['diagnostics']['failure_reason'] == 'axis_search_no_accepted_elliptic_root'
    assert result['diagnostics']['native_error_message'] == ''
    assert 'configured residual tolerance' in result['diagnostics']['error_message']


def test_capacity_rejection_is_distinct_from_inverse_failure():
    raw = _SgpuScoreResult();raw.status = 2
    raw.error_message = b'target volume outside certified geometry domain or inverse solve failed'
    raw.target_volume_requested = .2;raw.target_volume_maximum = .1
    capacity = _score_result_dict(raw)
    assert capacity['status'] == 'no_surface'
    assert capacity['diagnostics']['failure_reason'] == 'target_volume_exceeds_certified_capacity'
    assert capacity['diagnostics']['target_volume_requested'] == .2
    assert 'sampled angular rays' in capacity['diagnostics']['capacity_scope']
    raw.target_volume_maximum = .3
    inverse = _score_result_dict(raw)
    assert inverse['diagnostics']['failure_reason'] == 'target_volume_inverse_failed_within_capacity'
    assert inverse['diagnostics']['native_error_message'] == raw.error_message.decode()


def test_ambiguous_or_successful_records_are_not_reclassified():
    raw = _SgpuScoreResult();raw.status = 2
    raw.error_message = b'target volume outside certified geometry domain or inverse solve failed'
    raw.target_volume_requested = .2;raw.target_volume_maximum = np.nan
    assert 'failure_reason' not in _score_result_dict(raw)['diagnostics']
    raw.status = 0;raw.target_volume_maximum = .1
    assert 'failure_reason' not in _score_result_dict(raw)['diagnostics']


def test_forward_volume_error_is_not_labeled_as_a_flux_threshold_failure():
    raw = _SgpuScoreResult();raw.status = 4
    raw.error_message = b'target-volume forward precision tolerance exceeded'
    result = _score_result_dict(raw)
    assert result['status'] == 'flux_rejected'
    d = result['diagnostics']
    assert d['failure_reason'] == 'target_volume_forward_precision_exceeded'
    assert d['failure_stage'] == 'target_volume_forward_validation'
    assert d['native_error_message'] == raw.error_message.decode()
    raw.error_message = b'flux quality threshold exceeded'
    assert 'failure_reason' not in _score_result_dict(raw)['diagnostics']
