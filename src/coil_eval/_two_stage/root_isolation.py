"""Exact rational real-root isolation for one radial polynomial.

The input float coefficients are treated as exact binary rationals. This handles
rare companion-eigenvalue ambiguities while preserving the batched fast path.
It adds no external numerical dependency.
"""
from __future__ import annotations
from fractions import Fraction
import math


def _trim(poly):
    poly = list(poly)
    while poly and not poly[-1]:
        poly.pop()
    return poly


def _evaluate(poly, x):
    answer = Fraction(0)
    for coefficient in reversed(poly):
        answer = answer*x+coefficient
    return answer


def _remainder(dividend, divisor):
    remainder = _trim(dividend)
    while len(remainder) >= len(divisor):
        factor = remainder[-1]/divisor[-1]
        offset = len(remainder)-len(divisor)
        for j, coefficient in enumerate(divisor):
            remainder[j+offset] -= factor*coefficient
        remainder = _trim(remainder)
    return remainder


def _normalized(poly):
    poly = _trim(poly)
    if not poly:
        return poly
    scale = abs(poly[-1])
    return [coefficient/scale for coefficient in poly]


def _exact_quotient(dividend, divisor):
    remainder = _trim(dividend)
    quotient = [Fraction(0)]*(len(remainder)-len(divisor)+1)
    while len(remainder) >= len(divisor):
        factor = remainder[-1]/divisor[-1]
        offset = len(remainder)-len(divisor)
        quotient[offset] += factor
        for j, coefficient in enumerate(divisor):
            remainder[j+offset] -= factor*coefficient
        remainder = _trim(remainder)
    if remainder:
        raise ArithmeticError('polynomial square-free division was not exact')
    return _trim(quotient)


def _sturm(poly):
    first = _normalized(poly)
    second = _normalized([i*poly[i] for i in range(1, len(poly))])
    sequence = [first]
    if not second:
        return sequence
    sequence.append(second)
    while True:
        remainder = _normalized([-x for x in _remainder(sequence[-2], sequence[-1])])
        if not remainder:
            return sequence
        sequence.append(remainder)


def _variations(sequence, x):
    signs = []
    for polynomial in sequence:
        value = _evaluate(polynomial, x)
        if value:
            signs.append(1 if value > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def isolate_positive_roots(poly, maximum):
    """Disjoint exact intervals, each containing one distinct real root."""
    sequence = _sturm(poly)
    # At an exact repeated root, every member of the unreduced Sturm sequence
    # can vanish. Remove the exact gcd so variations there have the proper
    # one-sided meaning; interval signs still use the original polynomial.
    if len(sequence[-1]) > 1:
        poly = _exact_quotient(poly, sequence[-1])
        sequence = _sturm(poly)
    left = Fraction(0)
    right = Fraction.from_float(float(maximum))
    tolerance = max(Fraction(1), right)/2**48
    roots = []
    count = _variations(sequence, left)-_variations(sequence, right)
    # Sturm variations at a root equal the right-hand limiting variation. A
    # root exactly at the physical cap is irrelevant to the open ray domain.
    if not _evaluate(poly, right):
        count -= 1

    def split(lo, hi, number, depth):
        if not number:
            return
        if depth > 512:
            raise ArithmeticError('exact root isolation did not separate roots within 512 subdivisions')
        if number == 1 and hi-lo <= tolerance and _evaluate(poly, lo) and _evaluate(poly, hi):
            roots.append((lo, hi))
            return
        mid = (lo+hi)/2
        value = _evaluate(poly, mid)
        number_left = _variations(sequence, lo)-_variations(sequence, mid)
        if not value:
            number_left -= 1
            split(lo, mid, number_left, depth+1)
            roots.append((mid, mid))
            split(mid, hi, number-number_left-1, depth+1)
        else:
            split(lo, mid, number_left, depth+1)
            split(mid, hi, number-number_left, depth+1)

    split(left, right, count, 0)
    roots.sort()
    return roots


def certified_first_branch(polynomial, maximum_u):
    p = [Fraction.from_float(float(value)) for value in polynomial]
    if len(p) < 3 or p[0] or p[1] or p[2] <= 0 or not math.isfinite(maximum_u) or maximum_u <= 0:
        raise ValueError('positive quadratic axis-connected radial polynomial required')
    q = _trim([i*p[i] for i in range(2, len(p))])
    roots = isolate_positive_roots(q, maximum_u)
    maximum = Fraction.from_float(float(maximum_u))
    points = [Fraction(0)]
    for lo, hi in roots:
        points.extend((lo, hi))
    points.append(maximum)
    samples = [(points[2*i]+points[2*i+1])/2 for i in range(len(roots)+1)]
    signs = [1 if _evaluate(q, sample) > 0 else -1 if _evaluate(q, sample) < 0 else 0 for sample in samples]
    if not signs or signs[0] != 1 or 0 in signs:
        raise ArithmeticError('root isolation did not certify a positive connected initial interval')
    events, first_fold = [], None
    for index, (lo, hi) in enumerate(roots):
        left, right = signs[index], signs[index+1]
        root = float((lo+hi)/2)
        kind = 'flat_nonfold' if left == right else 'fold_maximum' if left > 0 else 'fold_minimum'
        events.append(dict(u=root, interval=[float(lo), float(hi)], left_interval_sign=left,
            right_interval_sign=right, kind=kind, exact_root=lo == hi))
        if first_fold is None and left > 0 and right < 0:
            first_fold = root
    cap = float(maximum_u if first_fold is None else first_fold)
    capacity = _evaluate(p, Fraction.from_float(cap))
    if cap <= 0 or capacity <= 0:
        raise ArithmeticError('nonpositive certified axis-connected capacity')
    return dict(maximum_u=cap, first_fold_u=first_fold,
        reason='radius_cap' if first_fold is None else 'first_genuine_fold',
        critical_points=events, interval_signs=signs,
        flat_nonfold_count=sum(event['kind'] == 'flat_nonfold' and event['u'] < cap for event in events),
        root_isolation='exact binary-rational Sturm sequence and interval signs',
        coefficient_semantics='exact real polynomial represented by input binary64 coefficients')


