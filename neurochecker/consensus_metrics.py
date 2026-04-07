"""Shared consensus scoring utilities for reviewer verdicts and delta masks."""

from typing import Dict, Optional

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for two binary masks."""
    if a.shape != b.shape:
        return 0.0
    a_bool = a > 0
    b_bool = b > 0
    intersection = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def acceptance_ratio(num_good: int, num_bad: int) -> Optional[float]:
    """Score how strongly reviewers accept a sample."""
    completed = int(num_good) + int(num_bad)
    if completed <= 0:
        return None
    return float(num_good) / float(completed)


def agreement_ratio(num_good: int, num_bad: int) -> Optional[float]:
    """Score reviewer agreement regardless of whether the sample passes."""
    completed = int(num_good) + int(num_bad)
    if completed <= 0:
        return None
    return float(max(int(num_good), int(num_bad))) / float(completed)


def hybrid_consensus(
    num_good: int,
    num_bad: int,
    *,
    mask_agreement: Optional[float] = None,
) -> Optional[float]:
    """Blend reviewer agreement with bad-mask alignment when edits exist."""
    verdict_agreement = agreement_ratio(num_good, num_bad)
    if verdict_agreement is None:
        return None
    if mask_agreement is None:
        return verdict_agreement
    return (verdict_agreement + float(mask_agreement)) * 0.5


def score_bundle(
    *,
    num_good: int,
    num_bad: int,
    num_pending: int = 0,
    mask_agreement: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Return the supported consensus scores for a review summary."""
    del num_pending  # Reserved for future weighting without changing the API.
    accept = acceptance_ratio(num_good, num_bad)
    agree = agreement_ratio(num_good, num_bad)
    hybrid = hybrid_consensus(num_good, num_bad, mask_agreement=mask_agreement)
    return {
        "acceptance_ratio": accept,
        "agreement_ratio": agree,
        "hybrid_consensus": hybrid,
        "consensus_score": accept,
    }
