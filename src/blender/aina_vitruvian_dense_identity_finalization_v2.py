#!/usr/bin/env python3
"""Metric-corrected entry point for dense AINA identity finalization."""
from __future__ import annotations

import numpy as np

import aina_vitruvian_dense_identity_finalization as base


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    # Selected dense indices are not in the canonical 68-point order, so use a
    # robust facial horizontal span rather than arbitrary array positions.
    span = float(np.percentile(target[:, 0], 98) - np.percentile(target[:, 0], 2))
    span = max(span, 1.0)
    value = np.sum(weights * np.sum((prediction - target) ** 2, axis=1)) / max(float(weights.sum()), 1e-9)
    return float(np.sqrt(value) / span)


base.weighted_rmse = weighted_rmse


if __name__ == "__main__":
    base.main()
