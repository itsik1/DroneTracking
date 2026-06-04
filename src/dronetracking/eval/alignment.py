"""Aligning an estimated point cloud onto truth for scoring.

A relative layout is only recoverable up to a *similarity* — and, because pure
distance geometry cannot observe chirality, that similarity may include a
reflection. To score an estimate fairly we first rigidly align it to the truth
(no scaling: the metric scale is fixed by the anchors elsewhere, and we want
errors in meters), allowing a reflection, and then measure residuals.
"""

from __future__ import annotations

import numpy as np

from dronetracking import transforms
from dronetracking.transforms import Similarity


def align_to_truth(estimated: np.ndarray, truth: np.ndarray) -> Similarity:
    """Best rigid (reflection-allowed, no-scale) transform mapping ``estimated`` onto ``truth``.

    Returns a :class:`~dronetracking.transforms.Similarity` ``sim`` such that
    ``sim.apply(estimated)`` is the least-squares fit to ``truth``. Use
    ``sim.is_reflection`` to report whether a chirality flip was needed.
    """
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return transforms.umeyama(
        estimated, truth, with_scaling=False, allow_reflection=True
    )
