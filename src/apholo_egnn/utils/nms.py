import numpy as np


def nms_order(coords: np.ndarray, confidence: np.ndarray, radius: float) -> list[int]:
    """Order candidates by confidence with NMS-based spatial repulsion (non maxima suppression).

    Picks the highest-confidence candidate, then each next pick must be at
    least *radius* Å from every already-chosen candidate. If no remaining
    candidate satisfies the constraint, fall back to the next best by
    confidence.

    Parameters
    ----------
    coords:
        ``[V, 3]`` candidate positions.
    confidence:
        ``[V]`` confidence scores. NaN sorts last.
    radius:
        Minimum separation in Å between successive picks.

    Returns
    -------
    list of int
        A permutation of all V indices in NMS order.
    """
    conf = np.where(np.isfinite(confidence), confidence, -np.inf)
    remaining = list(np.argsort(-conf))
    chosen: list[int] = []
    while remaining:
        pick = next(
            (
                i
                for i in remaining
                if all(float(np.linalg.norm(coords[i] - coords[c])) >= radius for c in chosen)
            ),
            remaining[0],
        )
        chosen.append(pick)
        remaining.remove(pick)
    return chosen
