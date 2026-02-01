"""Feature aggregation functions for multi-level pooling."""


import numpy as np


def aggregate_mean(features: np.ndarray) -> np.ndarray:
    """
    Mean pooling across axis 0.

    Args:
        features: Array to aggregate

    Returns:
        Mean-pooled vector
    """
    return np.mean(features, axis=0)


def aggregate_max(features: np.ndarray) -> np.ndarray:
    """
    Max pooling across axis 0.

    Args:
        features: Array to aggregate

    Returns:
        Max-pooled vector
    """
    return np.max(features, axis=0)


AGGREGATION_FUNCTIONS = {
    "mean": aggregate_mean,
    "max": aggregate_max,
}


def aggregate_role(response_features: list[np.ndarray]) -> dict[str, np.ndarray]:
    """
    Aggregate all responses for a role.

    Applies all aggregation strategies (mean, max) across responses.

    Args:
        response_features: List of (n_features,) vectors from individual responses

    Returns:
        Dict mapping strategy name to role-level vector
        Example: {"mean": role_vector, "max": role_vector}
    """
    # Stack response vectors
    stacked = np.array(response_features)  # Shape: (n_responses, n_features)

    return {
        name: func(stacked)
        for name, func in AGGREGATION_FUNCTIONS.items()
    }
