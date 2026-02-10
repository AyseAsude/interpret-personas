"""Offline builder for SAE persona visualization bundles."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from interpret_personas.config import VisualizationConfig
from interpret_personas.utils import ensure_dir

_LOGGER = logging.getLogger(__name__)


def _safe_float(value: float | np.floating, digits: int = 6) -> float:
    """Convert float-like value into a JSON-safe python float."""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return round(float(value), digits)


def _load_description_cache(cache_path: Path | None) -> dict[int, dict[str, str | None]]:
    """Load optional feature descriptions from JSON."""
    if cache_path is None:
        return {}

    def _coerce_feature_id(raw: object) -> int | None:
        try:
            feat_id = int(raw)
        except (TypeError, ValueError):
            return None
        if feat_id < 0:
            return None
        return feat_id

    def _coerce_text(raw: object) -> str | None:
        if not isinstance(raw, str):
            return None
        value = raw.strip()
        return value if value else None

    with open(cache_path, "r") as f:
        payload = json.load(f)

    descriptions: dict[int, dict[str, str | None]] = {}

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict) or "feature_id" not in item:
                continue
            feat_id = _coerce_feature_id(item["feature_id"])
            if feat_id is None:
                continue
            descriptions[feat_id] = {
                "description": _coerce_text(item.get("description")),
                "url": _coerce_text(item.get("url")),
            }
    elif isinstance(payload, dict):
        for key, value in payload.items():
            feat_id = _coerce_feature_id(key)
            if feat_id is None:
                continue
            if isinstance(value, dict):
                descriptions[feat_id] = {
                    "description": _coerce_text(value.get("description")),
                    "url": _coerce_text(value.get("url")),
                }
            elif isinstance(value, str):
                descriptions[feat_id] = {
                    "description": _coerce_text(value),
                    "url": None,
                }
    else:
        _LOGGER.warning(
            "Description cache payload at %s must be a list or dict, got %s",
            cache_path,
            type(payload).__name__,
        )

    return descriptions


def _compute_split_half_stability(
    role_names: np.ndarray,
    features_dir: Path,
    strategy: str,
    sae_dim: int,
    split_seed: int,
) -> np.ndarray:
    """Compute split-half stability for each SAE feature."""
    n_roles = len(role_names)
    key = f"{strategy}_features"
    a_half = np.zeros((n_roles, sae_dim), dtype=np.float32)
    b_half = np.zeros((n_roles, sae_dim), dtype=np.float32)

    rng = np.random.RandomState(split_seed)

    for role_idx, role_name in enumerate(role_names):
        role_file = features_dir / f"{role_name}.npz"
        if not role_file.exists():
            raise FileNotFoundError(f"Missing role feature file: {role_file}")

        with np.load(role_file) as role_npz:
            if key not in role_npz:
                raise KeyError(f"Missing key '{key}' in {role_file}")
            response_features = role_npz[key]

        if response_features.ndim != 2:
            raise ValueError(
                f"Expected 2D response features for {role_name}, got {response_features.shape}"
            )
        if response_features.shape[1] != sae_dim:
            raise ValueError(
                f"SAE dim mismatch for {role_name}: "
                f"expected {sae_dim}, got {response_features.shape[1]}"
            )
        if response_features.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 responses for split-half stability: {role_name}"
            )

        indices = rng.permutation(response_features.shape[0])
        midpoint = response_features.shape[0] // 2
        first_half = response_features[indices[:midpoint]]
        second_half = response_features[indices[midpoint:]]

        a_half[role_idx] = first_half.mean(axis=0)
        b_half[role_idx] = second_half.mean(axis=0)

    numerator = (a_half * b_half).sum(axis=0)
    denominator = np.linalg.norm(a_half, axis=0) * np.linalg.norm(b_half, axis=0)

    stability = np.divide(
        numerator,
        denominator,
        out=np.zeros(sae_dim, dtype=np.float32),
        where=denominator > 0,
    )
    return stability


def _compute_cosine_neighbors(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine neighbors in high-D feature space."""
    n = x.shape[0]
    if n <= 1 or k <= 0:
        return (
            np.empty((n, 0), dtype=np.int32),
            np.empty((n, 0), dtype=np.float32),
        )

    k_eff = min(k, n - 1)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff + 1)
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)

    neighbor_indices = np.full((n, k_eff), -1, dtype=np.int32)
    neighbor_similarities = np.zeros((n, k_eff), dtype=np.float32)

    for row_idx in range(n):
        row_indices = indices[row_idx]
        row_distances = distances[row_idx]
        mask = row_indices != row_idx
        filtered_indices = row_indices[mask][:k_eff]
        filtered_similarities = 1.0 - row_distances[mask][:k_eff]
        neighbor_indices[row_idx, : len(filtered_indices)] = filtered_indices
        neighbor_similarities[row_idx, : len(filtered_similarities)] = (
            filtered_similarities
        )

    return neighbor_indices, neighbor_similarities


def _compute_knn_overlap(
    high_d_vectors: np.ndarray,
    low_d_vectors: np.ndarray,
    k: int,
) -> float:
    """Compute neighborhood preservation between high-D and low-D spaces."""
    n = high_d_vectors.shape[0]
    if n <= 1 or k <= 0:
        return 0.0

    k_eff = min(k, n - 1)
    high_nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff + 1)
    low_nn = NearestNeighbors(metric="euclidean", algorithm="auto", n_neighbors=k_eff + 1)

    high_nn.fit(high_d_vectors)
    low_nn.fit(low_d_vectors)

    high_indices = high_nn.kneighbors(high_d_vectors, return_distance=False)[:, 1:]
    low_indices = low_nn.kneighbors(low_d_vectors, return_distance=False)[:, 1:]

    overlaps = np.zeros(n, dtype=np.float32)
    for row_idx in range(n):
        a_set = set(high_indices[row_idx].tolist())
        b_set = set(low_indices[row_idx].tolist())
        overlaps[row_idx] = len(a_set.intersection(b_set)) / k_eff

    return float(overlaps.mean())


def _top_roles_for_feature(
    role_values: np.ndarray,
    role_names: np.ndarray,
    top_n: int,
) -> list[dict[str, float | int | str]]:
    """Build top role summary for a single feature."""
    order = np.argsort(role_values)[::-1][:top_n]
    total = role_values.sum()

    top_roles: list[dict[str, float | int | str]] = []
    for role_idx in order:
        value = float(role_values[role_idx])
        share = value / total if total > 0 else 0.0
        top_roles.append(
            {
                "role_idx": int(role_idx),
                "role": str(role_names[role_idx]),
                "activation": _safe_float(value),
                "share": _safe_float(share),
            }
        )
    return top_roles


def _save_feature_csv(feature_rows: list[dict], output_file: Path) -> None:
    """Save flattened feature table for quick export workflows."""
    columns = [
        "feature_row",
        "feature_id",
        "preferred_role",
        "score",
        "stability",
        "sd",
        "mu",
        "pref_ratio",
        "active_frac",
        "cv",
        "mean_activation",
        "max_activation",
        "bridge_entropy",
        "description",
        "neuronpedia_url",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in feature_rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "feature_row": row["feature_row"],
                    "feature_id": row["feature_id"],
                    "preferred_role": row["preferred_role"],
                    "score": metrics["score"],
                    "stability": metrics["stability"],
                    "sd": metrics["sd"],
                    "mu": metrics["mu"],
                    "pref_ratio": metrics["pref_ratio"],
                    "active_frac": metrics["active_frac"],
                    "cv": metrics["cv"],
                    "mean_activation": metrics["mean_activation"],
                    "max_activation": metrics["max_activation"],
                    "bridge_entropy": metrics["bridge_entropy"],
                    "description": row["description"] or "",
                    "neuronpedia_url": row["neuronpedia_url"] or "",
                }
            )


def build_visualization_bundle(config: VisualizationConfig, logger) -> Path:
    """
    Build one visualization bundle from aggregated and per-response feature outputs.

    Returns:
        Path to generated bundle JSON.
    """
    logger.info("Loading aggregated role-level data...")

    try:
        from umap import UMAP  # type: ignore
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "umap-learn is required for visualization bundle build. "
            "Install visualization deps with: uv pip install -e '.[visualization]'"
        ) from error
    with np.load(config.aggregated_file, allow_pickle=True) as data:
        features = data["features"].astype(np.float32, copy=False)
        role_names = np.array([str(role) for role in data["role_names"]])

    n_roles, sae_dim = features.shape
    logger.info(f"Role matrix shape: {features.shape}")

    logger.info("Step 1/6: Applying log transform and basic variance filter...")
    role_matrix = np.log1p(features)
    mu = role_matrix.mean(axis=0)
    sd = role_matrix.std(axis=0)

    alive = mu > 0
    sd_threshold = float(np.median(sd[alive])) if alive.any() else 0.0
    pass_basic = alive & (sd >= sd_threshold)
    logger.info(
        f"Basic filter retained {int(pass_basic.sum()):,}/{sae_dim:,} features "
        f"(sd threshold={sd_threshold:.6f})"
    )

    logger.info("Step 2/6: Computing split-half stability...")
    stability = _compute_split_half_stability(
        role_names=role_names,
        features_dir=config.features_dir,
        strategy=config.strategy,
        sae_dim=sae_dim,
        split_seed=config.split_seed,
    )

    logger.info("Step 3/6: Ranking features by stability x variance...")
    score = stability * sd
    score[~pass_basic] = -1.0

    ranking = np.argsort(score)[::-1]
    top_k = min(config.top_k, sae_dim)
    selected = ranking[:top_k]
    logger.info(f"Selected top {top_k:,} features")

    logger.info("Step 4/6: Precomputing high-D structures...")
    x = role_matrix[:, selected].T.astype(np.float32, copy=False)
    preferred_role_idx = role_matrix[:, selected].argmax(axis=0)
    role_values_selected = role_matrix[:, selected]

    sorted_role_values = np.sort(role_values_selected, axis=0)
    top1 = sorted_role_values[-1]
    top2 = sorted_role_values[-2] if n_roles > 1 else np.zeros_like(top1)
    pref_ratio = np.divide(top1, np.maximum(top2, 1e-8))

    active_frac = (role_values_selected > 0).mean(axis=0)
    cv = np.divide(sd[selected], np.maximum(mu[selected], 1e-8))
    mean_activation = features[:, selected].mean(axis=0)
    max_activation = features[:, selected].max(axis=0)

    non_negative_vals = np.clip(role_values_selected, 0.0, None)
    totals = non_negative_vals.sum(axis=0, keepdims=True)
    probs = np.divide(
        non_negative_vals,
        totals,
        out=np.zeros_like(non_negative_vals),
        where=totals > 0,
    )
    log_probs = np.zeros_like(probs)
    np.log(probs, out=log_probs, where=probs > 0)
    entropy = -(probs * log_probs).sum(axis=0)
    if n_roles > 1:
        entropy /= np.log(float(n_roles))

    neighbor_indices, neighbor_similarities = _compute_cosine_neighbors(
        x,
        config.neighbor_k,
    )

    role_vectors = role_matrix[:, selected]
    role_similarity = cosine_similarity(role_vectors)

    logger.info("Step 5/6: Computing map coordinates and quality metrics...")
    umap = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=config.umap_random_state,
    )
    coords_umap = umap.fit_transform(x)

    pca = PCA(n_components=2, random_state=config.random_seed)
    coords_pca = pca.fit_transform(x)

    knn_overlap = _compute_knn_overlap(
        high_d_vectors=x,
        low_d_vectors=coords_umap,
        k=config.quality_k,
    )

    logger.info("Step 6/6: Assembling bundle payload...")
    descriptions = _load_description_cache(config.description_cache)

    feature_rows: list[dict] = []
    for feature_row, feature_id in enumerate(selected.tolist()):
        desc_item = descriptions.get(feature_id, {})
        feature_rows.append(
            {
                "feature_row": feature_row,
                "feature_id": int(feature_id),
                "preferred_role_idx": int(preferred_role_idx[feature_row]),
                "preferred_role": str(role_names[preferred_role_idx[feature_row]]),
                "top_roles": _top_roles_for_feature(
                    role_values_selected[:, feature_row], role_names, top_n=3
                ),
                "metrics": {
                    "score": _safe_float(score[feature_id]),
                    "stability": _safe_float(stability[feature_id]),
                    "sd": _safe_float(sd[feature_id]),
                    "mu": _safe_float(mu[feature_id]),
                    "pref_ratio": _safe_float(pref_ratio[feature_row]),
                    "active_frac": _safe_float(active_frac[feature_row]),
                    "cv": _safe_float(cv[feature_row]),
                    "mean_activation": _safe_float(mean_activation[feature_row]),
                    "max_activation": _safe_float(max_activation[feature_row]),
                    "bridge_entropy": _safe_float(entropy[feature_row]),
                },
                "description": desc_item.get("description"),
                "neuronpedia_url": desc_item.get("url"),
            }
        )

    dataset_dir = ensure_dir(config.output_dir / config.dataset_name)
    bundle_file = dataset_dir / "bundle.json"
    summary_csv = dataset_dir / "features.csv"

    bundle_payload = {
        "dataset": {
            "name": config.dataset_name,
            "aggregated_file": str(config.aggregated_file),
            "features_dir": str(config.features_dir),
            "strategy": config.strategy,
            "n_roles": int(n_roles),
            "sae_dim": int(sae_dim),
            "top_k": int(top_k),
            "random_seed": int(config.random_seed),
            "split_seed": int(config.split_seed),
            "umap": {
                "n_neighbors": int(config.umap_n_neighbors),
                "min_dist": float(config.umap_min_dist),
                "random_state": int(config.umap_random_state),
            },
            "selection": {
                "alive_count": int(alive.sum()),
                "pass_basic_count": int(pass_basic.sum()),
                "sd_threshold": _safe_float(sd_threshold),
            },
        },
        "guardrails": {
            "note": "Use map for navigation only. Related features and role similarity use high-D cosine space.",
            "knn_overlap_at_k": int(config.quality_k),
            "knn_overlap_score": _safe_float(knn_overlap),
        },
        "roles": role_names.tolist(),
        "feature_ids": selected.astype(int).tolist(),
        "coords": {
            "umap": np.asarray(coords_umap, dtype=np.float32).round(6).tolist(),
            "pca": np.asarray(coords_pca, dtype=np.float32).round(6).tolist(),
        },
        "neighbors": {
            "k": int(neighbor_indices.shape[1]),
            "indices": neighbor_indices.astype(int).tolist(),
            "similarities": np.asarray(
                neighbor_similarities, dtype=np.float32
            ).round(6).tolist(),
        },
        "role_similarity": np.asarray(role_similarity, dtype=np.float32).round(6).tolist(),
        "features": feature_rows,
    }

    with open(bundle_file, "w") as f:
        json.dump(bundle_payload, f, ensure_ascii=True, indent=2)

    _save_feature_csv(feature_rows, summary_csv)

    logger.info(f"Bundle written: {bundle_file}")
    logger.info(f"Feature table written: {summary_csv}")
    return bundle_file
