#!/usr/bin/env python3
"""Fetch and cache Neuronpedia feature descriptions for selected bundle features.

Usage:
    python pipeline/5_fetch_neuronpedia_cache.py \
      --bundle-file outputs/viz_bundle/<dataset>/bundle.json \
      --neuronpedia-id <neuronpedia_id> \
      --output-cache data/neuronpedia_cache.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from interpret_personas.utils import setup_logging

API_ROOT = "https://neuronpedia.org/api/feature"
PUBLIC_ROOT = "https://neuronpedia.org"
DEFAULT_NEURONPEDIA_ID = "gemma-3-27b-it/40-gemmascope-2-res-65k"
NEURONPEDIA_ID_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
RETRIABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _normalize_text(value: object, max_chars: int = 4000) -> str | None:
    """Normalize text for cache payload."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_chars:
        return normalized[: max_chars - 1] + "..."
    return normalized


def _coerce_feature_id(value: object) -> int | None:
    """Convert arbitrary values to non-negative feature IDs."""
    try:
        feature_id = int(value)
    except (TypeError, ValueError):
        return None
    if feature_id < 0:
        return None
    return feature_id


def _validate_neuronpedia_id(neuronpedia_id: str) -> None:
    """Validate that neuronpedia_id is path-safe before URL construction."""
    if not neuronpedia_id:
        raise ValueError("neuronpedia_id cannot be empty")
    if not NEURONPEDIA_ID_PATTERN.fullmatch(neuronpedia_id):
        raise ValueError(
            "neuronpedia_id contains invalid characters. "
            "Allowed: letters, digits, '.', '_', '-', '/'."
        )
    if ".." in neuronpedia_id or "//" in neuronpedia_id:
        raise ValueError("neuronpedia_id cannot contain '..' or '//'")


def _safe_neuronpedia_id(neuronpedia_id: str) -> str:
    """URL-encode neuronpedia_id while preserving path separators."""
    return quote(neuronpedia_id, safe="/._-")


def _api_url(neuronpedia_id: str, feature_id: int) -> str:
    return f"{API_ROOT}/{_safe_neuronpedia_id(neuronpedia_id)}/{feature_id}"


def _public_url(neuronpedia_id: str, feature_id: int) -> str:
    return f"{PUBLIC_ROOT}/{_safe_neuronpedia_id(neuronpedia_id)}/{feature_id}"


def _extract_description(payload: object) -> str | None:
    """Extract first usable description from API payload."""
    if not isinstance(payload, dict):
        return None

    explanations = payload.get("explanations")
    if isinstance(explanations, list):
        for item in explanations:
            if not isinstance(item, dict):
                continue
            description = _normalize_text(item.get("description"))
            if description:
                return description

    return _normalize_text(payload.get("description"))


def _http_get_json(url: str, timeout_seconds: float) -> Any:
    """GET JSON from URL with a conservative User-Agent."""
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "interpret-personas-neuronpedia-cache/0.1",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_seconds) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _fetch_one_feature(
    feature_id: int,
    neuronpedia_id: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[int, bool, str | None, str | None]:
    """Fetch description for one feature.

    Returns:
        (feature_id, success, description_or_none, error_or_none)
    """
    url = _api_url(neuronpedia_id, feature_id)

    for attempt in range(retries + 1):
        try:
            payload = _http_get_json(url, timeout_seconds)
            description = _extract_description(payload)
            return feature_id, True, description, None
        except HTTPError as err:
            if err.code == 404:
                # Cache negative result to avoid repeated misses.
                return feature_id, True, None, None
            retriable = err.code in RETRIABLE_HTTP_CODES
            error_message = f"HTTPError {err.code}"
        except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as err:
            retriable = True
            error_message = f"{type(err).__name__}: {err}"
        except Exception as err:  # Defensive fallback to keep batch running.
            retriable = False
            error_message = f"{type(err).__name__}: {err}"

        if attempt < retries and retriable:
            time.sleep(retry_backoff_seconds * (2 ** attempt))
            continue
        return feature_id, False, None, error_message

    return feature_id, False, None, "Unknown fetch failure"


def _load_feature_ids(bundle_file: Path) -> list[int]:
    """Read selected feature IDs from bundle JSON."""
    with open(bundle_file, "r") as f:
        payload = json.load(f)

    feature_ids = payload.get("feature_ids")
    if not isinstance(feature_ids, list):
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError(
                "Bundle JSON must contain either 'feature_ids' (list) or 'features' (list)."
            )
        feature_ids = [item.get("feature_id") for item in features if isinstance(item, dict)]

    deduped_ids: list[int] = []
    seen: set[int] = set()
    for raw_feature_id in feature_ids:
        feature_id = _coerce_feature_id(raw_feature_id)
        if feature_id is None or feature_id in seen:
            continue
        seen.add(feature_id)
        deduped_ids.append(feature_id)
    if not deduped_ids:
        raise ValueError(f"No valid feature IDs found in bundle: {bundle_file}")
    return deduped_ids


def _load_existing_cache(cache_file: Path, logger) -> dict[int, dict[str, str | None]]:
    """Load existing cache entries with defensive parsing."""
    if not cache_file.exists():
        return {}

    with open(cache_file, "r") as f:
        payload = json.load(f)

    cache: dict[int, dict[str, str | None]] = {}
    invalid_rows = 0

    if isinstance(payload, dict):
        for key, value in payload.items():
            feature_id = _coerce_feature_id(key)
            if feature_id is None:
                invalid_rows += 1
                continue

            if isinstance(value, dict):
                cache[feature_id] = {
                    "description": _normalize_text(value.get("description")),
                    "url": _normalize_text(value.get("url")),
                }
            elif isinstance(value, str):
                cache[feature_id] = {"description": _normalize_text(value), "url": None}
            else:
                invalid_rows += 1
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                invalid_rows += 1
                continue
            feature_id = _coerce_feature_id(item.get("feature_id"))
            if feature_id is None:
                invalid_rows += 1
                continue
            cache[feature_id] = {
                "description": _normalize_text(item.get("description")),
                "url": _normalize_text(item.get("url")),
            }
    else:
        raise ValueError(
            f"Cache file must be dict or list JSON, got {type(payload).__name__}"
        )

    if invalid_rows:
        logger.warning("Ignored %d invalid cache rows from %s", invalid_rows, cache_file)
    return cache


def _write_cache_atomic(cache_file: Path, cache: dict[int, dict[str, str | None]]) -> None:
    """Atomically write cache to disk using JSON dict keyed by feature ID."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        str(feature_id): {
            "description": values.get("description"),
            "url": values.get("url"),
        }
        for feature_id, values in sorted(cache.items())
    }

    temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with open(temp_file, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    temp_file.replace(cache_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Neuronpedia descriptions for bundle feature IDs"
    )
    parser.add_argument(
        "--bundle-file",
        type=Path,
        required=True,
        help="Path to visualization bundle JSON containing selected feature IDs",
    )
    parser.add_argument(
        "--neuronpedia-id",
        type=str,
        default=DEFAULT_NEURONPEDIA_ID,
        help=(
            "Neuronpedia feature set identifier used in API path "
            f"(default: {DEFAULT_NEURONPEDIA_ID})"
        ),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("data/neuronpedia_cache.json"),
        help="Where to write JSON cache (default: data/neuronpedia_cache.json)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Concurrent worker threads (default: 6)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for transient failures (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.75,
        help="Base exponential backoff in seconds (default: 0.75)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Persist cache every N processed features (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of IDs to fetch (debug/smoke runs)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch IDs that already exist in cache",
    )
    args = parser.parse_args()

    logger = setup_logging("neuronpedia_cache")
    logger.info("Bundle file: %s", args.bundle_file)
    logger.info("Output cache: %s", args.output_cache)
    logger.info("Neuronpedia ID: %s", args.neuronpedia_id)

    if not args.bundle_file.exists() or not args.bundle_file.is_file():
        raise ValueError(f"bundle_file does not exist or is not a file: {args.bundle_file}")
    if args.max_workers <= 0:
        raise ValueError(f"max_workers must be > 0, got {args.max_workers}")
    if args.timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {args.timeout_seconds}")
    if args.retries < 0:
        raise ValueError(f"retries must be >= 0, got {args.retries}")
    if args.retry_backoff_seconds < 0:
        raise ValueError(
            f"retry_backoff_seconds must be >= 0, got {args.retry_backoff_seconds}"
        )
    if args.save_every <= 0:
        raise ValueError(f"save_every must be > 0, got {args.save_every}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be > 0 when provided, got {args.limit}")

    _validate_neuronpedia_id(args.neuronpedia_id)

    feature_ids = _load_feature_ids(args.bundle_file)
    logger.info("Feature IDs in bundle: %d", len(feature_ids))

    existing_cache = _load_existing_cache(args.output_cache, logger)
    logger.info("Existing cache entries: %d", len(existing_cache))

    if args.refresh:
        pending_ids = feature_ids
    else:
        pending_ids = [feature_id for feature_id in feature_ids if feature_id not in existing_cache]

    if args.limit is not None:
        pending_ids = pending_ids[: args.limit]

    logger.info("Feature IDs to fetch: %d", len(pending_ids))
    if not pending_ids:
        logger.info("Nothing to fetch. Cache is already up to date.")
        _write_cache_atomic(args.output_cache, existing_cache)
        return

    completed = 0
    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _fetch_one_feature,
                feature_id,
                args.neuronpedia_id,
                args.timeout_seconds,
                args.retries,
                args.retry_backoff_seconds,
            ): feature_id
            for feature_id in pending_ids
        }

        for future in as_completed(futures):
            completed += 1
            try:
                feature_id, ok, description, error_message = future.result()
            except Exception as err:  # Should be rare, but keep the batch alive.
                failures += 1
                logger.warning("Unexpected worker failure: %s", err)
                continue

            if ok:
                existing_cache[feature_id] = {
                    "description": description,
                    "url": _public_url(args.neuronpedia_id, feature_id),
                }
                successes += 1
            else:
                failures += 1
                logger.warning(
                    "Failed to fetch feature %d after retries: %s",
                    feature_id,
                    error_message,
                )

            if completed % args.save_every == 0:
                _write_cache_atomic(args.output_cache, existing_cache)
                logger.info(
                    "Progress: %d/%d processed (success=%d, failure=%d)",
                    completed,
                    len(pending_ids),
                    successes,
                    failures,
                )

    _write_cache_atomic(args.output_cache, existing_cache)
    logger.info(
        "Done. Processed=%d, success=%d, failure=%d, total cache entries=%d",
        completed,
        successes,
        failures,
        len(existing_cache),
    )
    if failures > 0:
        logger.info("You can rerun the same command to retry failed IDs.")


if __name__ == "__main__":
    main()
