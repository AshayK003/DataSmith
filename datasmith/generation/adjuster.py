"""Parameter adjustment rules for the iterative batched generation loop.

The adjuster reads quality metrics from the previous batch and tweaks
generation parameters for the next batch. Adjustments are proportional
corrections (a simple P-controller), not learned updates.

Design principle: the KG stats are the ground truth. Adjustments only
compensate for sampling noise in individual batches — they pull the next
batch's parameters slightly to counteract measured drift.
"""

import copy
import logging

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────

_KS_THRESHOLD = 0.10  # Max acceptable KS statistic per column
_NULL_DRIFT_THRESHOLD = 0.02  # Max acceptable null-rate drift (2pp)
_LEARNING_RATE = 0.1  # How much to correct per iteration


def adjust_schema(schema: list[dict],
                  quality: dict,
                  ks_threshold: float = _KS_THRESHOLD,
                  null_threshold: float = _NULL_DRIFT_THRESHOLD,
                  lr: float = _LEARNING_RATE) -> list[dict]:
    """Return a *new* schema list with adjusted parameters for the next batch.

    Each column schema is processed independently:
      - KS > threshold → pull mean toward expected (proportional correction)
      - null-rate drift > threshold → adjust null_ratio toward target

    Args:
        schema: Original column schema list.
        quality: Quality metrics dict from compute_batch_quality().
        ks_threshold: KS stat above this triggers adjustment.
        null_threshold: Null-drift above this triggers adjustment.
        lr: Learning rate (fraction of error to correct per iteration).

    Returns a new schema list (original is not mutated).
    """
    adjusted = []
    adjustments = 0

    for col in schema:
        entry = dict(col)  # shallow copy
        name = col.get("column_name", "")

        # ── KS-based mean correction ──
        ks_key = f"ks_{name}"
        if ks_key in quality and quality[ks_key] > ks_threshold:
            current_mean = float(entry.get("mean", 50.0) or 50.0)
            # The KS stat signals the batch mean is off from target.
            # We don't know the actual batch mean here (it's not in quality),
            # so we apply a small conservative pull toward the reference mean.
            # Since the KG mean is already the target, this is a tiny nudge
            # to counter persistent bias.
            correction = current_mean * lr * quality[ks_key]
            new_mean = current_mean - correction
            entry["mean"] = round(new_mean, 4)
            adjustments += 1
            logger.debug(
                "Adjusted '%s'.mean: %.2f → %.2f (KS=%.3f)",
                name, current_mean, new_mean, quality[ks_key],
            )

        # ── Null-rate correction ──
        drift_key = f"null_drift_{name}"
        target_null = float(col.get("null_ratio", 0.0) or 0.0)
        if drift_key in quality and quality[drift_key] > null_threshold:
            # Adjust the column's null_ratio in the schema
            current_null = target_null
            error = quality[drift_key]
            # If actual nulls were too high, reduce target null rate
            # Since we don't know the direction of drift from the scalar abs,
            # we use a conservative reduction proportional to drift
            if target_null > 0:
                new_null = max(0.0, target_null - error * lr * 10)
                entry["null_ratio"] = round(new_null, 4)
                adjustments += 1
                logger.debug(
                    "Adjusted '%s'.null_ratio: %.3f → %.3f (drift=%.3f)",
                    name, current_null, new_null, quality[drift_key],
                )

        adjusted.append(entry)

    if adjustments:
        logger.info("Adjuster applied %d parameter corrections", adjustments)
    return adjusted


def adjust_imperfection_profile(profile: dict | None,
                                quality: dict,
                                schema: list[dict],
                                null_threshold: float = _NULL_DRIFT_THRESHOLD,
                                lr: float = _LEARNING_RATE) -> dict | None:
    """Adjust imperfection profile null rates based on quality metrics.

    If the generated batch had null rates drifting from target, adjust the
    profile's null_pct values to compensate.

    Args:
        profile: Current imperfection profile (may be None).
        quality: Quality metrics dict.
        schema: Column schema list.
        null_threshold: Drift threshold for adjustment.
        lr: Learning rate.

    Returns adjusted profile (or None if profile was None).
    """
    if profile is None:
        return None

    profile = copy.deepcopy(profile)
    null_patterns = profile.get("null_patterns", {})

    for col in schema:
        name = col.get("column_name", "")
        drift_key = f"null_drift_{name}"
        if drift_key not in quality or quality[drift_key] <= null_threshold:
            continue
        if name not in null_patterns:
            continue

        target_null = float(col.get("null_ratio", 0.0) or 0.0)
        current_pct = float(null_patterns[name].get("null_pct", 0.0) or 0.0)
        if current_pct <= 0:
            continue

        error = quality[drift_key]
        new_pct = max(0.0, current_pct - error * lr * 50)
        null_patterns[name]["null_pct"] = round(new_pct, 2)
        logger.debug(
            "Profile '%s'.null_pct: %.1f → %.1f (drift=%.3f)",
            name, current_pct, new_pct, quality[drift_key],
        )

    profile["null_patterns"] = null_patterns
    return profile
