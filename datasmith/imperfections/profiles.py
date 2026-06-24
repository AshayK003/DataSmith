"""Domain Imperfection Profiles — default + analyzed fingerprints for each domain.

Ponytail: no classes, just dicts. Default profiles serve as priors until
real analysis data is available. Real profiles are JSON stored in KG's
domain_profiles table.
"""

import json
import logging


logger = logging.getLogger(__name__)

# ── Default Profiles (priors until real analysis fills in) ────────────────
# Each is a realistic best-guess based on common data quality patterns.

DEFAULT_PROFILES: dict[str, dict] = {
    "e-commerce": {
        "null_patterns": {
            "shipping_address": {"null_pct": 15.0, "pattern": "MAR"},
            "gift_wrap": {"null_pct": 60.0, "pattern": "MCAR"},
            "discount_code": {"null_pct": 45.0, "pattern": "MAR"},
            "review_rating": {"null_pct": 70.0, "pattern": "MAR"},
        },
        "null_correlations": [
            {"cols": ["shipping_address", "gift_wrap"], "jaccard": 0.6},
        ],
        "outlier_patterns": {
            "price": {"direction": "high", "outlier_pct": 2.0},
            "quantity": {"direction": "high", "outlier_pct": 1.0},
        },
        "skew_profiles": {
            "price": {"distribution_hint": "powerlaw", "skewness": 3.5},
            "quantity": {"distribution_hint": "powerlaw", "skewness": 5.0},
            "discount_rate": {"distribution_hint": "uniform", "skewness": 0.2},
        },
        "noise_patterns": {
            "price": {"rounding_pct": 85.0, "precision": 0.01},
            "quantity": {"precision": 1.0},
        },
    },
    "healthcare": {
        "null_patterns": {
            "diagnosis_code": {"null_pct": 12.0, "pattern": "MNAR"},
            "secondary_diagnosis": {"null_pct": 25.0, "pattern": "MAR"},
            "lab_result_value": {"null_pct": 8.0, "pattern": "MNAR"},
            "patient_weight": {"null_pct": 20.0, "pattern": "MCAR"},
        },
        "null_correlations": [
            {"cols": ["diagnosis_code", "secondary_diagnosis"], "jaccard": 0.7},
        ],
        "outlier_patterns": {
            "lab_result_value": {"direction": "both", "outlier_pct": 3.0},
            "age": {"direction": "high", "outlier_pct": 0.5},
        },
        "skew_profiles": {
            "lab_result_value": {"distribution_hint": "lognormal", "skewness": 2.0},
            "age": {"distribution_hint": "normal", "skewness": -0.1},
        },
        "noise_patterns": {
            "lab_result_value": {"rounding_pct": 30.0, "precision": 0.01},
            "heart_rate": {"precision": 1.0},
        },
    },
    "finance": {
        "null_patterns": {
            "credit_score": {"null_pct": 10.0, "pattern": "MAR"},
            "annual_income": {"null_pct": 15.0, "pattern": "MNAR"},
            "loan_purpose": {"null_pct": 5.0, "pattern": "MCAR"},
        },
        "outlier_patterns": {
            "transaction_amount": {"direction": "high", "outlier_pct": 1.5},
            "annual_income": {"direction": "high", "outlier_pct": 1.0},
        },
        "skew_profiles": {
            "transaction_amount": {"distribution_hint": "powerlaw", "skewness": 4.0},
            "annual_income": {"distribution_hint": "powerlaw", "skewness": 2.5},
        },
        "noise_patterns": {
            "transaction_amount": {"rounding_pct": 75.0, "precision": 0.01},
        },
    },
    "education": {
        "null_patterns": {
            "parental_education": {"null_pct": 18.0, "pattern": "MAR"},
            "test_prep_course": {"null_pct": 30.0, "pattern": "MCAR"},
            "college_plan": {"null_pct": 22.0, "pattern": "MAR"},
        },
        "outlier_patterns": {
            "exam_score": {"direction": "low", "outlier_pct": 2.0},
            "hours_studied": {"direction": "high", "outlier_pct": 1.5},
        },
        "skew_profiles": {
            "exam_score": {"distribution_hint": "normal", "skewness": -0.3},
            "hours_studied": {"distribution_hint": "lognormal", "skewness": 1.5},
        },
        "noise_patterns": {
            "exam_score": {"precision": 1.0},
        },
    },
    "social-media": {
        "null_patterns": {
            "user_bio": {"null_pct": 40.0, "pattern": "MCAR"},
            "location": {"null_pct": 35.0, "pattern": "MAR"},
            "profile_picture_url": {"null_pct": 15.0, "pattern": "MCAR"},
        },
        "outlier_patterns": {
            "follower_count": {"direction": "high", "outlier_pct": 1.0},
            "engagement_rate": {"direction": "both", "outlier_pct": 5.0},
        },
        "skew_profiles": {
            "follower_count": {"distribution_hint": "powerlaw", "skewness": 8.0},
            "engagement_rate": {"distribution_hint": "lognormal", "skewness": 2.0},
        },
        "noise_patterns": {
            "follower_count": {"precision": 1.0},
        },
    },
    "iot-sensors": {
        "null_patterns": {
            "sensor_reading": {"null_pct": 2.0, "pattern": "MCAR"},
            "calibration_date": {"null_pct": 10.0, "pattern": "MAR"},
        },
        "outlier_patterns": {
            "temperature": {"direction": "both", "outlier_pct": 1.0},
            "humidity": {"direction": "both", "outlier_pct": 2.0},
        },
        "skew_profiles": {
            "temperature": {"distribution_hint": "normal", "skewness": 0.1},
            "humidity": {"distribution_hint": "normal", "skewness": -0.2},
        },
        "noise_patterns": {
            "temperature": {"rounding_pct": 20.0, "precision": 0.1},
            "humidity": {"rounding_pct": 15.0, "precision": 0.1},
        },
    },
    "real-estate": {
        "null_patterns": {
            "year_built": {"null_pct": 5.0, "pattern": "MAR"},
            "hoa_fee": {"null_pct": 25.0, "pattern": "MAR"},
            "lot_size": {"null_pct": 12.0, "pattern": "MCAR"},
        },
        "outlier_patterns": {
            "price": {"direction": "high", "outlier_pct": 1.5},
            "square_feet": {"direction": "high", "outlier_pct": 2.0},
        },
        "skew_profiles": {
            "price": {"distribution_hint": "powerlaw", "skewness": 3.0},
            "square_feet": {"distribution_hint": "lognormal", "skewness": 1.5},
        },
        "noise_patterns": {
            "price": {"rounding_pct": 90.0, "precision": 1000.0},
        },
    },
    "transportation": {
        "null_patterns": {
            "dropoff_location": {"null_pct": 8.0, "pattern": "MAR"},
            "driver_rating": {"null_pct": 30.0, "pattern": "MCAR"},
        },
        "outlier_patterns": {
            "trip_distance": {"direction": "high", "outlier_pct": 1.0},
            "fare_amount": {"direction": "high", "outlier_pct": 2.0},
        },
        "skew_profiles": {
            "trip_distance": {"distribution_hint": "lognormal", "skewness": 4.0},
            "fare_amount": {"distribution_hint": "lognormal", "skewness": 3.5},
            "passenger_count": {"distribution_hint": "powerlaw", "skewness": 5.0},
        },
        "noise_patterns": {
            "trip_distance": {"precision": 0.1},
            "fare_amount": {"rounding_pct": 80.0, "precision": 0.01},
        },
    },
    "energy": {
        "null_patterns": {
            "wind_speed": {"null_pct": 5.0, "pattern": "MCAR"},
            "solar_irradiance": {"null_pct": 8.0, "pattern": "MAR"},
        },
        "outlier_patterns": {
            "consumption_kwh": {"direction": "high", "outlier_pct": 1.0},
            "voltage": {"direction": "both", "outlier_pct": 0.5},
        },
        "skew_profiles": {
            "consumption_kwh": {"distribution_hint": "lognormal", "skewness": 1.5},
            "temperature": {"distribution_hint": "normal", "skewness": 0.0},
        },
        "noise_patterns": {
            "consumption_kwh": {"precision": 0.01},
            "voltage": {"rounding_pct": 30.0, "precision": 0.1},
        },
    },
    "manufacturing": {
        "null_patterns": {
            "defect_code": {"null_pct": 5.0, "pattern": "MNAR"},
            "quality_check_result": {"null_pct": 3.0, "pattern": "MCAR"},
        },
        "outlier_patterns": {
            "production_defect_rate": {"direction": "high", "outlier_pct": 2.0},
            "cycle_time": {"direction": "high", "outlier_pct": 1.5},
        },
        "skew_profiles": {
            "production_defect_rate": {"distribution_hint": "lognormal", "skewness": 2.5},
            "cycle_time": {"distribution_hint": "lognormal", "skewness": 2.0},
        },
        "noise_patterns": {
            "cycle_time": {"precision": 0.1},
            "temperature": {"rounding_pct": 25.0, "precision": 0.5},
        },
    },
}


def get_default_profile(domain_name: str) -> dict:
    """Return the default imperfection profile for a domain.

    Falls back to a generic profile if the domain isn't recognized.
    """
    return DEFAULT_PROFILES.get(domain_name, {
        "null_patterns": {},
        "outlier_patterns": {},
        "skew_profiles": {},
        "noise_patterns": {},
    })


def merge_profile(existing: dict, analysis: dict) -> dict:
    """Merge a real analysis fingerprint into an existing profile.

    Real analysis data takes precedence over defaults. Numeric values
    are averaged if both exist; lists are deduped and merged.
    """
    merged = dict(existing)

    # Merge null_patterns from analysis
    if "null_patterns" in analysis and "null_correlations" in analysis:
        merged.setdefault("null_patterns", {}).update(analysis.get("null_patterns", {}))
        merged.setdefault("null_correlations", analysis.get("null_correlations", []))

        # Merge missingness column analysis
        missing = analysis.get("missingness", {}).get("columns", {})
        for col, info in missing.items():
            if col not in merged["null_patterns"]:
                merged["null_patterns"][col] = {
                    "null_pct": info.get("null_pct", 0),
                    "pattern": info.get("pattern", "MCAR"),
                }
            else:
                # Average null_pct if both exist
                current = merged["null_patterns"][col].get("null_pct", 0)
                incoming = info.get("null_pct", 0)
                if current is not None and incoming is not None:
                    merged["null_patterns"][col]["null_pct"] = round((current + incoming) / 2, 1)

    # Merge outlier patterns
    outliers = analysis.get("outliers", {})
    if outliers:
        merged.setdefault("outlier_patterns", {})
        for col, info in outliers.items():
            merged["outlier_patterns"][col] = {
                "direction": info.get("direction", "both"),
                "outlier_pct": round(info.get("outlier_pct", 0), 1),
            }

    # Merge skew profiles
    skew = analysis.get("skew_profiles", {})
    if skew:
        merged.setdefault("skew_profiles", {})
        for col, info in skew.items():
            merged["skew_profiles"][col] = info

    # Merge noise patterns
    noise = analysis.get("noise", {})
    if noise:
        merged.setdefault("noise_patterns", {})
        for col, info in noise.items():
            merged["noise_patterns"][col] = info

    return merged


def save_profile_to_kg(kg, domain_name: str, profile: dict) -> bool:
    """Save (or update) a domain imperfection profile in the KG.

    Args:
        kg: KnowledgeGraph instance.
        domain_name: Domain name (e.g. "e-commerce").
        profile: Imperfection fingerprint dict.

    Returns True on success.
    """
    domain = kg.get_domain_by_name(domain_name)
    if not domain:
        logger.warning("Domain '%s' not found — upserting", domain_name)
        domain_id = kg.upsert_domain(domain_name,
                                     f"{domain_name.capitalize()} data domain")
    else:
        domain_id = domain.id

    kg.upsert_domain_profile(domain_id, json.dumps(profile))
    logger.info("Saved profile for '%s' (domain_id=%d)", domain_name, domain_id)
    return True


def load_profile_from_kg(kg, domain_name: str) -> dict:
    """Load a domain imperfection profile, merging real data with defaults.

    If the KG has analysis data, it's merged into the default profile.
    The result is always a complete profile (never partial).
    """
    default = get_default_profile(domain_name)
    domain = kg.get_domain_by_name(domain_name)
    if not domain:
        return default

    db_profile = kg.get_domain_profile(domain.id)
    if not db_profile or not db_profile.profile_json:
        return default

    try:
        real_data = json.loads(db_profile.profile_json)
    except (json.JSONDecodeError, TypeError):
        return default

    merged = merge_profile(default, real_data)
    return merged
