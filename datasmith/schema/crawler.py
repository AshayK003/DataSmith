"""Schema Crawler — extracts column schemas from real datasets via Kaggle + Frictionless.

Ponytail: two external deps (kagglehub, frictionless). Everything else is stdlib.
kagglehub handles auth, rate limits, caching. Frictionless handles type inference
and schema extraction in one call.
"""

import json
import logging
import os
import time
from typing import Optional

import kagglehub

from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.schema.models import ColumnSchema, DatasetSchema

logger = logging.getLogger(__name__)

# ── 10 seed domains for initial crawl ──

SEED_DOMAINS = {
    "e-commerce": "Retail transactions, product catalogs, customer data",
    "healthcare": "Medical records, patient data, clinical trials",
    "finance": "Banking, trading, lending, insurance data",
    "education": "Student performance, enrollment, institutional data",
    "social-media": "User engagement, content, network data",
    "iot-sensors": "Sensor readings, telemetry, device data",
    "real-estate": "Property listings, prices, location data",
    "transportation": "Traffic, logistics, ride-sharing, transit data",
    "energy": "Consumption, production, grid, renewable data",
    "manufacturing": "Production lines, quality control, supply chain",
}

# Mapping: domain → Kaggle dataset slugs to seed the KG
SEED_DATASETS: dict[str, list[str]] = {
    "e-commerce": [
        "olistbr/brazilian-ecommerce",
        "carrie1/ecommerce-data",
        "shilongzhuang/amazon-sales-dataset-2023",
    ],
    "healthcare": [
        "miriansaei/healthcare-datasets",
        "saurabhshahane/patient-treatment-classification",
        "mathchi/diabetes-data-set",
    ],
    "finance": [
        "borismk/credit-card-transactions-dataset",
        "camnugent/credit-card-fraud-detection",
        "nsharan/housing-prices-dataset",
    ],
    "education": [
        "spsci/academic-data",
        "jcprogjava/student-performance-data",
        "yogesh70/students-scores-in-exams",
    ],
    "social-media": [
        "shivkumarganesh/social-media-usage-data",
        "ma7555/instagram-user-data",
        "benjaminawd/youtube-trending-stats",
    ],
    "iot-sensors": [
        "robervalt/sensor-readings-dataset",
        "hugomathien/sensors-energy-prediction",
        "uciml/electric-power-consumption-data-set",
    ],
    "real-estate": [
        "ahmedshahriarsakib/usa-real-estate-dataset",
        "yjb00/us-housing-prices",
        "quantbruce/real-estate-price-prediction",
    ],
    "transportation": [
        "dansbecker/nyc-taxi-trip-duration",
        "new-york-city/nyc-taxi-trip-duration",
        "arashnic/rideshare-trip-data",
    ],
    "energy": [
        "timmayer/energy-consumption",
        "robikscube/hourly-energy-consumption",
        "bobn31/power-generation-data",
    ],
    "manufacturing": [
        "rabieelkharoua/manufacturing-dataset",
        "sagarnildass/predictive-maintenance-dataset",
        "shyambhu/predictive-maintenance-data",
    ],
}


def _extract_type(ftype: str) -> str:
    """Map Frictionless type → our simplified type taxonomy."""
    mapping = {
        "string": "text",
        "number": "numeric",
        "integer": "numeric",
        "date": "datetime",
        "datetime": "datetime",
        "time": "datetime",
        "year": "numeric",
        "yearmonth": "datetime",
        "boolean": "boolean",
        "binary": "text",
        "object": "text",
        "array": "text",
        "geopoint": "text",
        "geojson": "text",
        "any": "text",
    }
    return mapping.get(ftype, ftype)


def extract_schema(file_path: str, dataset_id: int) -> list[ColumnSchema]:
    """Extract column schemas from a CSV using Frictionless."""
    from frictionless import describe

    resource = describe(file_path)
    columns = []
    row_count = 0

    for field in (resource.schema.fields if resource.schema else []):
        col = ColumnSchema(
            dataset_id=dataset_id,
            column_name=field.name,
            column_description=field.description or "",
            data_type=_extract_type(field.type),
            null_ratio=None,
        )
        # Extract constraints
        if field.constraints:
            if field.constraints.get("required"):
                col.null_ratio = 0.0
            else:
                col.null_ratio = None  # unknown until we scan
        columns.append(col)

    return columns


def scan_data_quality(file_path: str,
                      columns: list[ColumnSchema]) -> list[ColumnSchema]:
    """Scan actual data to fill null_ratio, distinct_count, stats for numeric cols."""
    import pandas as pd
    import numpy as np

    try:
        df = pd.read_csv(file_path, nrows=10000)
        row_count = len(df)
    except Exception:
        return columns

    name_map = {c.column_name: c for c in columns}
    for col_name in df.columns:
        col_def = name_map.get(col_name)
        if col_def is None:
            continue
        series = df[col_name]
        col_def.null_ratio = float(series.isna().mean())
        col_def.distinct_count = int(series.nunique())
        if series.dtype in (np.float64, np.int64, float, int):
            col_def.mean = float(series.mean()) if not series.isna().all() else None
            col_def.std = float(series.std()) if not series.isna().all() else None
            col_def.minimum = float(series.min()) if not series.isna().all() else None
            col_def.maximum = float(series.max()) if not series.isna().all() else None
        sample = series.dropna().unique()[:5].tolist()
        col_def.sample_values = json.dumps([str(s) for s in sample])

    return columns


def crawl_dataset_schema(kg: KnowledgeGraph, domain_id: int,
                         dataset_slug: str, dataset_name: str) -> Optional[int]:
    """Download a Kaggle dataset, extract its schema, store in KG.

    Returns the dataset_schemas.id or None on failure.
    """
    logger.info("Crawling %s (%s)", dataset_slug, dataset_name)
    try:
        path = kagglehub.dataset_download(dataset_slug)
    except Exception as e:
        logger.warning("Failed to download %s: %s", dataset_slug, e)
        return None

    if isinstance(path, str):
        path = path

    dataset = DatasetSchema(
        source="kaggle",
        source_url=f"https://kaggle.com/datasets/{dataset_slug}",
        domain_id=domain_id,
        dataset_name=dataset_name,
    )

    dataset_id = kg.upsert_dataset(dataset)

    # Find CSV files in the downloaded path
    csv_files = []
    if os.path.isfile(path) and path.endswith(".csv"):
        csv_files = [path]
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

    if not csv_files:
        logger.warning("No CSV files found in %s", dataset_slug)
        return dataset_id

    # Use the first CSV for schema extraction
    csv_path = csv_files[0]
    columns = extract_schema(csv_path, dataset_id)
    columns = scan_data_quality(csv_path, columns)
    kg.insert_columns(columns)

    # Update dataset metadata
    try:
        df = pd.read_csv(csv_path, nrows=10000)
        row_count = len(df)
        col_count = len(df.columns)
        kg.db.execute(
            "UPDATE dataset_schemas SET row_count=?, column_count=? WHERE id=?",
            (row_count, col_count, dataset_id),
        )
        kg.db.commit()
    except Exception:
        kg.db.execute(
            "UPDATE dataset_schemas SET column_count=? WHERE id=?",
            (len(columns), dataset_id),
        )
        kg.db.commit()

    logger.info("Stored %d columns from %s", len(columns), dataset_slug)
    return dataset_id


def seed_knowledge_graph(kg: KnowledgeGraph,
                         datasets: Optional[dict[str, list[str]]] = None,
                         delay: float = 2.0) -> dict:
    """Seed the KG with initial datasets across domains.

    Args:
        kg: KnowledgeGraph instance.
        datasets: Domain→[dataset_slug] mapping. Defaults to SEED_DATASETS.
        delay: Seconds between crawls to respect Kaggle rate limits.

    Returns: {domain: {dataset: success_or_error}}
    """
    import pandas as pd  # noqa: F401 — used by scan_data_quality

    results: dict = {}
    datasets = datasets or SEED_DATASETS

    for domain_name, slugs in datasets.items():
        domain_id = kg.upsert_domain(domain_name, SEED_DOMAINS.get(domain_name, ""))
        domain_results = {}
        for slug in slugs:
            name = slug.split("/")[-1].replace("-", " ").title()
            dataset_id = crawl_dataset_schema(kg, domain_id, slug, name)
            if dataset_id:
                domain_results[slug] = "ok"
            else:
                domain_results[slug] = "failed"
            time.sleep(delay)
        results[domain_name] = domain_results

    return results
