"""Schema Crawler — multi-source schema extraction from real datasets.

Sources (in priority order):
1. Kaggle (via kagglehub — no auth needed for public datasets)
2. HuggingFace Datasets (via requests — free, no auth for public)
3. Direct CSV URLs (UCI, open-data portals)

Every source gets wrapped in try/except — failures are logged, never fatal.
"""

import json
import logging
import os
import time
from typing import Optional

import kagglehub
import requests

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

# Datasets keyed by domain. Each entry: (source, identifier, label)
# source: "kaggle" | "url"
#   Kaggle needs auth (set KAGGLE_USERNAME + KAGGLE_KEY env vars)
#   URL sources are direct CSV/Excel/ZIP links (no auth needed)
# HuggingFace sources temporarily disabled — most free-tier datasets
# use Parquet/Arrow, not CSV. Will re-enable when CSV pipeline is ready.
SEED_DATASETS: dict[str, list[tuple[str, str, str]]] = {
    "e-commerce": [
        ("kaggle", "olistbr/brazilian-ecommerce", "Brazilian E-Commerce"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/alcohol-consumption/drinks.csv", "Alcohol Consumption"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv", "Candy Rankings"),
    ],
    "healthcare": [
        ("url", "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv", "Heart Failure Clinical"),
        ("url", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "Wine Quality Red"),
        ("url", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "Wine Quality White"),
        ("kaggle", "mathchi/diabetes-data-set", "Diabetes Dataset"),
    ],
    "finance": [
        ("kaggle", "borismk/credit-card-transactions-dataset", "Credit Card Transactions"),
        ("url", "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv", "CO2 Emissions (OWID)"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv", "Airline Safety"),
    ],
    "education": [
        ("kaggle", "spsci/academic-data", "Academic Data"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/grad-students.csv", "College Graduate Students"),
        ("url", "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/latest/owid-covid-latest.csv", "COVID-19 Latest (OWID)"),
    ],
    "social-media": [
        ("kaggle", "benjaminawd/youtube-trending-stats", "YouTube Trending"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/masculinity-survey/masculinity-survey.csv", "Masculinity Survey"),
    ],
    "iot-sensors": [
        ("kaggle", "uciml/electric-power-consumption-data-set", "Power Consumption"),
        ("url", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", "Airline Passengers"),
        ("url", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "Daily Min Temperatures"),
    ],
    "real-estate": [
        ("kaggle", "ahmedshahriarsakib/usa-real-estate-dataset", "USA Real Estate"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/hate-crimes/hate_crimes.csv", "Hate Crimes by Metro"),
        ("url", "https://raw.githubusercontent.com/fivethirtyeight/data/master/bad-drivers/bad-drivers.csv", "Bad Drivers"),
    ],
    "transportation": [
        ("kaggle", "dansbecker/nyc-taxi-trip-duration", "NYC Taxi Trips"),
        ("url", "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv", "Vaccinations Data"),
    ],
    "energy": [
        ("kaggle", "robikscube/hourly-energy-consumption", "Hourly Energy Consumption"),
        ("url", "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv", "OWID Energy Data"),
    ],
    "manufacturing": [
        ("kaggle", "rabieelkharoua/manufacturing-dataset", "Manufacturing Quality"),
        ("url", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv", "Shampoo Sales"),
    ],
}


def _extract_type(ftype: str) -> str:
    """Map Frictionless type → our simplified type taxonomy."""
    mapping = {
        "string": "text", "number": "numeric", "integer": "numeric",
        "date": "datetime", "datetime": "datetime", "time": "datetime",
        "year": "numeric", "yearmonth": "datetime",
        "boolean": "boolean", "binary": "text", "object": "text",
        "array": "text", "geopoint": "text", "geojson": "text", "any": "text",
    }
    return mapping.get(ftype, ftype)


def _download_file(url: str, dest: str, timeout: int = 120) -> Optional[str]:
    """Download a file from a URL to a local path. Returns the path or None."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest
    except Exception as e:
        logger.warning("Download failed %s: %s", url, e)
        return None


def _find_csv_files(path: str) -> list[str]:
    """Find all CSV files in a path (file or directory)."""
    if os.path.isfile(path):
        return [path] if path.endswith(".csv") else []
    csvs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs


def extract_schema(file_path: str, dataset_id: int) -> list[ColumnSchema]:
    """Extract column schemas from a CSV using Frictionless."""
    from frictionless import describe

    try:
        resource = describe(file_path)
    except Exception as e:
        logger.warning("Frictionless describe failed for %s: %s", file_path, e)
        return []

    columns = []
    for field in (resource.schema.fields if resource.schema else []):
        col = ColumnSchema(
            dataset_id=dataset_id,
            column_name=field.name,
            column_description=field.description or "",
            data_type=_extract_type(field.type),
            null_ratio=0.0 if field.constraints and field.constraints.get("required") else None,
        )
        columns.append(col)
    return columns


def _read_csv_safe(file_path: str, nrows: Optional[int] = 10000):
    """Read a CSV with automatic delimiter detection."""
    import pandas as pd
    import csv
    
    # Try comma first (fast path)
    try:
        df = pd.read_csv(file_path, nrows=nrows, encoding="utf-8")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try semicolon (common in UCI datasets)
    try:
        df = pd.read_csv(file_path, nrows=nrows, sep=";", encoding="utf-8")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try Python CSV Sniffer
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample)
            f.seek(0)
            df = pd.read_csv(f, nrows=nrows, sep=dialect.delimiter)
            return df
    except Exception:
        pass
    
    # Final fallback
    return pd.read_csv(file_path, nrows=nrows, encoding="utf-8")


def scan_data_quality(file_path: str,
                      columns: list[ColumnSchema]) -> list[ColumnSchema]:
    """Scan actual data to fill null_ratio, distinct_count, stats for numeric cols."""
    import pandas as pd
    import numpy as np

    try:
        df = _read_csv_safe(file_path)
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


def _store_schema(kg: KnowledgeGraph, dataset: DatasetSchema,
                  csv_path: str) -> Optional[int]:
    """Extract schema from CSV and store in KG. Returns dataset_id or None."""
    dataset_id = kg.upsert_dataset(dataset)
    columns = extract_schema(csv_path, dataset_id)
    if not columns:
        logger.warning("No columns extracted from %s", csv_path)
        return dataset_id
    columns = scan_data_quality(csv_path, columns)
    kg.insert_columns(columns)

    # Update metadata
    try:
        import pandas as pd
        df = _read_csv_safe(csv_path)
        kg.db.execute(
            "UPDATE dataset_schemas SET row_count=?, column_count=? WHERE id=?",
            (len(df), len(df.columns), dataset_id),
        )
    except Exception:
        kg.db.execute(
            "UPDATE dataset_schemas SET column_count=? WHERE id=?",
            (len(columns), dataset_id),
        )
    kg.db.commit()
    logger.info("Stored %d columns from dataset %d", len(columns), dataset_id)
    return dataset_id


def _crawl_kaggle(kg: KnowledgeGraph, dataset_slug: str,
                  dataset_name: str, domain_id: int) -> Optional[int]:
    """Download and process a Kaggle dataset. Returns dataset_id or None."""
    logger.info("Kaggle: %s (%s)", dataset_slug, dataset_name)
    try:
        path = kagglehub.dataset_download(dataset_slug)
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "permission" in msg or "authenticated" in msg:
            logger.warning("Kaggle %s needs auth — skipping", dataset_slug)
        else:
            logger.warning("Kaggle %s failed: %s", dataset_slug, e)
        return None

    csv_files = _find_csv_files(path)
    if not csv_files:
        logger.warning("No CSVs in Kaggle %s", dataset_slug)
        return None

    dataset = DatasetSchema(
        source="kaggle",
        source_url=f"https://kaggle.com/datasets/{dataset_slug}",
        dataset_name=dataset_name,
        domain_id=domain_id,
    )
    return _store_schema(kg, dataset, csv_files[0])


def _crawl_huggingface(kg: KnowledgeGraph, dataset_name: str,
                       display_name: str, domain_id: int) -> Optional[int]:
    """Download and process a HuggingFace dataset. Returns dataset_id or None."""
    logger.info("HuggingFace: %s (%s)", dataset_name, display_name)
    try:
        # List files via HF API
        api_url = f"https://huggingface.co/api/datasets/{dataset_name}"
        r = requests.get(api_url, timeout=15)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        logger.warning("HF API failed for %s: %s", dataset_name, e)
        return None

    # Find CSV files from the dataset info
    csv_urls = []
    siblings = info.get("siblings", [])
    for sib in siblings:
        rfilename = sib.get("rfilename", "")
        if rfilename.endswith(".csv"):
            csv_urls.append(
                f"https://huggingface.co/datasets/{dataset_name}/raw/main/{rfilename}"
            )

    if not csv_urls:
        logger.warning("No CSVs found in HF %s", dataset_name)
        return None

    # Download first CSV
    import tempfile
    dest = os.path.join(tempfile.gettempdir(),
                        f"datasmith_hf_{dataset_name.replace('/', '_')}_{os.path.basename(csv_urls[0])}")
    local_path = _download_file(csv_urls[0], dest)
    if not local_path:
        return None

    dataset = DatasetSchema(
        source="huggingface",
        source_url=f"https://huggingface.co/datasets/{dataset_name}",
        dataset_name=display_name,
        domain_id=domain_id,
    )
    return _store_schema(kg, dataset, local_path)


def _crawl_url(kg: KnowledgeGraph, url: str,
               display_name: str, domain_id: int) -> Optional[int]:
    """Download a CSV from a direct URL and process it. Returns dataset_id or None."""
    import tempfile
    logger.info("URL: %s (%s)", url, display_name)
    ext = os.path.splitext(url.split("?")[0])[1] or ".csv"
    dest = os.path.join(tempfile.gettempdir(),
                        f"datasmith_url_{display_name.lower().replace(' ', '_')}{ext}")
    local_path = _download_file(url, dest)
    if not local_path:
        return None

    # If it's a zip, extract and find CSVs
    if local_path.endswith(".zip"):
        import zipfile
        try:
            with zipfile.ZipFile(local_path, "r") as zf:
                extract_dir = tempfile.mkdtemp()
                zf.extractall(extract_dir)
                csvs = _find_csv_files(extract_dir)
                if not csvs:
                    logger.warning("No CSVs in zip %s", url)
                    return None
                local_path = csvs[0]
        except Exception as e:
            logger.warning("Failed to extract %s: %s", url, e)
            return None

    # If it's Excel, convert first CSV sheet
    if local_path.endswith((".xls", ".xlsx")):
        try:
            import pandas as pd
            df = pd.read_excel(local_path)
            csv_path = local_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)
            local_path = csv_path
        except Exception as e:
            logger.warning("Failed to convert Excel %s: %s", url, e)
            return None

    if not local_path.endswith(".csv"):
        logger.warning("Unsupported format for %s", url)
        return None

    dataset = DatasetSchema(
        source="url",
        source_url=url,
        dataset_name=display_name,
        domain_id=domain_id,
    )
    return _store_schema(kg, dataset, local_path)


# Source dispatch table
_SOURCE_DISPATCH = {
    "kaggle": _crawl_kaggle,
    "huggingface": _crawl_huggingface,
    "url": _crawl_url,
}


def seed_knowledge_graph(kg: KnowledgeGraph,
                         datasets: Optional[dict[str, list[tuple[str, str, str]]]] = None,
                         delay: float = 1.0) -> dict:
    """Seed the KG with initial datasets across multiple sources.

    Args:
        kg: KnowledgeGraph instance.
        datasets: Domain → [(source, identifier, label), ...] mapping.
                  source: "kaggle" | "huggingface" | "url"
        delay: Seconds between crawls to respect rate limits.

    Returns: {domain: {display_name: "ok"|"skipped"|"failed"}}
    """
    import pandas as pd  # noqa: F401
    import numpy as np  # noqa: F401

    results: dict = {}
    datasets = datasets or SEED_DATASETS

    for domain_name, entries in datasets.items():
        domain_id = kg.upsert_domain(domain_name, SEED_DOMAINS.get(domain_name, ""))
        domain_results = {}
        for source, identifier, label in entries:
            crawler = _SOURCE_DISPATCH.get(source)
            if not crawler:
                domain_results[f"{source}:{identifier}"] = "skipped"
                continue
            ds_id = crawler(kg, identifier, label, domain_id)
            domain_results[label] = "ok" if ds_id else "failed"
            time.sleep(delay)
        results[domain_name] = domain_results

    return results
