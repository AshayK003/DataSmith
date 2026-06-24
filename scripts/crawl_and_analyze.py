#!/usr/bin/env python3
"""Batch crawl + analyze all domains. Suitable for daily cron / GitHub Actions.

Usage:
    python scripts/crawl_and_analyze.py                    # crawl + analyze
    python scripts/crawl_and_analyze.py --dry-run          # show what would be done
    python scripts/crawl_and_analyze.py --commit           # commit data changes (CI mode)
    python scripts/crawl_and_analyze.py --source url       # only URL sources (fastest)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crawl_and_analyze")

# Ensure project root is on path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Batch crawl + analyze domains")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--commit", action="store_true",
                        help="Commit data/ changes (for GitHub Actions)")
    parser.add_argument("--source", choices=["all", "url", "kaggle", "huggingface"],
                        default="all", help="Filter by source")
    args = parser.parse_args()

    from datasmith.core.database import Database
    from datasmith.schema.knowledge_graph import KnowledgeGraph
    from datasmith.schema.crawler import (
        seed_knowledge_graph, SEED_DATASETS, SEED_DOMAINS,
    )
    from datasmith.imperfections.analyzer import analyze_kg_datasets

    # Filter datasets by source
    datasets = {}
    total_entries = 0
    for domain, entries in SEED_DATASETS.items():
        filtered = [e for e in entries if args.source == "all" or e[0] == args.source]
        if filtered:
            datasets[domain] = filtered
            total_entries += len(filtered)

    if args.dry_run:
        print(f"Would crawl {total_entries} datasets across {len(datasets)} domains:")
        for domain, entries in datasets.items():
            for source, ident, label in entries:
                print(f"  [{source:12s}] {domain:15s} → {label}")
        return

    db_path = str(_PROJECT_ROOT / "data" / "datasmith.db")
    db = Database(db_path)
    kg = KnowledgeGraph(db)

    # ── Step 1: Crawl ──────────────────────────────────────────────────
    logger.info("=== Step 1: Crawling %d datasets across %d domains ===",
                total_entries, len(datasets))
    results = seed_knowledge_graph(kg, datasets, delay=0.5)

    ok = sum(1 for d in results.values() for v in d.values() if v == "ok")
    failed = sum(1 for d in results.values() for v in d.values() if v == "failed")
    logger.info("Crawl results: %d OK, %d failed", ok, failed)
    for domain, entries in results.items():
        for label, status in entries.items():
            logger.info("  %s: %s — %s", domain, label, status)

    # ── Step 2: Analyze ────────────────────────────────────────────────
    logger.info("=== Step 2: Analyzing domains ===")
    analyzed = 0
    for domain_name in SEED_DOMAINS:
        try:
            result = analyze_kg_datasets(kg, domain_name)
            if result:
                analyzed += 1
                logger.info("  %s: %d fingerprint(s)", domain_name, len(result))
        except Exception as e:
            logger.warning("  %s analysis failed: %s", domain_name, e)

    logger.info("Analyzed %d/%d domains", analyzed, len(SEED_DOMAINS))

    # ── Stats ──────────────────────────────────────────────────────────
    domain_count = kg.db.fetchall("SELECT COUNT(*) as c FROM domains")[0]["c"]
    ds_count = kg.db.fetchall("SELECT COUNT(*) as c FROM dataset_schemas")[0]["c"]
    col_count = kg.db.fetchall("SELECT COUNT(*) as c FROM column_schemas")[0]["c"]
    prof_count = kg.db.fetchall(
        "SELECT COUNT(*) as c FROM domain_profiles")[0]["c"]

    print()
    print("=" * 55)
    print(f"  Schema Knowledge Graph — {domain_count} domains, "
          f"{ds_count} datasets, {col_count} columns")
    print(f"  Domain profiles: {prof_count}")
    print("=" * 55)
    for domain_name in SEED_DOMAINS:
        d = kg.get_domain_by_name(domain_name)
        if d:
            dss = kg.list_datasets(domain_id=d.id)
            label = SEED_DOMAINS.get(domain_name, "")
            flag = " ✓" if dss else ""
            print(f"  {domain_name:20s} ({len(dss)} datasets, {label[:40]}){flag}")
        else:
            print(f"  {domain_name:20s} (not seeded)")

    if args.commit:
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            cwd=_PROJECT_ROOT, capture_output=True,
        )
        has_staged = result.returncode != 0
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=_PROJECT_ROOT, capture_output=True,
        )
        has_unstaged = result.returncode != 0

        if has_staged or has_unstaged:
            subprocess.run(["git", "add", "data/datasmith.db"], cwd=_PROJECT_ROOT)
            date = time.strftime("%Y-%m-%d")
            subprocess.run(
                ["git", "commit", "-m", f"daily seed crawl: {date}"],
                cwd=_PROJECT_ROOT,
            )
            logger.info("Committed data/datasmith.db")
        else:
            logger.info("No changes to commit")

    db.close()


if __name__ == "__main__":
    main()
