#!/usr/bin/env python3
"""Crawl new datasets from Kaggle and update the Schema Knowledge Graph.

Usage:
    python scripts/crawl_schemas.py                     # Seed crawl (first run)
    python scripts/crawl_schemas.py --incremental        # Crawl new datasets only
    python scripts/crawl_schemas.py --domain healthcare  # Single domain
    python scripts/crawl_schemas.py --stats              # Show KG stats only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasmith.core.database import Database
from datasmith.schema.crawler import seed_knowledge_graph
from datasmith.schema.knowledge_graph import KnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crawl_schemas")


def show_stats(kg: KnowledgeGraph) -> None:
    stats = kg.stats()
    print(f"\n{'='*50}")
    print(f"  Schema Knowledge Graph Stats")
    print(f"{'='*50}")
    print(f"  Domains:    {stats['domains']}")
    print(f"  Datasets:   {stats['datasets']}")
    print(f"  Columns:    {stats['columns']}")
    print(f"  Profiles:   {stats['profiles']}")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl datasets into the Schema KG")
    parser.add_argument("--incremental", action="store_true",
                        help="Crawl new datasets only")
    parser.add_argument("--domain", type=str, default=None,
                        help="Crawl a single domain")
    parser.add_argument("--stats", action="store_true",
                        help="Show KG stats and exit")
    parser.add_argument("--db", type=str, default="data/datasmith.db",
                        help="Path to SQLite database")
    args = parser.parse_args()

    db = Database(args.db)
    kg = KnowledgeGraph(db)

    if args.stats:
        show_stats(kg)
        return

    if args.domain:
        from datasmith.schema.crawler import SEED_DATASETS
        if args.domain not in SEED_DATASETS:
            logger.error("Unknown domain '%s'. Available: %s",
                         args.domain, list(SEED_DATASETS.keys()))
            sys.exit(1)
        datasets = {args.domain: SEED_DATASETS[args.domain]}
    else:
        from datasmith.schema.crawler import SEED_DATASETS
        datasets = SEED_DATASETS

    logger.info("Starting crawl across %d domains", len(datasets))
    results = seed_knowledge_graph(kg, datasets)

    # Summary
    ok = sum(1 for r in results.values()
             for v in r.values() if v == "ok")
    failed = sum(1 for r in results.values()
                 for v in r.values() if v == "failed")
    logger.info("Crawl complete: %d datasets OK, %d failed", ok, failed)
    show_stats(kg)


if __name__ == "__main__":
    main()
