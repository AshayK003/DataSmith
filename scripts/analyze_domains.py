#!/usr/bin/env python3
"""Analyze datasets in the Knowledge Graph and build imperfection fingerprints.

Usage:
    python scripts/analyze_domains.py                          # All domains
    python scripts/analyze_domains.py --domain e-commerce      # Single domain
    python scripts/analyze_domains.py --list                   # List domains in KG
    python scripts/analyze_domains.py --dry-run                # Show what would run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.imperfections.analyzer import analyze_kg_datasets
from datasmith.imperfections.profiles import (
    merge_profile,
    save_profile_to_kg,
    load_profile_from_kg,
    get_default_profile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_domains")


def show_list(kg: KnowledgeGraph) -> None:
    """Show domains and dataset counts."""
    from datasmith.schema.crawler import SEED_DOMAINS
    stats = kg.stats()
    print(f"\n{'='*55}")
    print(f"  Schema Knowledge Graph — {stats['domains']} domains, "
          f"{stats['datasets']} datasets, {stats['columns']} columns")
    print(f"  Profiles: {stats['profiles']}")
    print(f"{'='*55}")
    for name, desc in SEED_DOMAINS.items():
        domain = kg.get_domain_by_name(name)
        datasets = kg.list_datasets(domain_id=domain.id) if domain else []
        profile = kg.get_domain_profile(domain.id) if domain else None
        flag = " ✓" if profile and profile.profile_json else ""
        print(f"  {name:20s}  ({len(datasets)} datasets, {desc[:30]}...){flag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze datasets in the KG")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List domains")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", type=str, default="data/datasmith.db")
    args = parser.parse_args()

    db = Database(args.db)
    kg = KnowledgeGraph(db)

    if args.list:
        show_list(kg)
        return

    from datasmith.schema.crawler import SEED_DOMAINS
    domains = [args.domain] if args.domain else list(SEED_DOMAINS.keys())

    if args.dry_run:
        print(f"Would analyze: {', '.join(domains)}")
        return

    for domain_name in domains:
        logger.info("Analyzing domain: %s", domain_name)
        fingerprints = analyze_kg_datasets(kg, domain_name)

        if not fingerprints:
            logger.warning("No analysis data for '%s' — saving default profile", domain_name)
            default = get_default_profile(domain_name)
            save_profile_to_kg(kg, domain_name, default)
            continue

        # Merge multiple dataset analyses into one domain profile
        merged = {}
        for fp in fingerprints:
            merged = merge_profile(merged, fp)

        save_profile_to_kg(kg, domain_name, merged)
        logger.info("Saved merged profile for '%s' (%d fingerprint(s))",
                    domain_name, len(fingerprints))

    show_list(kg)


if __name__ == "__main__":
    main()
