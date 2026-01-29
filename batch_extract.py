#!/usr/bin/env python3
"""
Run extraction on visit directories to generate output.json files.

Usage:
    python batch_extract.py                           # all visits in photos/
    python batch_extract.py --csv test_eval_gt_setup.csv  # only visits from CSV
    python batch_extract.py --force                   # regenerate even if exists
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from extractor import process_visit
from utils import make_visit_id, parse_date
import json


def get_visit_ids_from_csv(csv_path: str) -> list[str]:
    """Extract visit IDs from CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    visit_ids = []
    for _, row in df.iterrows():
        # Skip rows marked as ignore
        project_link = row.get('Matched Project Link', '')
        if pd.notna(project_link) and 'IGNORE THIS ROW' in str(project_link).upper():
            continue

        job_id = row['Job ID']
        address = str(row['Location Address']).strip()
        completion_date = parse_date(row['Completion Date'])
        date_str = completion_date.strftime('%Y-%m-%d') if completion_date else 'unknown'

        visit_id = make_visit_id(str(job_id), address, date_str)
        visit_ids.append(visit_id)

    return visit_ids


def main():
    parser = argparse.ArgumentParser(
        description="Run extraction on visit directories"
    )
    parser.add_argument(
        "--photos-dir",
        default="photos",
        help="Directory containing visit folders (default: photos)"
    )
    parser.add_argument(
        "--csv",
        help="Only process visits from this CSV file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate output.json even if it already exists"
    )

    args = parser.parse_args()

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        print(f"Error: Photos directory not found: {photos_dir}")
        sys.exit(1)

    # Get visit directories to process
    if args.csv:
        print(f"Reading visits from: {args.csv}")
        visit_ids = get_visit_ids_from_csv(args.csv)
        visit_dirs = [photos_dir / vid for vid in visit_ids if (photos_dir / vid).exists()]
        print(f"Found {len(visit_dirs)} matching visit directories\n")
    else:
        # Find all visit directories (directories with images)
        visit_dirs = sorted([
            d for d in photos_dir.iterdir()
            if d.is_dir() and any(f.suffix.lower() in ('.jpg', '.jpeg', '.png') for f in d.iterdir())
        ])

    print(f"Found {len(visit_dirs)} visit directories\n")

    results = {'success': [], 'skipped': [], 'errors': []}

    for i, visit_dir in enumerate(visit_dirs):
        print(f"[{i + 1}/{len(visit_dirs)}] {visit_dir.name}")

        output_path = visit_dir / "output.json"

        # Skip if already exists (unless --force)
        if output_path.exists() and not args.force:
            print(f"  ✓ output.json exists, skipping (use --force to regenerate)")
            results['skipped'].append(visit_dir.name)
            continue

        try:
            result = process_visit(str(visit_dir))

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            equipment_count = len(result.get('aggregated', []))
            print(f"  ✓ Saved output.json ({equipment_count} equipment found)")
            results['success'].append(visit_dir.name)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['errors'].append(visit_dir.name)

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Processed: {len(results['success'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")
    print(f"✗ Errors: {len(results['errors'])}")

    if results['errors']:
        print(f"\nFailed visits:")
        for v in results['errors']:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
