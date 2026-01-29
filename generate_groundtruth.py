#!/usr/bin/env python3
"""
Generate ground truth files by running extraction on specific labeled images.

Reads a CSV file identifying which images contain extractable data plates,
runs the extraction model on those images, and outputs groundtruth.json files
for human review.

Usage:
    python generate_groundtruth.py labeled_images.csv

CSV format:
    Job ID,Job #,Job Type,Business Unit,Location Address,Completion Date,
    Matched Project Link,Image Link,Image Type,Image Link,Image Type,...

    (Image Link, Image Type) pairs can repeat for multiple images per visit.
    Image Type should be: "hvac" or "water_heater"
    Image Link can be a CompanyCam URL or a filename in the visit directory.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    import pandas as pd
except ImportError:
    print("Please install pandas: pip install pandas")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("Please install google-genai: pip install google-genai")
    sys.exit(1)

from utils import make_visit_id, parse_date
from extractor import extract_from_image, aggregate_equipment


# Map CSV equipment types to standardized types
EQUIPMENT_TYPE_MAP = {
    'outdoor condensor': 'outdoor_condenser',
    'outdoor condenser': 'outdoor_condenser',
    'air handler': 'air_handler',
    'evaporator coil': 'evaporator_coil',
    'furnace': 'furnace',
    'water heater': 'water_heater',
}


def normalize_equipment_type(raw_type: str) -> str:
    """Normalize equipment type from CSV to standardized format."""
    raw_type = raw_type.lower().strip()
    return EQUIPMENT_TYPE_MAP.get(raw_type, raw_type.replace(' ', '_'))


def extract_image_pairs(row: pd.Series) -> list[tuple[str, str]]:
    """Extract (Image Link, Image Type) pairs from a row.

    Handles variable number of image pairs, including those in unnamed
    columns beyond the headers.
    """
    pairs = []
    columns = list(row.index)

    # Find the first "Image Link" column
    start_idx = None
    for i, col in enumerate(columns):
        if str(col).lower().strip() == 'image link':
            start_idx = i
            break

    if start_idx is None:
        return pairs

    # From that point, read pairs of values (link, type) by position
    i = start_idx
    while i + 1 < len(row):
        link = row.iloc[i]
        img_type = row.iloc[i + 1]

        # Stop if we hit empty values
        if pd.isna(link) or not str(link).strip():
            break

        link_str = str(link).strip()
        type_str = str(img_type).strip().lower() if pd.notna(img_type) else 'unknown'
        pairs.append((link_str, type_str))

        i += 2

    return pairs


def find_image_in_visit(visit_dir: Path, image_filename: str) -> Path | None:
    """Find the image file in the visit directory by filename."""
    # Try exact filename match
    exact_match = visit_dir / image_filename
    if exact_match.exists():
        return exact_match

    # Try case-insensitive match
    image_filename_lower = image_filename.lower()
    for img_file in visit_dir.iterdir():
        if img_file.name.lower() == image_filename_lower:
            return img_file

    return None


def process_csv(csv_path: str, photos_dir: Path, force: bool = False):
    """Process the CSV file and generate ground truth for each visit."""
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = ['Job ID', 'Location Address', 'Completion Date']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Found {len(df)} rows in CSV file\n")

    # Initialize Gemini client
    client = genai.Client()

    results = {
        'success': [],
        'skipped_ignore': [],
        'no_visit_dir': [],
        'no_images': [],
        'count_mismatch': [],
        'errors': [],
    }

    for idx, row in df.iterrows():
        job_id = row['Job ID']
        address = str(row['Location Address']).strip()
        completion_date = parse_date(row['Completion Date'])
        date_str = completion_date.strftime('%Y-%m-%d') if completion_date else 'unknown'

        visit_id = make_visit_id(str(job_id), address, date_str)
        visit_dir = photos_dir / visit_id

        print(f"[{idx + 1}/{len(df)}] Processing: {visit_id}")

        # Check for "IGNORE THIS ROW" in Matched Project Link
        project_link = row.get('Matched Project Link', '')
        if pd.notna(project_link) and 'IGNORE THIS ROW' in str(project_link).upper():
            print(f"  ⊘ Skipping (marked as ignore)")
            results['skipped_ignore'].append(visit_id)
            continue

        # Check if visit directory exists
        if not visit_dir.exists():
            print(f"  ✗ Visit directory not found: {visit_dir.name}")
            results['no_visit_dir'].append(visit_id)
            continue

        # Check if groundtruth.json already exists
        gt_path = visit_dir / "groundtruth.json"
        if gt_path.exists() and not force:
            print(f"  ✓ Ground truth already exists, skipping (use --force to regenerate)")
            results['success'].append(visit_id)
            continue

        # Extract image pairs
        image_pairs = extract_image_pairs(row)

        # Get expected unique equipment count (try both column name variants)
        unique_equipment_count = row.get('Unique Equipment Pieces', row.get('Unique Equipment', None))
        if pd.notna(unique_equipment_count):
            unique_equipment_count = int(unique_equipment_count)

            # Validate count matches image pairs
            if len(image_pairs) != unique_equipment_count:
                print(f"  ✗ Count mismatch: {len(image_pairs)} image pairs vs {unique_equipment_count} unique equipment")
                results['count_mismatch'].append(f"{visit_id} (got {len(image_pairs)}, expected {unique_equipment_count})")
                continue

        # Handle case with no equipment (empty ground truth)
        if not image_pairs:
            print(f"  No equipment images - creating empty ground truth")
            groundtruth = {
                "equipment": [],
                "_metadata": {
                    "generated_by": "generate_groundtruth.py",
                    "generated_at": datetime.now().isoformat(),
                    "source_images": [],
                    "status": "NEEDS_REVIEW"
                }
            }
            with open(gt_path, 'w') as f:
                json.dump(groundtruth, f, indent=2)
            print(f"  ✓ Generated empty groundtruth.json")
            results['success'].append(visit_id)
            continue

        print(f"  Found {len(image_pairs)} labeled images")

        # Process each image
        all_extractions = []
        processed_images = []

        for img_link, img_type in image_pairs:
            print(f"    Processing: {img_link[:50]}... (type: {img_type})")

            # Find the image in the visit directory
            img_path = find_image_in_visit(visit_dir, img_link)

            if not img_path:
                print(f"      ✗ Image not found in visit directory")
                continue

            print(f"      Found: {img_path.name}")

            # Extract equipment from image using shared extractor
            extractions = extract_from_image(client, str(img_path))

            # Override equipment_type with the labeled type (normalized)
            normalized_type = normalize_equipment_type(img_type)
            for eq in extractions:
                eq['equipment_type'] = normalized_type

            if extractions:
                print(f"      Extracted {len(extractions)} equipment")
                all_extractions.append(extractions)
                processed_images.append({
                    'image': img_path.name,
                    'type': img_type,
                    'extractions': extractions
                })
            else:
                print(f"      No equipment extracted")

            # Rate limiting
            time.sleep(0.2)

        if not all_extractions:
            print(f"  ✗ No equipment extracted from any image")
            results['no_images'].append(visit_id)
            continue

        # Aggregate equipment
        equipment = aggregate_equipment(all_extractions)

        # Write ground truth file
        groundtruth = {
            "equipment": equipment,
            "_metadata": {
                "generated_by": "generate_groundtruth.py",
                "generated_at": datetime.now().isoformat(),
                "source_images": processed_images,
                "status": "NEEDS_REVIEW"
            }
        }

        with open(gt_path, 'w') as f:
            json.dump(groundtruth, f, indent=2)

        print(f"  ✓ Generated groundtruth.json with {len(equipment)} equipment")
        print(f"    REMINDER: Please review {gt_path.name} for accuracy")
        results['success'].append(visit_id)
        print()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Generated: {len(results['success'])}")
    print(f"⊘ Skipped (ignore): {len(results['skipped_ignore'])}")
    print(f"✗ Visit dir not found: {len(results['no_visit_dir'])}")
    print(f"✗ Count mismatch: {len(results['count_mismatch'])}")
    print(f"✗ No extractions: {len(results['no_images'])}")
    print(f"✗ Errors: {len(results['errors'])}")

    if results['count_mismatch']:
        print(f"\nCount mismatches (image pairs vs unique equipment):")
        for v in results['count_mismatch'][:10]:
            print(f"  - {v}")
        if len(results['count_mismatch']) > 10:
            print(f"  ... and {len(results['count_mismatch']) - 10} more")

    print("\n⚠️  IMPORTANT: Review all generated groundtruth.json files for accuracy!")
    print("    Look for files with '_metadata.status': 'NEEDS_REVIEW'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth files from labeled images"
    )
    parser.add_argument(
        "csv_file",
        help="CSV file with labeled images"
    )
    parser.add_argument(
        "--photos-dir",
        default="photos",
        help="Directory containing visit folders (default: photos)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate groundtruth.json even if it already exists"
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        print(f"Error: Photos directory not found: {photos_dir}")
        sys.exit(1)

    process_csv(args.csv_file, photos_dir, force=args.force)


if __name__ == "__main__":
    main()
