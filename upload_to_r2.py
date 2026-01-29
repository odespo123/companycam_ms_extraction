#!/usr/bin/env python3
"""Upload images to Cloudflare R2 for Streamlit Cloud deployment.

Requires these environment variables in .env:
    R2_ACCOUNT_ID - Your Cloudflare account ID
    R2_ACCESS_KEY_ID - R2 API access key
    R2_SECRET_ACCESS_KEY - R2 API secret key
    R2_BUCKET_NAME - Your R2 bucket name

Usage:
    python upload_to_r2.py                    # Upload all images
    python upload_to_r2.py --visit 123456...  # Upload specific visit
    python upload_to_r2.py --dry-run          # Preview without uploading
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("Please install boto3: pip install boto3")
    sys.exit(1)


def get_r2_client():
    """Create R2 client using S3-compatible API."""
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key, secret_key]):
        print("Error: Missing R2 credentials. Set these in .env:")
        print("  R2_ACCOUNT_ID")
        print("  R2_ACCESS_KEY_ID")
        print("  R2_SECRET_ACCESS_KEY")
        print("  R2_BUCKET_NAME")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )


def get_existing_keys(client, bucket: str, prefix: str = "") -> set:
    """Get set of existing object keys in bucket."""
    existing = set()
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])

    return existing


def upload_images(
    photos_dir: Path,
    bucket: str,
    client,
    visit_filter: str = None,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """Upload images from photos directory to R2."""
    visit_dirs = sorted([d for d in photos_dir.iterdir() if d.is_dir()])

    if visit_filter:
        visit_dirs = [d for d in visit_dirs if visit_filter in d.name]

    if not visit_dirs:
        print("No visit directories found")
        return

    print(f"Found {len(visit_dirs)} visit directories")

    # Get existing files if skipping
    existing_keys = set()
    if skip_existing and not dry_run:
        print("Checking existing files in R2...")
        existing_keys = get_existing_keys(client, bucket, "photos/")
        print(f"Found {len(existing_keys)} existing files")

    uploaded = 0
    skipped = 0
    errors = 0

    for visit_dir in visit_dirs:
        visit_id = visit_dir.name
        image_files = [
            f for f in visit_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ]

        if not image_files:
            continue

        print(f"\n{visit_id}: {len(image_files)} images")

        for img_file in image_files:
            key = f"photos/{visit_id}/{img_file.name}"

            if key in existing_keys:
                skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would upload: {key}")
                uploaded += 1
                continue

            try:
                content_type = "image/jpeg" if img_file.suffix.lower() in ('.jpg', '.jpeg') else "image/png"
                client.upload_file(
                    str(img_file),
                    bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                )
                uploaded += 1
                print(f"  Uploaded: {img_file.name}")
            except Exception as e:
                errors += 1
                print(f"  Error uploading {img_file.name}: {e}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Uploaded: {uploaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Upload images to Cloudflare R2")
    parser.add_argument(
        "--photos-dir",
        default="photos",
        help="Directory containing visit folders (default: photos)",
    )
    parser.add_argument(
        "--visit",
        help="Only upload specific visit (partial match)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview uploads without actually uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload all files even if they already exist",
    )

    args = parser.parse_args()

    bucket = os.environ.get("R2_BUCKET_NAME")
    if not bucket:
        print("Error: R2_BUCKET_NAME not set in .env")
        sys.exit(1)

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        print(f"Error: Photos directory not found: {photos_dir}")
        sys.exit(1)

    client = get_r2_client()

    print(f"Bucket: {bucket}")
    print(f"Photos dir: {photos_dir}")
    if args.visit:
        print(f"Filter: {args.visit}")
    if args.dry_run:
        print("Mode: DRY RUN")

    upload_images(
        photos_dir=photos_dir,
        bucket=bucket,
        client=client,
        visit_filter=args.visit,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
