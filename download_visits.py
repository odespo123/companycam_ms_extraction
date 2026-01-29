#!/usr/bin/env python3
"""
Download photos from CompanyCam for evaluation visits.

Reads an Excel file with job information and downloads corresponding photos
from CompanyCam into visit directories.

Usage:
    python download_visits.py jobs.xlsx --token YOUR_API_TOKEN

    # Or set token as environment variable:
    export COMPANYCAM_TOKEN=YOUR_API_TOKEN
    python download_visits.py jobs.xlsx
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests

try:
    import pandas as pd
except ImportError:
    print("Please install pandas: pip install pandas openpyxl")
    sys.exit(1)

from utils import normalize_address, sanitize_filename, make_visit_id, parse_date


# CompanyCam API base URL
API_BASE = "https://api.companycam.com/v2"

# Rate limiting: be conservative
REQUEST_DELAY = 0.5  # seconds between API calls


class CompanyCamClient:
    """Simple client for CompanyCam API."""

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict | list:
        """Make a GET request to the API with retry logic."""
        url = f"{API_BASE}{endpoint}"
        time.sleep(REQUEST_DELAY)  # Rate limiting

        last_error = None
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"    Network error (attempt {attempt + 1}/{retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                # Don't retry HTTP errors (4xx, 5xx)
                raise

        raise last_error

    def search_projects(self, query: str) -> list[dict]:
        """Search for projects by name or address."""
        projects = []
        page = 1

        while True:
            result = self._get("/projects", params={
                "query": query,
                "per_page": 50,
                "page": page,
            })

            if not result:
                break

            projects.extend(result)

            if len(result) < 50:
                break
            page += 1

        return projects

    def find_project_by_address(self, address: str) -> dict | None:
        """Find a project matching the given address."""
        # Extract street address for search (first line)
        street = address.split(',')[0].strip()

        print(f"    Searching with: '{street}'")
        projects = self.search_projects(street)

        # Try normalized address if no results
        if not projects:
            normalized = normalize_address(street)
            if normalized != street:
                print(f"    Trying normalized: '{normalized}'")
                projects = self.search_projects(normalized)

        # Try with just the street number and first few words
        if not projects:
            parts = street.split()
            if len(parts) >= 2:
                short_query = ' '.join(parts[:3])
                print(f"    Trying short query: '{short_query}'")
                projects = self.search_projects(short_query)

        if not projects:
            return None

        # If multiple matches, try to find best match
        # Normalize both for comparison
        address_normalized = normalize_address(address.lower())
        for project in projects:
            proj_addr = project.get('address', {})
            proj_street = (proj_addr.get('street_address_1') or '').lower()
            proj_street_normalized = normalize_address(proj_street)

            # Check if either original or normalized matches
            if proj_street and (
                proj_street in address.lower() or
                proj_street_normalized in address_normalized
            ):
                return project

        # Return first match if no exact match
        return projects[0] if projects else None

    def get_project_photos(
        self,
        project_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> list[dict]:
        """Get photos for a project, optionally filtered by date range."""
        photos = []
        page = 1

        params = {"per_page": 50, "page": page}

        if start_date:
            params["start_date"] = int(start_date.timestamp())
        if end_date:
            params["end_date"] = int(end_date.timestamp())

        while True:
            params["page"] = page
            result = self._get(f"/projects/{project_id}/photos", params=params)

            if not result:
                break

            photos.extend(result)

            if len(result) < 50:
                break
            page += 1

        return photos

    def download_photo(self, photo: dict, dest_path: Path) -> bool:
        """Download a photo to the specified path."""
        # Get the best available URI (prefer 'original', fall back to 'web')
        uris = photo.get('uris', [])

        url = None
        for uri in uris:
            if uri.get('type') == 'original':
                url = uri.get('url')
                break

        if not url:
            for uri in uris:
                if uri.get('type') == 'web':
                    url = uri.get('url')
                    break

        if not url and uris:
            url = uris[0].get('url')

        if not url:
            return False

        try:
            time.sleep(REQUEST_DELAY)
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    Error downloading photo: {e}")
            return False


def process_excel(excel_path: str, client: CompanyCamClient, output_dir: Path):
    """Process the Excel file and download photos for each visit."""
    print(f"Reading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)

    # Normalize column names (handle variations in spacing/case)
    df.columns = df.columns.str.strip()

    required_cols = ['Job ID', 'Location Address', 'Completion Date']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Found {len(df)} rows in Excel file\n")

    results = {
        'success': [],
        'no_project': [],
        'no_photos': [],
        'errors': [],
    }

    for idx, row in df.iterrows():
        job_id = row['Job ID']
        address = str(row['Location Address']).strip()
        completion_date = parse_date(row['Completion Date'])

        # For display
        date_str = completion_date.strftime('%Y-%m-%d') if completion_date else 'unknown'
        visit_id = make_visit_id(str(job_id), address, date_str)

        print(f"[{idx + 1}/{len(df)}] Processing: {visit_id}")

        # Check if already downloaded (resume capability)
        visit_dir = output_dir / visit_id
        link_file = visit_dir / "project_link.txt"
        if link_file.exists():
            print(f"  ✓ Already downloaded, skipping")
            results['success'].append(visit_id)
            continue

        # Find project
        print(f"  Searching for project: {address[:50]}...")
        project = client.find_project_by_address(address)

        if not project:
            print(f"  ✗ No project found for address")
            results['no_project'].append(visit_id)
            continue

        project_id = project['id']
        project_name = project.get('name', 'Unnamed')
        print(f"  Found project: {project_name} (ID: {project_id})")

        # Get photos for completion date (same day)
        if completion_date:
            start = completion_date.replace(hour=0, minute=0, second=0)
            end = completion_date.replace(hour=23, minute=59, second=59)
        else:
            start = None
            end = None

        print(f"  Fetching photos for date: {date_str}")
        photos = client.get_project_photos(project_id, start_date=start, end_date=end)

        if not photos:
            print(f"  ✗ No photos found for this date")
            results['no_photos'].append(visit_id)
            continue

        print(f"  Found {len(photos)} photos")

        # Create visit directory
        visit_dir.mkdir(parents=True, exist_ok=True)

        # Write project link file for easy reference
        project_url = f"https://app.companycam.com/projects/{project_id}"
        with open(link_file, 'w') as f:
            f.write(f"CompanyCam Project: {project_name}\n")
            f.write(f"Project ID: {project_id}\n")
            f.write(f"URL: {project_url}\n")
            f.write(f"\nJob ID: {job_id}\n")
            f.write(f"Address: {address}\n")
            f.write(f"Completion Date: {date_str}\n")

        # Download photos
        downloaded = 0
        for i, photo in enumerate(photos):
            photo_id = photo.get('id', f'photo_{i}')
            captured_at = photo.get('captured_at')

            # Create filename with timestamp
            if captured_at:
                ts = datetime.fromtimestamp(captured_at)
                filename = f"{i+1:02d}_{ts.strftime('%H%M%S')}_{photo_id}.jpg"
            else:
                filename = f"{i+1:02d}_{photo_id}.jpg"

            dest_path = visit_dir / filename

            if dest_path.exists():
                print(f"    Skipping (exists): {filename}")
                downloaded += 1
                continue

            print(f"    Downloading: {filename}")
            if client.download_photo(photo, dest_path):
                downloaded += 1

        print(f"  ✓ Downloaded {downloaded}/{len(photos)} photos to {visit_dir.name}")
        results['success'].append(visit_id)
        print()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ No project found: {len(results['no_project'])}")
    print(f"✗ No photos for date: {len(results['no_photos'])}")
    print(f"✗ Errors: {len(results['errors'])}")

    if results['no_project']:
        print(f"\nVisits with no matching project:")
        for v in results['no_project'][:10]:
            print(f"  - {v}")
        if len(results['no_project']) > 10:
            print(f"  ... and {len(results['no_project']) - 10} more")

    if results['no_photos']:
        print(f"\nVisits with no photos on completion date:")
        for v in results['no_photos'][:10]:
            print(f"  - {v}")
        if len(results['no_photos']) > 10:
            print(f"  ... and {len(results['no_photos']) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Download CompanyCam photos for evaluation visits"
    )
    parser.add_argument(
        "excel_file",
        help="Excel file with job information"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("COMPANYCAM_TOKEN"),
        help="CompanyCam API token (or set COMPANYCAM_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        default="photos",
        help="Output directory for downloaded photos (default: photos)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search for projects but don't download photos"
    )

    args = parser.parse_args()

    if not args.token:
        print("Error: No API token provided.")
        print("Set COMPANYCAM_TOKEN environment variable or use --token flag")
        sys.exit(1)

    if not os.path.exists(args.excel_file):
        print(f"Error: Excel file not found: {args.excel_file}")
        sys.exit(1)

    client = CompanyCamClient(args.token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_excel(args.excel_file, client, output_dir)


if __name__ == "__main__":
    main()
