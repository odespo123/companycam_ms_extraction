"""Export false positives to CSV for manual categorization."""

import csv
import json
from pathlib import Path
from urllib.parse import urlencode, quote

PHOTOS_DIR = Path("photos")
BASE_URL = "https://paschal-company-cam-mse.streamlit.app"


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_source_images(equipment: dict, per_image_results: list) -> list[str]:
    """Find which images contributed to this predicted equipment."""
    source_images = []
    model = equipment.get("model_number")
    serial = equipment.get("serial_number")

    for img_data in per_image_results:
        for eq in img_data.get("equipment", []):
            if (model and eq.get("model_number") == model) or \
               (serial and eq.get("serial_number") == serial):
                source_images.append(img_data["image"])
                break

    return source_images


def main():
    eval_path = PHOTOS_DIR / "eval_results.json"
    if not eval_path.exists():
        print("Error: eval_results.json not found. Run `python eval.py` first.")
        return

    with open(eval_path) as f:
        eval_results = json.load(f)

    rows = []

    for visit in eval_results["per_visit"]:
        visit_id = visit["visit_id"]

        # Load output.json for per-image data
        output = load_json(PHOTOS_DIR / visit_id / "output.json")
        per_image = output.get("per_image", []) if output else []

        # Find false positives (predicted but no ground truth match)
        for match in visit.get("matches", []):
            pred = match.get("predicted")
            gt = match.get("ground_truth")

            if pred and not gt:
                # This is a false positive
                source_imgs = find_source_images(pred, per_image)
                photo_id = source_imgs[0] if source_imgs else ""

                # Build Streamlit URLs
                visit_url = f"{BASE_URL}/?{urlencode({'visit': visit_id})}"
                photo_url = ""
                if photo_id:
                    photo_url = f"{BASE_URL}/?{urlencode({'view_image': visit_id, 'img': photo_id})}"

                row = {
                    "photo_id": photo_id,
                    "visit_id": visit_id,
                    "photo_url": photo_url,
                    "visit_url": visit_url,
                    "equipment_type": pred.get("equipment_type", ""),
                    "category": "",  # For manual labeling
                    "notes": "",     # For manual notes
                }
                rows.append(row)

    # Write TSV (tabs work better with Google Sheets copy-paste)
    output_path = Path("false_positives.tsv")
    fieldnames = ["photo_id", "visit_id", "photo_url", "visit_url",
                  "equipment_type", "category", "notes"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} false positives to {output_path}")
    print("(Tab-separated - paste directly into Google Sheets)")

    # Summary by equipment type
    by_type = {}
    for row in rows:
        eq_type = row["equipment_type"] or "unknown"
        by_type[eq_type] = by_type.get(eq_type, 0) + 1

    print("\nBy equipment type:")
    for eq_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {eq_type}: {count}")


if __name__ == "__main__":
    main()
