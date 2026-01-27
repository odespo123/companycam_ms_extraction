"""Extract equipment info from visit images using Gemini."""

import json
import os
import base64
import time
from pathlib import Path
from google import genai
from google.genai import errors

# Expected output format per equipment:
# {
#     "equipment_type": "hvac" | "water_heater",
#     "model_number": str | null,
#     "serial_number": str | null,
#     "manufacture_date": str | null
# }

EXTRACTION_PROMPT = """Analyze this image from a home service visit. Look for data plates or labels on:
- Outdoor HVAC units (air conditioners, heat pumps, condensers)
- Water heaters

IMPORTANT: Only extract information from clear, close-up shots where you can fully read the data plate/label text. Do NOT attempt to extract from far away shots, angled images, or partially visible labels. If the text is not clearly legible, return an empty array.

For each piece of equipment with a clearly readable data plate, extract:
- equipment_type: "hvac" or "water_heater"
- model_number: the model number if visible
- serial_number: the serial number if visible
- manufacture_date: the manufacture date if visible (any format)

Return a JSON array of equipment found. If no clearly readable data plate is visible, return an empty array [].

Return ONLY valid JSON, no other text."""


def encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64 encode an image, return (data, mime_type)."""
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    return data, mime_types.get(ext, "image/jpeg")


def extract_from_image(client: genai.Client, image_path: str, max_retries: int = 5) -> list[dict]:
    """Extract equipment info from a single image with retry logic."""
    image_data, mime_type = encode_image(image_path)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": image_data}},
                            {"text": EXTRACTION_PROMPT},
                        ],
                    }
                ],
            )

            text = response.text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse response for {image_path}: {text[:100]}")
                return []

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise

    print(f"  Error: Max retries exceeded for {image_path}")
    return []


def aggregate_equipment(all_extractions: list[list[dict]]) -> list[dict]:
    """Merge equipment extractions across images.

    Simple strategy: dedupe by (equipment_type, model_number, serial_number).
    If same equipment appears multiple times, merge fields.
    """
    equipment_map = {}  # key -> equipment dict

    for extractions in all_extractions:
        for eq in extractions:
            # Create a key for deduplication
            key = (
                eq.get("equipment_type"),
                eq.get("model_number"),
                eq.get("serial_number"),
            )

            if key in equipment_map:
                # Merge: fill in any missing fields
                existing = equipment_map[key]
                for field in ["model_number", "serial_number", "manufacture_date"]:
                    if not existing.get(field) and eq.get(field):
                        existing[field] = eq[field]
            else:
                equipment_map[key] = eq.copy()

    return list(equipment_map.values())


def process_visit(visit_dir: str, client: genai.Client = None) -> dict:
    """Process all images in a visit directory and return aggregated results."""
    if client is None:
        client = genai.Client()  # Uses GOOGLE_API_KEY env var

    visit_path = Path(visit_dir)
    image_files = sorted(
        [f for f in visit_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=lambda x: x.name
    )

    print(f"Processing {visit_path.name}: {len(image_files)} images")

    per_image_results = []
    all_extractions = []
    for i, img in enumerate(image_files):
        print(f"  Extracting from {img.name}...")
        extractions = extract_from_image(client, str(img))
        if extractions:
            print(f"    Found {len(extractions)} equipment")
        all_extractions.append(extractions)
        per_image_results.append({
            "image": img.name,
            "equipment": extractions,
        })
        # Rate limit: wait between requests
        if i < len(image_files) - 1:
            time.sleep(1)

    equipment = aggregate_equipment(all_extractions)

    return {
        "visit_id": visit_path.name,
        "per_image": per_image_results,
        "aggregated": equipment,
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <visit_dir> [--save]")
        sys.exit(1)

    visit_dir = sys.argv[1]
    save = "--save" in sys.argv

    result = process_visit(visit_dir)

    print("\nAggregated Results:")
    print(json.dumps(result["aggregated"], indent=2))

    if save:
        output_path = Path(visit_dir) / "output.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
