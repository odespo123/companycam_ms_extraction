"""Evaluate extraction results against ground truth."""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def match_equipment(predicted: list[dict], ground_truth: list[dict]) -> list[tuple]:
    """Match predicted equipment to ground truth by model/serial overlap.

    Returns list of (pred, gt) tuples. Unmatched pred has gt=None, unmatched gt has pred=None.
    """
    matches = []
    used_gt = set()
    used_pred = set()

    # First pass: exact match on model AND serial (when both exist)
    for i, pred in enumerate(predicted):
        for j, gt in enumerate(ground_truth):
            if j in used_gt:
                continue
            # Match if model numbers match, or serial numbers match (when present)
            model_match = pred.get("model_number") and pred["model_number"] == gt.get("model_number")
            serial_match = pred.get("serial_number") and pred["serial_number"] == gt.get("serial_number")

            if model_match or serial_match:
                matches.append((pred, gt))
                used_pred.add(i)
                used_gt.add(j)
                break

    # Add unmatched predictions (false positives)
    for i, pred in enumerate(predicted):
        if i not in used_pred:
            matches.append((pred, None))

    # Add unmatched ground truth (false negatives)
    for j, gt in enumerate(ground_truth):
        if j not in used_gt:
            matches.append((None, gt))

    return matches


def eval_field(pred_val, gt_val) -> bool:
    """Check if predicted value matches ground truth, handling nulls."""
    # Normalize None and null
    if pred_val is None or pred_val == "null":
        pred_val = None
    if gt_val is None or gt_val == "null":
        gt_val = None

    return pred_val == gt_val


def eval_visit(visit_dir: Path) -> dict:
    """Evaluate a single visit, return metrics."""
    output_path = visit_dir / "output.json"
    gt_path = visit_dir / "groundtruth.json"

    if not output_path.exists():
        return {"error": "No output.json found", "visit_id": visit_dir.name}
    if not gt_path.exists():
        return {"error": "No groundtruth.json found", "visit_id": visit_dir.name}

    output = load_json(output_path)
    gt = load_json(gt_path)

    predicted = output.get("aggregated", [])
    ground_truth = gt.get("equipment", [])

    matches = match_equipment(predicted, ground_truth)

    # Calculate metrics
    true_positives = sum(1 for p, g in matches if p and g)
    false_positives = sum(1 for p, g in matches if p and not g)
    false_negatives = sum(1 for p, g in matches if not p and g)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else None

    # Field accuracy (only on matched pairs)
    matched_pairs = [(p, g) for p, g in matches if p and g]

    field_results = {
        "equipment_type": {"correct": 0, "total": 0},
        "model_number": {"correct": 0, "total": 0},
        "serial_number": {"correct": 0, "total": 0},
        "manufacture_date": {"correct": 0, "total": 0},
    }

    for pred, gt in matched_pairs:
        for field in field_results:
            field_results[field]["total"] += 1
            if eval_field(pred.get(field), gt.get(field)):
                field_results[field]["correct"] += 1

    field_accuracy = {}
    for field, counts in field_results.items():
        if counts["total"] > 0:
            field_accuracy[field] = counts["correct"] / counts["total"]
        else:
            field_accuracy[field] = None

    return {
        "visit_id": visit_dir.name,
        "predicted_count": len(predicted),
        "ground_truth_count": len(ground_truth),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "field_accuracy": field_accuracy,
        "matches": [
            {
                "predicted": p,
                "ground_truth": g,
                "field_correct": {
                    field: eval_field(p.get(field) if p else None, g.get(field) if g else None)
                    for field in ["equipment_type", "model_number", "serial_number", "manufacture_date"]
                } if p and g else None
            }
            for p, g in matches
        ],
    }


def aggregate_results(visit_results: list[dict]) -> dict:
    """Aggregate metrics across all visits."""
    total_tp = sum(v.get("true_positives", 0) for v in visit_results if "error" not in v)
    total_fp = sum(v.get("false_positives", 0) for v in visit_results if "error" not in v)
    total_fn = sum(v.get("false_negatives", 0) for v in visit_results if "error" not in v)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None

    # Aggregate field accuracy
    field_totals = {
        "equipment_type": {"correct": 0, "total": 0},
        "model_number": {"correct": 0, "total": 0},
        "serial_number": {"correct": 0, "total": 0},
        "manufacture_date": {"correct": 0, "total": 0},
    }

    for v in visit_results:
        if "error" in v:
            continue
        for match in v.get("matches", []):
            if match["field_correct"]:
                for field, correct in match["field_correct"].items():
                    field_totals[field]["total"] += 1
                    if correct:
                        field_totals[field]["correct"] += 1

    field_accuracy = {}
    for field, counts in field_totals.items():
        if counts["total"] > 0:
            field_accuracy[field] = counts["correct"] / counts["total"]
        else:
            field_accuracy[field] = None

    return {
        "total_visits": len(visit_results),
        "visits_with_errors": sum(1 for v in visit_results if "error" in v),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "field_accuracy": field_accuracy,
    }


def main():
    import sys

    photos_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("photos")

    visit_dirs = sorted([d for d in photos_dir.iterdir() if d.is_dir()])

    visit_results = []
    for visit_dir in visit_dirs:
        result = eval_visit(visit_dir)
        visit_results.append(result)

        # Print per-visit summary
        if "error" in result:
            print(f"{result.get('visit_id', visit_dir.name)}: {result['error']}")
        else:
            p = result["precision"]
            r = result["recall"]
            if p is not None:
                print(f"{result['visit_id']}: P={p:.0%} R={r:.0%} (pred={result['predicted_count']}, gt={result['ground_truth_count']})")
            else:
                print(f"{result['visit_id']}: No equipment (pred={result['predicted_count']}, gt={result['ground_truth_count']})")

    # Aggregate
    agg = aggregate_results(visit_results)

    print("\n" + "="*50)
    print("AGGREGATE RESULTS")
    print("="*50)
    print(f"Visits: {agg['total_visits']} ({agg['visits_with_errors']} with errors)")
    print(f"Equipment: TP={agg['total_true_positives']}, FP={agg['total_false_positives']}, FN={agg['total_false_negatives']}")
    if agg["precision"] is not None:
        print(f"Precision: {agg['precision']:.1%}")
        print(f"Recall: {agg['recall']:.1%}")
    print(f"\nField Accuracy:")
    for field, acc in agg["field_accuracy"].items():
        if acc is not None:
            print(f"  {field}: {acc:.1%}")
        else:
            print(f"  {field}: N/A")

    # Save full results
    output = {
        "per_visit": visit_results,
        "aggregate": agg,
    }
    output_path = photos_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
