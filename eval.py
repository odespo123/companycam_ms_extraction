"""Evaluate extraction results against ground truth."""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def match_equipment(predicted: list[dict], ground_truth: list[dict]) -> list[tuple]:
    """Match predicted equipment to ground truth by model/serial overlap.

    Uses priority matching:
    1. Both model AND serial match (strongest)
    2. Serial matches (serials are unique)
    3. Model matches (weakest - multiple units can share model)

    Returns list of (pred, gt) tuples. Unmatched pred has gt=None, unmatched gt has pred=None.
    """
    matches = []
    used_gt = set()
    used_pred = set()

    def model_match(pred, gt):
        return pred.get("model_number") and pred["model_number"] == gt.get("model_number")

    def serial_match(pred, gt):
        return pred.get("serial_number") and pred["serial_number"] == gt.get("serial_number")

    # Pass 1: both model AND serial match
    for i, pred in enumerate(predicted):
        if i in used_pred:
            continue
        for j, gt in enumerate(ground_truth):
            if j in used_gt:
                continue
            if model_match(pred, gt) and serial_match(pred, gt):
                matches.append((pred, gt))
                used_pred.add(i)
                used_gt.add(j)
                break

    # Pass 2: serial matches
    for i, pred in enumerate(predicted):
        if i in used_pred:
            continue
        for j, gt in enumerate(ground_truth):
            if j in used_gt:
                continue
            if serial_match(pred, gt):
                matches.append((pred, gt))
                used_pred.add(i)
                used_gt.add(j)
                break

    # Pass 3: model matches
    for i, pred in enumerate(predicted):
        if i in used_pred:
            continue
        for j, gt in enumerate(ground_truth):
            if j in used_gt:
                continue
            if model_match(pred, gt):
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


def eval_visit(visit_dir: Path) -> dict | None:
    """Evaluate a single visit, return metrics or None if missing files."""
    output_path = visit_dir / "output.json"
    gt_path = visit_dir / "groundtruth.json"

    if not output_path.exists() or not gt_path.exists():
        return None

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

    # Per-equipment-type breakdown (detection + field accuracy)
    by_type = {}
    for p, g in matches:
        # Use ground truth type if available, else predicted type
        eq_type = (g.get("equipment_type") if g else None) or (p.get("equipment_type") if p else "unknown")
        if eq_type not in by_type:
            by_type[eq_type] = {
                "expected": 0,
                "found": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "field_correct": {"model_number": 0, "serial_number": 0, "manufacture_date": 0},
                "field_total": {"model_number": 0, "serial_number": 0, "manufacture_date": 0},
            }

        if g:
            by_type[eq_type]["expected"] += 1
        if p and g:
            by_type[eq_type]["tp"] += 1
            by_type[eq_type]["found"] += 1
            # Track field accuracy for matched pairs
            for field in ["model_number", "serial_number", "manufacture_date"]:
                by_type[eq_type]["field_total"][field] += 1
                if eval_field(p.get(field), g.get(field)):
                    by_type[eq_type]["field_correct"][field] += 1
        elif p and not g:
            by_type[eq_type]["fp"] += 1
            by_type[eq_type]["found"] += 1
        elif not p and g:
            by_type[eq_type]["fn"] += 1

    # Compute field accuracy percentages per type
    for eq_type, data in by_type.items():
        data["field_accuracy"] = {}
        for field in ["model_number", "serial_number", "manufacture_date"]:
            if data["field_total"][field] > 0:
                data["field_accuracy"][field] = data["field_correct"][field] / data["field_total"][field]
            else:
                data["field_accuracy"][field] = None

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
        "by_equipment_type": by_type,
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


def build_confusion_matrix(visit_results: list[dict]) -> dict:
    """Build confusion matrix between predicted and ground truth equipment types.

    Returns dict with:
    - matrix: dict of {(gt_type, pred_type): count}
    - gt_types: list of ground truth types (including "None" for FP)
    - pred_types: list of predicted types (including "None" for FN)
    """
    from collections import defaultdict

    matrix = defaultdict(int)
    gt_types_set = set()
    pred_types_set = set()

    for visit in visit_results:
        for match in visit.get("matches", []):
            pred = match.get("predicted")
            gt = match.get("ground_truth")

            pred_type = pred.get("equipment_type") if pred else "None"
            gt_type = gt.get("equipment_type") if gt else "None"

            matrix[(gt_type, pred_type)] += 1
            gt_types_set.add(gt_type)
            pred_types_set.add(pred_type)

    # Sort types, but keep "None" at the end
    def sort_key(t):
        return (t == "None", t)

    gt_types = sorted(gt_types_set, key=sort_key)
    pred_types = sorted(pred_types_set, key=sort_key)

    return {
        "matrix": {f"{gt}|{pred}": count for (gt, pred), count in matrix.items()},
        "gt_types": gt_types,
        "pred_types": pred_types,
    }


def aggregate_results(visit_results: list[dict]) -> dict:
    """Aggregate metrics across all visits."""
    total_tp = sum(v.get("true_positives", 0) for v in visit_results)
    total_fp = sum(v.get("false_positives", 0) for v in visit_results)
    total_fn = sum(v.get("false_negatives", 0) for v in visit_results)

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

    # Aggregate by equipment type
    by_type = {}
    for v in visit_results:
        for eq_type, data in v.get("by_equipment_type", {}).items():
            if eq_type not in by_type:
                by_type[eq_type] = {
                    "expected": 0,
                    "found": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "field_correct": {"model_number": 0, "serial_number": 0, "manufacture_date": 0},
                    "field_total": {"model_number": 0, "serial_number": 0, "manufacture_date": 0},
                }
            by_type[eq_type]["expected"] += data["expected"]
            by_type[eq_type]["found"] += data["found"]
            by_type[eq_type]["tp"] += data["tp"]
            by_type[eq_type]["fp"] += data["fp"]
            by_type[eq_type]["fn"] += data["fn"]
            for field in ["model_number", "serial_number", "manufacture_date"]:
                by_type[eq_type]["field_correct"][field] += data["field_correct"][field]
                by_type[eq_type]["field_total"][field] += data["field_total"][field]

    # Compute precision, recall, field accuracy per type
    for eq_type, data in by_type.items():
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        data["precision"] = tp / (tp + fp) if (tp + fp) > 0 else None
        data["recall"] = tp / (tp + fn) if (tp + fn) > 0 else None
        data["field_accuracy"] = {}
        for field in ["model_number", "serial_number", "manufacture_date"]:
            if data["field_total"][field] > 0:
                data["field_accuracy"][field] = data["field_correct"][field] / data["field_total"][field]
            else:
                data["field_accuracy"][field] = None

    # Build confusion matrix
    confusion = build_confusion_matrix(visit_results)

    return {
        "total_visits": len(visit_results),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "field_accuracy": field_accuracy,
        "by_equipment_type": by_type,
        "confusion_matrix": confusion,
    }


def main():
    import sys

    photos_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("photos")

    visit_dirs = sorted([d for d in photos_dir.iterdir() if d.is_dir()])

    visit_results = []
    skipped = 0
    for visit_dir in visit_dirs:
        result = eval_visit(visit_dir)

        if result is None:
            skipped += 1
            continue

        visit_results.append(result)

        # Print per-visit summary
        p = result["precision"]
        r = result["recall"]
        if p is not None and r is not None:
            print(f"{result['visit_id']}: P={p:.0%} R={r:.0%} (pred={result['predicted_count']}, gt={result['ground_truth_count']})")
        else:
            print(f"{result['visit_id']}: No equipment (pred={result['predicted_count']}, gt={result['ground_truth_count']})")

    if skipped > 0:
        print(f"\n(Skipped {skipped} visits missing output.json or groundtruth.json)")

    # Aggregate
    agg = aggregate_results(visit_results)

    print("\n" + "="*50)
    print("AGGREGATE RESULTS")
    print("="*50)
    print(f"Visits: {agg['total_visits']}")
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
