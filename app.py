"""Streamlit app to visualize extraction eval results."""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Equipment Extraction Eval", layout="wide")

PHOTOS_DIR = Path("photos")


@st.cache_data
def load_eval_results():
    eval_path = PHOTOS_DIR / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_visit_output(visit_id):
    output_path = PHOTOS_DIR / visit_id / "output.json"
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_visit_groundtruth(visit_id):
    gt_path = PHOTOS_DIR / visit_id / "groundtruth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return None


def render_home(eval_results):
    """Render aggregate results home screen."""
    st.title("Equipment Extraction Eval")

    agg = eval_results["aggregate"]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Visits", agg["total_visits"])
    with col2:
        precision = agg["precision"]
        st.metric("Precision", f"{precision:.1%}" if precision else "N/A",
                  help="Of equipment we predicted, how many were correct? Low precision = hallucinations")
    with col3:
        recall = agg["recall"]
        st.metric("Recall", f"{recall:.1%}" if recall else "N/A",
                  help="Of equipment that exists, how many did we find? Low recall = missed equipment")
    with col4:
        st.metric("Equipment Found", agg["total_true_positives"])

    # Field accuracy
    st.subheader("Field Accuracy")
    st.caption("For matched equipment, how often did we get each field exactly correct?")
    field_cols = st.columns(4)
    for i, (field, acc) in enumerate(agg["field_accuracy"].items()):
        with field_cols[i]:
            st.metric(field.replace("_", " ").title(), f"{acc:.1%}" if acc else "N/A")

    # Detection summary
    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Positives", agg["total_true_positives"], help="Correctly found equipment")
    with col2:
        st.metric("False Positives", agg["total_false_positives"], help="Hallucinated equipment")
    with col3:
        st.metric("False Negatives", agg["total_false_negatives"], help="Missed equipment")

    st.divider()

    # Visit list
    st.subheader("Visits")
    for visit in eval_results["per_visit"]:
        visit_id = visit["visit_id"]

        if "error" in visit:
            status = "⚠️"
            summary = visit["error"]
        elif visit["ground_truth_count"] == 0 and visit["predicted_count"] == 0:
            status = "✓"
            summary = "No equipment (correct)"
        else:
            p = visit["precision"]
            r = visit["recall"]
            if p == 1.0 and r == 1.0:
                status = "✓"
            elif p is None:
                status = "−"
            else:
                status = "✗" if p < 1.0 or r < 1.0 else "✓"
            summary = f"P={p:.0%} R={r:.0%}" if p else "No predictions"

            # Add field accuracy
            fa = visit.get("field_accuracy", {})
            field_parts = []
            for field, acc in fa.items():
                if acc is not None:
                    short_name = {"equipment_type": "type", "model_number": "model",
                                  "serial_number": "serial", "manufacture_date": "date"}[field]
                    field_parts.append(f"{short_name}={acc:.0%}")
            if field_parts:
                summary += f" | {' '.join(field_parts)}"

            summary += f" (pred={visit['predicted_count']}, gt={visit['ground_truth_count']})"

        col1, col2, col3 = st.columns([1, 4, 3])
        with col1:
            st.write(status)
        with col2:
            if st.button(visit_id, key=f"btn_{visit_id}", use_container_width=True):
                st.session_state.selected_visit = visit_id
                st.rerun()
        with col3:
            st.write(summary)


def render_visit(visit_id, eval_results):
    """Render single visit detail view."""

    if st.button("← Back to Home"):
        st.session_state.selected_visit = None
        st.rerun()

    st.title(f"Visit: {visit_id}")

    visit_eval = next((v for v in eval_results["per_visit"] if v["visit_id"] == visit_id), None)
    output = load_visit_output(visit_id)
    groundtruth = load_visit_groundtruth(visit_id)

    if not visit_eval:
        st.error("Visit not found in eval results")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        p = visit_eval.get("precision")
        st.metric("Precision", f"{p:.0%}" if p else "N/A",
                  help="Of equipment we predicted, how many were correct?")
    with col2:
        r = visit_eval.get("recall")
        st.metric("Recall", f"{r:.0%}" if r else "N/A",
                  help="Of equipment that exists, how many did we find?")
    with col3:
        st.metric("Predicted", visit_eval.get("predicted_count", 0))
    with col4:
        st.metric("Ground Truth", visit_eval.get("ground_truth_count", 0))

    # Field accuracy for this visit
    fa = visit_eval.get("field_accuracy", {})
    if any(v is not None for v in fa.values()):
        st.subheader("Field Accuracy")
        field_cols = st.columns(4)
        for i, (field, acc) in enumerate(fa.items()):
            with field_cols[i]:
                st.metric(field.replace("_", " ").title(), f"{acc:.0%}" if acc else "N/A")

    st.divider()

    # Side by side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted")
        if output and output.get("aggregated"):
            for eq in output["aggregated"]:
                with st.container(border=True):
                    st.write(f"**{eq.get('equipment_type', 'unknown')}**")
                    st.write(f"Model: `{eq.get('model_number', 'N/A')}`")
                    st.write(f"Serial: `{eq.get('serial_number', 'N/A')}`")
                    st.write(f"Mfg Date: `{eq.get('manufacture_date', 'N/A')}`")
        else:
            st.write("No equipment predicted")

    with col2:
        st.subheader("Ground Truth")
        if groundtruth and groundtruth.get("equipment"):
            for eq in groundtruth["equipment"]:
                with st.container(border=True):
                    st.write(f"**{eq.get('equipment_type', 'unknown')}**")
                    st.write(f"Model: `{eq.get('model_number', 'N/A')}`")
                    st.write(f"Serial: `{eq.get('serial_number', 'N/A')}`")
                    st.write(f"Mfg Date: `{eq.get('manufacture_date', 'N/A')}`")
        else:
            st.write("No equipment in ground truth")

    st.divider()

    # Per-image results
    st.subheader("Per-Image Results")

    if output and output.get("per_image"):
        for img_data in output["per_image"]:
            image_name = img_data["image"]
            equipment = img_data.get("equipment", [])
            image_path = PHOTOS_DIR / visit_id / image_name

            col1, col2 = st.columns([1, 1])
            with col1:
                if image_path.exists():
                    st.image(str(image_path), caption=image_name, use_container_width=True)
                else:
                    st.write(f"Image not found: {image_name}")
            with col2:
                if equipment:
                    for eq in equipment:
                        with st.container(border=True):
                            st.write(f"**{eq.get('equipment_type', 'unknown')}**")
                            st.write(f"Model: `{eq.get('model_number', 'N/A')}`")
                            st.write(f"Serial: `{eq.get('serial_number', 'N/A')}`")
                            st.write(f"Mfg Date: `{eq.get('manufacture_date', 'N/A')}`")
                else:
                    st.write("No equipment found")

            st.divider()


def main():
    eval_results = load_eval_results()

    if not eval_results:
        st.error("No eval_results.json found. Run `python eval.py` first.")
        return

    if "selected_visit" not in st.session_state:
        st.session_state.selected_visit = None

    if st.session_state.selected_visit:
        render_visit(st.session_state.selected_visit, eval_results)
    else:
        render_home(eval_results)


if __name__ == "__main__":
    main()
