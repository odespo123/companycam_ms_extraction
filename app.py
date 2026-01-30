"""Streamlit app to visualize extraction eval results."""

import streamlit as st
import json
import urllib.parse
import base64
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Equipment Extraction Eval", layout="wide")

PHOTOS_DIR = Path("photos")

# R2 configuration - set R2_PUBLIC_URL in environment or Streamlit secrets
# Format: https://pub-xxxxx.r2.dev or custom domain
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL")
if not R2_PUBLIC_URL:
    try:
        R2_PUBLIC_URL = st.secrets.get("R2_PUBLIC_URL")
    except Exception:
        R2_PUBLIC_URL = None


def get_image_url(visit_id: str, image_name: str) -> str | None:
    """Get image URL - local path if exists, otherwise R2 URL."""
    local_path = PHOTOS_DIR / visit_id / image_name
    if local_path.exists():
        return str(local_path)

    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/photos/{visit_id}/{urllib.parse.quote(image_name)}"

    return None


def get_image_b64(visit_id: str, image_name: str) -> str | None:
    """Get base64 encoded image data for clickable thumbnails."""
    local_path = PHOTOS_DIR / visit_id / image_name

    if local_path.exists():
        with open(local_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    if R2_PUBLIC_URL:
        url = f"{R2_PUBLIC_URL}/photos/{visit_id}/{urllib.parse.quote(image_name)}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return base64.b64encode(resp.content).decode()
        except Exception:
            pass

    return None


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
        st.metric("Precision", f"{precision:.1%}" if precision is not None else "N/A",
                  help="Of equipment we predicted, how many were correct? Low precision = hallucinations")
    with col3:
        recall = agg["recall"]
        st.metric("Recall", f"{recall:.1%}" if recall is not None else "N/A",
                  help="Of equipment that exists, how many did we find? Low recall = missed equipment")
    with col4:
        st.metric("Equipment Found", agg["total_true_positives"])

    # Visit type breakdown
    st.subheader("Visit Types")
    hvac_types = {"outdoor_condenser", "air_handler", "evaporator_coil", "furnace"}
    plumbing_types = {"water_heater"}

    hvac_only = 0
    plumbing_only = 0
    hvac_and_plumbing = 0
    none_count = 0

    for visit in eval_results["per_visit"]:
        gt_types = set()
        by_type = visit.get("by_equipment_type", {})
        for eq_type, data in by_type.items():
            if data.get("expected", 0) > 0:
                gt_types.add(eq_type)

        has_hvac = bool(gt_types & hvac_types)
        has_plumbing = bool(gt_types & plumbing_types)

        if has_hvac and has_plumbing:
            hvac_and_plumbing += 1
        elif has_hvac:
            hvac_only += 1
        elif has_plumbing:
            plumbing_only += 1
        else:
            none_count += 1

    type_cols = st.columns(4)
    with type_cols[0]:
        st.metric("HVAC Only", hvac_only)
    with type_cols[1]:
        st.metric("Plumbing Only", plumbing_only)
    with type_cols[2]:
        st.metric("HVAC + Plumbing", hvac_and_plumbing)
    with type_cols[3]:
        st.metric("None", none_count)

    # Field accuracy
    st.subheader("Field Accuracy")
    st.caption("For matched equipment, how often did we get each field exactly correct?")
    field_cols = st.columns(4)
    for i, (field, acc) in enumerate(agg["field_accuracy"].items()):
        with field_cols[i]:
            st.metric(field.replace("_", " ").title(), f"{acc:.1%}" if acc is not None else "N/A")

    # Detection summary
    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Positives", agg["total_true_positives"], help="Correctly found equipment")
    with col2:
        st.metric("False Positives", agg["total_false_positives"], help="Hallucinated equipment")
    with col3:
        st.metric("False Negatives", agg["total_false_negatives"], help="Missed equipment")

    # Metrics by equipment type
    by_type = agg.get("by_equipment_type", {})
    if by_type:
        st.subheader("Metrics by Equipment Type")
        for eq_type in sorted(by_type.keys()):
            data = by_type[eq_type]
            with st.expander(f"**{eq_type.replace('_', ' ').title()}** — Expected: {data['expected']}, Found: {data['found']}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    p = data.get("precision")
                    st.metric("Precision", f"{p:.0%}" if p is not None else "N/A")
                with col2:
                    r = data.get("recall")
                    st.metric("Recall", f"{r:.0%}" if r is not None else "N/A")
                with col3:
                    st.metric("TP / FP / FN", f"{data['tp']} / {data['fp']} / {data['fn']}")
                with col4:
                    st.metric("Expected", data['expected'])

                # Field accuracy for this type
                fa = data.get("field_accuracy", {})
                if any(v is not None for v in fa.values()):
                    st.caption("Field Accuracy")
                    fcols = st.columns(3)
                    for i, (field, acc) in enumerate(fa.items()):
                        with fcols[i]:
                            st.metric(field.replace("_", " ").title(), f"{acc:.0%}" if acc is not None else "N/A")

                # Filter links
                st.caption("View visits:")
                link_cols = st.columns(4)
                with link_cols[0]:
                    if st.button(f"✓ Correct ({data['tp']})", key=f"tp_{eq_type}", disabled=data['tp']==0):
                        st.session_state.filter_type = eq_type
                        st.session_state.filter_status = "correct"
                        st.rerun()
                with link_cols[1]:
                    if st.button(f"✗ Missed ({data['fn']})", key=f"fn_{eq_type}", disabled=data['fn']==0):
                        st.session_state.filter_type = eq_type
                        st.session_state.filter_status = "missed"
                        st.rerun()
                with link_cols[2]:
                    if st.button(f"✗ Extra ({data['fp']})", key=f"fp_{eq_type}", disabled=data['fp']==0):
                        st.session_state.filter_type = eq_type
                        st.session_state.filter_status = "extra"
                        st.rerun()
                with link_cols[3]:
                    if st.button("All", key=f"all_{eq_type}"):
                        st.session_state.filter_type = eq_type
                        st.session_state.filter_status = "all"
                        st.rerun()

    st.divider()

    # Categorize visits
    def is_perfect(visit):
        p = visit.get("precision")
        r = visit.get("recall")
        if visit["ground_truth_count"] == 0 and visit["predicted_count"] == 0:
            return True
        if p != 1.0 or r != 1.0:
            return False
        # Also check all fields are 100% correct
        fa = visit.get("field_accuracy", {})
        for acc in fa.values():
            if acc is not None and acc != 1.0:
                return False
        return True

    # Equipment type filter
    def visit_matches_filter(visit, eq_type, status):
        """Check if visit matches the current filter."""
        visit_by_type = visit.get("by_equipment_type", {})
        if eq_type not in visit_by_type:
            return False
        type_data = visit_by_type[eq_type]
        if status == "all":
            return True
        elif status == "correct":
            return type_data["tp"] > 0
        elif status == "missed":
            return type_data["fn"] > 0
        elif status == "extra":
            return type_data["fp"] > 0
        return False

    # Apply filter if active
    filter_type = st.session_state.get("filter_type")
    filter_status = st.session_state.get("filter_status")

    all_visits = eval_results["per_visit"]
    if filter_type and filter_status:
        all_visits = [v for v in all_visits if visit_matches_filter(v, filter_type, filter_status)]

        # Show filter indicator
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(f"Filtered: {filter_type.replace('_', ' ').title()} — {filter_status.title()} ({len(all_visits)} visits)")
        with col2:
            if st.button("Clear filter"):
                st.session_state.filter_type = None
                st.session_state.filter_status = None
                st.rerun()

    perfect_visits = [v for v in all_visits if is_perfect(v)]
    needs_review = [v for v in all_visits if not is_perfect(v)]

    # Progress bar
    total = len(all_visits)
    perfect_count = len(perfect_visits)
    st.progress(perfect_count / total if total > 0 else 0)
    st.caption(f"{perfect_count}/{total} visits perfect ({perfect_count/total*100:.0f}%)" if total > 0 else "No visits")

    # Tabs
    tab_review, tab_perfect, tab_all = st.tabs([
        f"Needs Review ({len(needs_review)})",
        f"Perfect ({len(perfect_visits)})",
        f"All ({total})"
    ])

    def render_visit_grid(visits, key_prefix):
        """Render visits in a grid with links that open in new tabs."""
        if not visits:
            st.write("No visits")
            return

        cols_per_row = 4
        for i in range(0, len(visits), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(visits):
                    break

                visit = visits[idx]
                visit_id = visit["visit_id"]
                p = visit.get("precision")
                r = visit.get("recall")

                # Determine status and color (must match is_perfect logic)
                if is_perfect(visit):
                    status = "✓"
                    color = "#d4edda"  # green
                elif p is None:
                    status = "−"
                    color = "#e2e3e5"  # gray
                else:
                    status = "✗"
                    color = "#f8d7da"  # red

                with col:
                    # Card-like container
                    params = urllib.parse.urlencode({"visit": visit_id})

                    # Short visit ID for display (job number)
                    short_id = visit_id.split(" - ")[0] if " - " in visit_id else visit_id[:15]

                    # Stats line
                    if p is not None and r is not None:
                        stats = f"P={p:.0%} R={r:.0%}"
                    else:
                        stats = "No equipment"

                    eq_count = f"{visit['predicted_count']}/{visit['ground_truth_count']}"

                    # Field accuracy line
                    fa = visit.get("field_accuracy", {})
                    field_parts = []
                    for field, acc in fa.items():
                        if acc is not None:
                            short_name = {"equipment_type": "type", "model_number": "mod",
                                          "serial_number": "ser", "manufacture_date": "date"}[field]
                            field_parts.append(f"{short_name}={acc:.0%}")
                    field_line = " ".join(field_parts) if field_parts else ""

                    # Ground truth equipment types
                    by_type = visit.get("by_equipment_type", {})
                    gt_types = []
                    for eq_type, data in by_type.items():
                        if data.get("expected", 0) > 0:
                            short_type = {"outdoor_condenser": "OC", "air_handler": "AH",
                                          "evaporator_coil": "EC", "furnace": "FU",
                                          "water_heater": "WH"}.get(eq_type, eq_type[:2].upper())
                            gt_types.append(short_type)
                    gt_line = " ".join(gt_types) if gt_types else "—"

                    # Build card with link
                    card_content = f'{status} {short_id}<br>'
                    card_content += f'<span style="font-size: 14px; color: #666;">{stats}</span><br>'
                    card_content += f'<span style="font-size: 13px; color: #555;">GT: {gt_line} | pred/gt: {eq_count}</span>'
                    if field_line:
                        card_content += f'<br><span style="font-size: 12px; color: #666;">{field_line}</span>'

                    st.markdown(
                        f'''<a href="?{params}" target="_blank" style="text-decoration: none; display: block;">
                        <div style="background: {color}; padding: 12px; border-radius: 8px; margin-bottom: 8px; color: #333; font-weight: bold; font-size: 16px;">
                        {card_content}
                        </div></a>''',
                        unsafe_allow_html=True
                    )

    with tab_review:
        render_visit_grid(needs_review, "review")

    with tab_perfect:
        render_visit_grid(perfect_visits, "perfect")

    with tab_all:
        render_visit_grid(all_visits, "all")


def find_source_images_for_prediction(equipment: dict, per_image_results: list) -> list[str]:
    """Find which images contributed to this predicted equipment.

    Uses same priority matching as eval.py:
    1. Both model AND serial match (strongest)
    2. Serial matches (serials are unique)
    3. Model matches (weakest - multiple units can share model)
    """
    model = equipment.get("model_number")
    serial = equipment.get("serial_number")

    # Pass 1: both model AND serial match
    for img_data in per_image_results:
        for eq in img_data.get("equipment", []):
            if model and serial and eq.get("model_number") == model and eq.get("serial_number") == serial:
                return [img_data["image"]]

    # Pass 2: serial matches
    for img_data in per_image_results:
        for eq in img_data.get("equipment", []):
            if serial and eq.get("serial_number") == serial:
                return [img_data["image"]]

    # Pass 3: model matches
    for img_data in per_image_results:
        for eq in img_data.get("equipment", []):
            if model and eq.get("model_number") == model:
                return [img_data["image"]]

    return []


def find_source_image_for_groundtruth(equipment: dict, metadata: dict) -> str | None:
    """Find the source image for this ground truth equipment."""
    source_images = metadata.get("source_images", [])
    model = equipment.get("model_number")
    serial = equipment.get("serial_number")

    for src in source_images:
        for ext in src.get("extractions", []):
            if (model and ext.get("model_number") == model) or \
               (serial and ext.get("serial_number") == serial):
                return src.get("image")

    return None


def render_visit(visit_id, eval_results):
    """Render single visit detail view."""

    # Scroll to top when entering visit view
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.markdown(
        """<script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>""",
        unsafe_allow_html=True
    )

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
        st.metric("Precision", f"{p:.0%}" if p is not None else "N/A",
                  help="Of equipment we predicted, how many were correct?")
    with col2:
        r = visit_eval.get("recall")
        st.metric("Recall", f"{r:.0%}" if r is not None else "N/A",
                  help="Of equipment that exists, how many did we find?")
    with col3:
        st.metric("Predicted", visit_eval.get("predicted_count", 0))
    with col4:
        st.metric("Ground Truth", visit_eval.get("ground_truth_count", 0))

    # Expected vs Found by equipment type
    by_type = visit_eval.get("by_equipment_type", {})
    if by_type:
        st.subheader("By Equipment Type")
        for eq_type in sorted(by_type.keys()):
            data = by_type[eq_type]
            expected = data["expected"]
            found = data["found"]
            tp, fp, fn = data["tp"], data["fp"], data["fn"]

            # Status indicator
            if tp == expected and fp == 0:
                status = "✓"
            elif fn > 0:
                status = "✗ missed"
            elif fp > 0:
                status = "✗ extra"
            else:
                status = ""

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{eq_type.replace('_', ' ').title()}**")
            with col2:
                st.write(f"Expected: {expected}, Found: {found}")
            with col3:
                st.write(status)

    # Field accuracy for this visit
    fa = visit_eval.get("field_accuracy", {})
    if any(v is not None for v in fa.values()):
        st.subheader("Field Accuracy")
        field_cols = st.columns(4)
        for i, (field, acc) in enumerate(fa.items()):
            with field_cols[i]:
                display_val = f"{acc:.0%}" if acc is not None else "N/A"
                st.metric(field.replace("_", " ").title(), display_val)

    st.divider()

    # Side by side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted")
        if output and output.get("aggregated"):
            per_image = output.get("per_image", [])
            matches = visit_eval.get("matches", [])

            # Build list of predictions with match status
            predictions_with_status = []
            for eq in output["aggregated"]:
                # Find if this prediction matched a ground truth
                matched_gt = None
                field_correct = None
                for m in matches:
                    pred = m.get("predicted")
                    if pred and pred.get("model_number") == eq.get("model_number") and \
                       pred.get("serial_number") == eq.get("serial_number"):
                        matched_gt = m.get("ground_truth")
                        field_correct = m.get("field_correct")
                        break

                predictions_with_status.append({
                    "eq": eq,
                    "matched": matched_gt is not None,
                    "matched_gt": matched_gt,
                    "field_correct": field_correct,
                })

            # Sort: correct first, then incorrect
            predictions_with_status.sort(key=lambda x: (not x["matched"], x["eq"].get("equipment_type", "")))

            for item in predictions_with_status:
                eq = item["eq"]
                matched = item["matched"]
                field_correct = item["field_correct"]

                # Determine status: fully correct, partial, or extra
                wrong_fields = []
                if matched and field_correct:
                    wrong_fields = [f for f, correct in field_correct.items() if not correct and f != "equipment_type"]

                if not matched:
                    status_icon = "✗ extra"
                    is_problem = True
                elif wrong_fields:
                    status_icon = "⚠ partial"
                    is_problem = True
                else:
                    status_icon = "✓"
                    is_problem = False

                eq_type = eq.get('equipment_type', 'unknown')
                label = f"{status_icon} {eq_type.replace('_', ' ').title()}"

                with st.expander(label, expanded=False):
                    st.write(f"Model: `{eq.get('model_number', 'N/A')}`")
                    st.write(f"Serial: `{eq.get('serial_number', 'N/A')}`")
                    st.write(f"Mfg Date: `{eq.get('manufacture_date', 'N/A')}`")

                    # Show field correctness if matched
                    if wrong_fields:
                        st.warning(f"Wrong fields: {', '.join(wrong_fields)}")

                    # Show source images (toggleable)
                    source_imgs = find_source_images_for_prediction(eq, per_image)
                    if source_imgs:
                        img_url = get_image_url(visit_id, source_imgs[0])
                        if img_url:
                            img_key = f"pred_img_{eq.get('model_number')}_{eq.get('serial_number')}"
                            if img_key not in st.session_state:
                                st.session_state[img_key] = False

                            if st.session_state[img_key]:
                                if st.button(f"📷 Hide image", key=f"hide_{img_key}"):
                                    st.session_state[img_key] = False
                                    st.rerun()
                                st.image(img_url, use_container_width=True)
                            else:
                                if st.button(f"📷 {source_imgs[0]}", key=f"show_{img_key}"):
                                    st.session_state[img_key] = True
                                    st.rerun()
        else:
            st.write("No equipment predicted")

    with col2:
        st.subheader("Ground Truth")
        if groundtruth and groundtruth.get("equipment"):
            metadata = groundtruth.get("_metadata", {})

            for eq in groundtruth["equipment"]:
                eq_type = eq.get('equipment_type', 'unknown')
                label = eq_type.replace('_', ' ').title()

                with st.expander(label):
                    st.write(f"Model: `{eq.get('model_number', 'N/A')}`")
                    st.write(f"Serial: `{eq.get('serial_number', 'N/A')}`")
                    st.write(f"Mfg Date: `{eq.get('manufacture_date', 'N/A')}`")

                    # Show source image (toggleable)
                    source_img = find_source_image_for_groundtruth(eq, metadata)
                    if source_img:
                        img_url = get_image_url(visit_id, source_img)
                        if img_url:
                            img_key = f"gt_img_{eq.get('model_number')}_{eq.get('serial_number')}"
                            if img_key not in st.session_state:
                                st.session_state[img_key] = False

                            if st.session_state[img_key]:
                                if st.button(f"📷 Hide image", key=f"hide_{img_key}"):
                                    st.session_state[img_key] = False
                                    st.rerun()
                                st.image(img_url, use_container_width=True)
                            else:
                                if st.button(f"📷 {source_img}", key=f"show_{img_key}"):
                                    st.session_state[img_key] = True
                                    st.rerun()
        else:
            st.write("No equipment in ground truth")

    st.divider()

    # Per-image results
    st.subheader("Per-Image Results")
    st.info("The AI is instructed to only extract equipment data when there's a clearly visible data plate (i.e., a \"money shot\"). Images without legible plates are intentionally skipped.")

    if output and output.get("per_image"):
        # Sort option
        sort_option = st.radio(
            "Sort by",
            ["Original order", "Predicted equipment first"],
            horizontal=True,
            key=f"sort_{visit_id}"
        )

        per_image_data = output["per_image"]
        if sort_option == "Predicted equipment first":
            per_image_data = sorted(
                per_image_data,
                key=lambda x: (len(x.get("equipment", [])) == 0, x["image"])
            )

        cols_per_row = 4

        for i in range(0, len(per_image_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(per_image_data):
                    break

                img_data = per_image_data[idx]
                image_name = img_data["image"]
                equipment = img_data.get("equipment", [])

                with col:
                    img_b64 = get_image_b64(visit_id, image_name)
                    if img_b64:
                        # Clickable thumbnail that opens full image view
                        params = urllib.parse.urlencode({"view_image": visit_id, "img": image_name})
                        st.markdown(
                            f'<a href="?{params}" target="_blank">'
                            f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; cursor:pointer;">'
                            f'</a>',
                            unsafe_allow_html=True
                        )

                        # Show full equipment data
                        if equipment:
                            for eq in equipment:
                                eq_type = eq.get('equipment_type', 'unknown').replace('_', ' ').title()
                                model = eq.get('model_number') or '—'
                                serial = eq.get('serial_number') or '—'
                                mfg_date = eq.get('manufacture_date') or '—'
                                st.caption(f"**{eq_type}**")
                                st.caption(f"Model: `{model}`")
                                st.caption(f"Serial: `{serial}`")
                                st.caption(f"Date: `{mfg_date}`")
                        else:
                            st.caption("No equipment")
                    else:
                        st.write(f"Not found: {image_name}")


def render_image_view(visit_id: str, image_name: str):
    """Render dedicated full-size image view with extraction data."""
    image_url = get_image_url(visit_id, image_name)
    if not image_url:
        st.error(f"Image not found: {visit_id}/{image_name}")
        return

    # Load output to get extraction data for this image
    output = load_visit_output(visit_id)
    equipment = []
    if output:
        for img_data in output.get("per_image", []):
            if img_data["image"] == image_name:
                equipment = img_data.get("equipment", [])
                break

    # Two-column layout: image left, prediction right
    col_img, col_info = st.columns([3, 2])

    with col_img:
        # Constrain image height to fit in viewport
        st.markdown(
            """<style>
            [data-testid="stImage"] img { max-height: 80vh; width: auto; object-fit: contain; }
            </style>""",
            unsafe_allow_html=True
        )
        st.image(image_url, use_container_width=True)

    with col_info:
        st.subheader("Predicted Equipment")
        if equipment:
            for eq in equipment:
                eq_type = eq.get('equipment_type', 'unknown').replace('_', ' ').title()
                st.markdown(f"**{eq_type}**")
                st.write(f"**Model:** `{eq.get('model_number') or '—'}`")
                st.write(f"**Serial:** `{eq.get('serial_number') or '—'}`")
                st.write(f"**Mfg Date:** `{eq.get('manufacture_date') or '—'}`")
                st.divider()
        else:
            st.write("No equipment extracted from this image")

        st.caption(f"Image: {image_name}")


def main():
    query_params = st.query_params

    # Check for image view query params
    if "view_image" in query_params and "img" in query_params:
        visit_id = query_params["view_image"]
        image_name = query_params["img"]
        render_image_view(visit_id, image_name)
        return

    # Check for visit query param (opened in new tab)
    if "visit" in query_params:
        visit_id = query_params["visit"]
        eval_results = load_eval_results()
        if eval_results:
            render_visit(visit_id, eval_results)
        else:
            st.error("No eval_results.json found.")
        return

    eval_results = load_eval_results()

    if not eval_results:
        st.error("No eval_results.json found. Run `python eval.py` first.")
        return

    render_home(eval_results)


if __name__ == "__main__":
    main()
