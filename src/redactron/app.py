"""Redactron Streamlit dashboard — redaction QA monitoring UI."""

from __future__ import annotations

import multiprocessing
import threading
from datetime import datetime
from pathlib import Path

# Required for Windows where multiprocessing uses 'spawn' instead of 'fork'.
multiprocessing.freeze_support()

import pandas as pd
import streamlit as st

from redactron import config
from redactron.core.pipeline import process_batch
from redactron.models.database import export_batch_json, load_batch, load_batches
from redactron.models.schemas import BatchResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Redactron", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "processing": False,
    "progress_current": 0,
    "progress_total": 0,
    "progress_filename": "",
    "batch_result": None,
    "selected_batch_id": None,
    "error_message": None,
}

for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    return Path.cwd() / config.DATABASE_FILENAME


def _color_for_metric(value: float) -> str:
    """Return CSS-friendly colour based on metric quality band."""
    if value > 0.95:
        return "green"
    if value > 0.80:
        return "orange"
    return "red"


def _metric_delta_color(value: float) -> str:
    """Return streamlit delta_color keyword based on value quality."""
    if value > 0.95:
        return "normal"
    if value > 0.80:
        return "off"
    return "inverse"


def _flat_findings_df(batch: BatchResult) -> pd.DataFrame:
    """Flatten all findings across documents into a single DataFrame."""
    rows: list[dict] = []
    for doc_result in batch.documents:
        for f in doc_result.findings:
            rows.append(
                {
                    "Document": doc_result.document.filename,
                    "Page": f.page,
                    "PII Type": f.entity_type,
                    "Snippet": f.text[:50] + ("..." if len(f.text) > 50 else ""),
                    "Full Text": f.text,
                    "Confidence": round(f.confidence, 3),
                    "Source": f.source,
                }
            )
    return pd.DataFrame(rows)


def _run_processing(
    directory: str,
    name: str,
    confidence_threshold: float,
    risk_weights: dict[str, int],
) -> None:
    """Run batch processing in a background thread."""

    def _progress(processed: int, total: int, filename: str) -> None:
        st.session_state.progress_current = processed
        st.session_state.progress_total = total
        st.session_state.progress_filename = filename

    try:
        result = process_batch(
            directory=directory,
            name=name,
            confidence_threshold=confidence_threshold,
            risk_weights=risk_weights,
            progress_callback=_progress,
            db_path=_db_path(),
        )
        st.session_state.batch_result = result
        st.session_state.selected_batch_id = result.batch_id
    except Exception as exc:  # noqa: BLE001
        st.session_state.error_message = str(exc)
    finally:
        st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Redactron")
    st.markdown("**Redaction Quality Assurance**")
    st.divider()

    batch_dir = st.text_input("Batch directory path", placeholder="/path/to/documents")
    batch_name = st.text_input(
        "Batch name",
        value=f"batch-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    with st.expander("Risk Weights"):
        risk_weights: dict[str, int] = {}
        for entity_type, default_weight in config.DEFAULT_RISK_WEIGHTS.items():
            risk_weights[entity_type] = st.number_input(
                entity_type,
                min_value=0,
                max_value=20,
                value=default_weight,
                key=f"weight_{entity_type}",
            )

    st.divider()

    col_start, col_export = st.columns(2)

    with col_start:
        start_disabled = st.session_state.processing or not batch_dir
        if st.button("Start Processing", disabled=start_disabled, use_container_width=True):
            st.session_state.processing = True
            st.session_state.batch_result = None
            st.session_state.error_message = None
            st.session_state.progress_current = 0
            st.session_state.progress_total = 0
            st.session_state.progress_filename = ""
            thread = threading.Thread(
                target=_run_processing,
                args=(batch_dir, batch_name, confidence_threshold, risk_weights),
                daemon=True,
            )
            thread.start()
            st.rerun()

    with col_export:
        export_disabled = st.session_state.batch_result is None
        if st.button("Export JSON", disabled=export_disabled, use_container_width=True):
            batch: BatchResult = st.session_state.batch_result
            out_path = Path(batch.directory) / f"{batch.name}.json"
            export_batch_json(batch, out_path)
            st.success(f"Exported to {out_path}")

# ---------------------------------------------------------------------------
# Auto-refresh while processing
# ---------------------------------------------------------------------------

if st.session_state.processing:
    import time

    time.sleep(1)
    st.rerun()

# ---------------------------------------------------------------------------
# Error banner
# ---------------------------------------------------------------------------

if st.session_state.error_message:
    st.error(f"Processing error: {st.session_state.error_message}")
    st.session_state.error_message = None

# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_findings, tab_history = st.tabs(["Dashboard", "Findings", "History"])

# ---- Dashboard tab --------------------------------------------------------

with tab_dashboard:
    # Progress bar while processing
    if st.session_state.processing:
        total = st.session_state.progress_total or 1
        current = st.session_state.progress_current
        filename = st.session_state.progress_filename
        st.progress(
            current / total,
            text=f"Processing file {current} of {total}: {filename}",
        )

    # Resolve which batch to display
    active_batch: BatchResult | None = st.session_state.batch_result
    if active_batch is None and st.session_state.selected_batch_id:
        active_batch = load_batch(_db_path(), st.session_state.selected_batch_id)

    if active_batch is not None:
        m = active_batch.metrics

        # --- Metric cards ---
        c1, c2, c3 = st.columns(3)

        with c1:
            clean_pct = m.clean_doc_rate * 100
            color = _color_for_metric(m.clean_doc_rate)
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<h4 style='margin:0'>Clean Doc Rate</h4>"
                f"<h2 style='margin:0; color:{color}'>{clean_pct:.1f}%</h2></div>",
                unsafe_allow_html=True,
            )

        with c2:
            redact_pct = m.entity_redaction_rate * 100
            color = _color_for_metric(m.entity_redaction_rate)
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<h4 style='margin:0'>Entity Redaction Rate</h4>"
                f"<h2 style='margin:0; color:{color}'>{redact_pct:.1f}%</h2></div>",
                unsafe_allow_html=True,
            )

        with c3:
            risk = m.weighted_risk_score
            # Risk score: 1.0 = perfect (no missed PII), 0.0 = worst
            risk_color = _color_for_metric(risk)
            st.markdown(
                f"<div style='border-left: 4px solid {risk_color}; padding-left: 12px;'>"
                f"<h4 style='margin:0'>Weighted Risk Score</h4>"
                f"<h2 style='margin:0; color:{risk_color}'>{risk:.3f}</h2></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"**{m.total_documents}** documents | **{m.total_findings}** findings"
        )
        st.divider()

        # --- Charts ---
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.subheader("Findings by Entity Type")
            if m.findings_by_type:
                type_df = pd.DataFrame(
                    list(m.findings_by_type.items()),
                    columns=["Entity Type", "Count"],
                ).set_index("Entity Type")
                st.bar_chart(type_df)
            else:
                st.info("No findings to chart.")

        with chart_right:
            st.subheader("Findings by Source")
            findings_df = _flat_findings_df(active_batch)
            if not findings_df.empty:
                source_counts = findings_df["Source"].value_counts()
                source_df = pd.DataFrame(
                    {"Source": source_counts.index, "Count": source_counts.values}
                )
                # Use plotly-style pie via matplotlib-free approach
                import altair as alt

                pie = (
                    alt.Chart(source_df)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Source", type="nominal"),
                        tooltip=["Source", "Count"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(pie, use_container_width=True)
            else:
                st.info("No findings to chart.")
    else:
        st.info("Run a batch or select one from History to see the dashboard.")

# ---- Findings tab ---------------------------------------------------------

with tab_findings:
    display_batch: BatchResult | None = st.session_state.batch_result
    if display_batch is None and st.session_state.selected_batch_id:
        display_batch = load_batch(_db_path(), st.session_state.selected_batch_id)

    if display_batch is not None:
        findings_df = _flat_findings_df(display_batch)

        if findings_df.empty:
            st.success("No findings — all documents appear clean.")
        else:
            # Filters
            filter_cols = st.columns(3)
            with filter_cols[0]:
                types = ["All"] + sorted(findings_df["PII Type"].unique().tolist())
                type_filter = st.selectbox("Filter by entity type", types, key="filter_type")
            with filter_cols[1]:
                sources = ["All"] + sorted(findings_df["Source"].unique().tolist())
                source_filter = st.selectbox("Filter by source", sources, key="filter_source")
            with filter_cols[2]:
                docs = ["All"] + sorted(findings_df["Document"].unique().tolist())
                doc_filter = st.selectbox("Filter by document", docs, key="filter_doc")

            filtered = findings_df.copy()
            if type_filter != "All":
                filtered = filtered[filtered["PII Type"] == type_filter]
            if source_filter != "All":
                filtered = filtered[filtered["Source"] == source_filter]
            if doc_filter != "All":
                filtered = filtered[filtered["Document"] == doc_filter]

            st.markdown(f"Showing **{len(filtered)}** of **{len(findings_df)}** findings")

            # Display table (without Full Text column — use expander for details)
            display_cols = ["Document", "Page", "PII Type", "Snippet", "Confidence", "Source"]
            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Expandable detail section
            with st.expander("View full finding details"):
                if not filtered.empty:
                    for idx, row in filtered.iterrows():
                        st.markdown(
                            f"**{row['Document']}** p.{row['Page']} — "
                            f"`{row['PII Type']}` ({row['Source']}) "
                            f"confidence {row['Confidence']}"
                        )
                        st.code(row["Full Text"], language=None)
                        st.divider()
    else:
        st.info("Run a batch or select one from History to view findings.")

# ---- History tab ----------------------------------------------------------

with tab_history:
    db = _db_path()
    batches = load_batches(db)

    if not batches:
        st.info("No past batches found. Run a processing batch to populate history.")
    else:
        st.subheader("Past Batches")

        history_rows = []
        for b in batches:
            history_rows.append(
                {
                    "Batch ID": b.batch_id,
                    "Name": b.name,
                    "Date": b.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Documents": b.metrics.total_documents,
                    "Findings": b.metrics.total_findings,
                    "Clean Rate": f"{b.metrics.clean_doc_rate * 100:.1f}%",
                    "Risk Score": f"{b.metrics.weighted_risk_score:.3f}",
                }
            )

        history_df = pd.DataFrame(history_rows)

        # Selectable table
        selection = st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selection and selection.selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            selected_id = history_rows[selected_idx]["Batch ID"]
            if st.button("Load selected batch"):
                loaded = load_batch(db, selected_id)
                if loaded:
                    st.session_state.batch_result = loaded
                    st.session_state.selected_batch_id = loaded.batch_id
                    st.rerun()

        # Trend chart — risk score over time
        if len(batches) >= 2:
            st.divider()
            st.subheader("Risk Score Trend")
            trend_data = pd.DataFrame(
                {
                    "Date": [b.timestamp for b in reversed(batches)],
                    "Risk Score": [
                        b.metrics.weighted_risk_score for b in reversed(batches)
                    ],
                }
            ).set_index("Date")
            st.line_chart(trend_data)
