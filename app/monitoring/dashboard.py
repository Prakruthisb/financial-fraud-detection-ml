from pathlib import Path
import pandas as pd
import os

from app.monitoring.logger import load_predictions
from app.monitoring.drift import run_drift_report,prepare_current_window
from app.monitoring.performance import run_performance_report
from app.monitoring.alerts import check_alerts
 
 
# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = os.getenv("DATABASE_URL") 
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)
 
# Alert thresholds — tune these to your business tolerance
ALERT_THRESHOLDS = {
    "recall_min"         : 0.85,   # alert if recall drops below this
    "precision_min"      : 0.30,   # alert if precision drops below this
    "fraud_rate_max"     : 0.05,   # alert if live fraud rate exceeds 5%
    "drift_share_max"    : 0.30,   # alert if >30% of features are drifting
    "missing_values_max" : 0.01,   # alert if >1% of values are missing
}

# =============================================================================
# STREAMLIT MONITORING TAB
# Add this to your existing app.py:
#
#   from fraud_monitoring import render_monitoring_tab
#   tab1, tab2 = st.tabs(["Predict", "Monitoring"])
#   with tab2:
#       render_monitoring_tab()
# =============================================================================
 
def render_monitoring_tab():
    """
    Drop-in Streamlit monitoring dashboard.
    Paste into your app.py as a second tab.
    """
    import streamlit as st
 
    st.markdown("## 📊 Model Monitoring")
    st.markdown(
        "<p style='color:#6b7280;margin-top:-12px;font-size:0.9rem;'>"
        "Drift detection · Performance tracking · Data quality</p>",
        unsafe_allow_html=True,
    )
 
    col_days, col_thresh, col_run = st.columns([1, 1, 2])
    with col_days:
        days = st.selectbox("Window", [1, 3, 7, 14, 30], index=2,
                            format_func=lambda x: f"Last {x} days")
    with col_thresh:
        threshold = st.number_input("Threshold", 0.1, 0.99, 0.9, 0.01)
    with col_run:
        run_btn = st.button("🔄  Run Monitoring Reports", type="primary",
                            use_container_width=True)
 
    # Live stats from DB
    st.divider()
    pred_df = load_predictions(days=days)
 
    if pred_df.empty:
        st.info("No predictions logged yet. Make some predictions first.")
        return
 
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, len(pred_df),                              "Predictions",        ""),
        (c2, f"{pred_df['predicted_fraud'].mean()*100:.2f}%", "Flagged as fraud", ""),
        (c3, f"{pred_df['amount'].mean():,.0f}",        "Avg amount (₹)",     ""),
        (c4, pred_df['actual_fraud'].notna().sum(),     "Labelled",           ""),
    ]
    for col, val, label, _ in metrics:
        col.metric(label, val)
 
    # Fraud rate over time
    st.markdown("#### Fraud flag rate over time")
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df["date"]      = pred_df["timestamp"].dt.date
    daily = pred_df.groupby("date").agg(
        total=("id", "count"),
        flagged=("predicted_fraud", "sum"),
    ).reset_index()
    daily["fraud_rate"] = daily["flagged"] / daily["total"]
 
    st.line_chart(daily.set_index("date")["fraud_rate"],
                  use_container_width=True)
 
    # Run Evidently reports
    if run_btn:
        if not Path(REFERENCE_PATH).exists():
            st.error(
                "Reference data not found. "
                "Run `build_reference_data()` first (see fraud_monitoring.py)."
            )
            return
 
        with st.spinner("Running drift analysis..."):
            try:
                current_df   = prepare_current_window(days=days,
                                                       threshold=threshold)
                drift_metrics = run_drift_report(current_df)
 
                perf_metrics = None
                if "target" in current_df.columns and \
                   current_df["target"].notna().sum() > 50:
                    perf_metrics = run_performance_report(current_df)
 
                alerts = check_alerts(drift_metrics, perf_metrics)
 
            except Exception as e:
                st.error(f"Monitoring error: {e}")
                return
 
        # Alerts
        st.markdown("#### Alerts")
        for alert in alerts:
            if "🚨" in alert:
                st.error(alert)
            elif "⚠️" in alert:
                st.warning(alert)
            else:
                st.success(alert)
 
        # Drift metrics
        st.markdown("#### Drift summary")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Drifted features",
                   f"{drift_metrics.get('drifted_features', 0)} / "
                   f"{drift_metrics.get('total_features', 0)}")
        dc2.metric("Drift share",
                   f"{drift_metrics.get('drift_share', 0)*100:.1f}%")
        dc3.metric("Missing values",
                   f"{drift_metrics.get('missing_share', 0)*100:.3f}%")
 
        # Performance metrics (if labels available)
        if perf_metrics:
            st.markdown("#### Model performance")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Recall",    f"{perf_metrics.get('recall', 0):.2%}")
            pc2.metric("Precision", f"{perf_metrics.get('precision', 0):.2%}")
            pc3.metric("F1",        f"{perf_metrics.get('f1', 0):.2%}")
 
        # Report links
        st.markdown("#### Full reports")
        drift_path = drift_metrics.get("report_path", "")
        if drift_path and Path(drift_path).exists():
            with open(drift_path, "rb") as f:
                st.download_button(
                    "⬇ Download drift report (HTML)",
                    f, file_name=Path(drift_path).name,
                    mime="text/html"
                )
 
        if perf_metrics:
            perf_path = perf_metrics.get("report_path", "")
            if perf_path and Path(perf_path).exists():
                with open(perf_path, "rb") as f:
                    st.download_button(
                        "⬇ Download performance report (HTML)",
                        f, file_name=Path(perf_path).name,
                        mime="text/html"
                    )