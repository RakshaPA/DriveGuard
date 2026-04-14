"""
pages/5_Report.py  —  Generate and download session report
"""

import streamlit as st
import os, sys, json, glob
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.styles import inject_css, page_header, sidebar_branding

st.set_page_config(page_title="Report · DriveGuard", layout="wide", page_icon="📄")
inject_css()
sidebar_branding()
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

page_header("Generate Report", "Download session summary as CSV, JSON, or HTML report")

# ── Load session ───────────────────────────────────────────────────────────────
session = st.session_state.get("last_session", None)
session_files = sorted(glob.glob("outputs/session_*.json"), reverse=True)

if not session and session_files:
    with open(session_files[0]) as f:
        session = json.load(f)

if session_files:
    st.sidebar.markdown("---")
    names = [os.path.basename(f) for f in session_files]
    choice = st.sidebar.selectbox("Session", names)
    if st.sidebar.button("Load"):
        with open(f"outputs/{choice}") as f:
            session = json.load(f)
        st.session_state["last_session"] = session

if session is None:
    st.warning("No session data found. Run Live Detection first.")
    if st.button("▶  Go to Live Detection"):
        st.switch_page("pages/1_Live_Detection.py")
    st.stop()

# ── Session details ────────────────────────────────────────────────────────────
risk_log     = session.get("risk_log", [])
label_log    = session.get("label_log", [])
incidents    = session.get("incidents", [])
driver       = session.get("driver", "Unknown")
date_str     = session.get("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
total_seq    = session.get("total_sequences", len(label_log))
n_normal     = session.get("normal", label_log.count("Normal"))
n_dist       = session.get("distracted", label_log.count("Distracted"))
n_phone      = session.get("phone_usage", label_log.count("Phone Usage"))
max_risk     = session.get("max_risk", max(risk_log) if risk_log else 0)
avg_risk     = session.get("avg_risk", int(np.mean(risk_log)) if risk_log else 0)
safe_pct     = round(n_normal / max(total_seq, 1) * 100, 1)
safety_score = max(0, 100 - avg_risk)
grade        = "A" if safety_score >= 90 else ("B" if safety_score >= 75 else ("C" if safety_score >= 60 else ("D" if safety_score >= 45 else "F")))
grade_color  = "#16a34a" if grade in ("A", "B") else ("#d97706" if grade == "C" else "#dc2626")

# ── Build incident rows safely (no backslashes inside f-strings) ───────────────
incident_rows_html = ""
for i in incidents[:10]:
    inc_color = "#dc2626" if i["label"] == "Phone Usage" else "#d97706"
    incident_rows_html += (
        '<div style="display:flex;justify-content:space-between;'
        'border-bottom:1px solid #e2e8f0;padding:6px 0;font-size:0.82rem;">'
        f'<span style="color:#64748b;">{i["time"]}</span>'
        f'<span style="font-weight:600;color:{inc_color};">{i["label"].upper()}</span>'
        f'<span style="color:#1e293b;">{i["risk"]}/100</span>'
        '</div>'
    )

overflow_note = ""
if len(incidents) > 10:
    overflow_note = f'<div style="color:#94a3b8;font-size:0.75rem;margin-top:4px;">... and {len(incidents)-10} more</div>'

incident_section = ""
if incidents:
    incident_section = (
        '<div style="margin-top:1.5rem;border-top:1px solid #e2e8f0;padding-top:1.2rem;">'
        f'<div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;margin-bottom:0.8rem;">INCIDENT LOG ({len(incidents)} events)</div>'
        + incident_rows_html
        + overflow_note
        + '</div>'
    )

# ── Summary grid rows ──────────────────────────────────────────────────────────
summary_top = ""
for lbl, val, col in [
    ("SEQUENCES",  total_seq,         "#0f172a"),
    ("SAFE %",     f"{safe_pct}%",    "#16a34a"),
    ("INCIDENTS",  len(incidents),    "#d97706"),
    ("MAX RISK",   f"{max_risk}/100", "#dc2626"),
]:
    summary_top += (
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
        'padding:0.8rem;text-align:center;">'
        f'<div style="font-size:0.65rem;color:#64748b;letter-spacing:0.1em;">{lbl}</div>'
        f'<div style="font-size:1.3rem;font-weight:700;color:{col};margin-top:4px;">{val}</div>'
        '</div>'
    )

summary_bottom = ""
for lbl, val, col in [
    ("NORMAL EVENTS",  n_normal,          "#16a34a"),
    ("DISTRACTED",     n_dist,            "#d97706"),
    ("PHONE USAGE",    n_phone,           "#dc2626"),
    ("AVG RISK",       f"{avg_risk}/100", "#0f172a"),
    ("SAFETY SCORE",   f"{safety_score}/100", "#16a34a"),
    ("SAFETY GRADE",   grade,             grade_color),
]:
    summary_bottom += (
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:0.8rem;">'
        f'<div style="font-size:0.65rem;color:#64748b;letter-spacing:0.1em;margin-bottom:4px;">{lbl}</div>'
        f'<div style="font-size:1rem;font-weight:700;color:{col};">{val}</div>'
        '</div>'
    )

# ── Report preview ─────────────────────────────────────────────────────────────
st.markdown("### Report preview")
st.markdown(
    '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:2rem;">'

    '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
    'border-bottom:1px solid #e2e8f0;padding-bottom:1.2rem;margin-bottom:1.2rem;">'
    '<div>'
    '<div style="font-size:0.7rem;font-weight:600;color:#2563eb;letter-spacing:0.15em;">DRIVEGUARD · SESSION REPORT</div>'
    f'<div style="font-size:1.4rem;font-weight:700;color:#0f172a;margin-top:4px;">{driver}</div>'
    f'<div style="font-size:0.78rem;color:#64748b;margin-top:2px;">{date_str}</div>'
    '</div>'
    f'<div style="text-align:right;font-size:2.5rem;font-weight:700;color:{grade_color};">{grade}'
    '<div style="font-size:0.7rem;color:#64748b;font-weight:400;">SAFETY GRADE</div></div>'
    '</div>'

    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1rem;">{summary_top}</div>'
    f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">{summary_bottom}</div>'
    + incident_section +
    '</div>',
    unsafe_allow_html=True
)

# ── Download buttons ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Download options")
dl1, dl2 = st.columns(2)

with dl1:
    csv_data = [{"sequence": i, "label": lbl, "risk": risk}
                for i, (lbl, risk) in enumerate(zip(label_log, risk_log))]
    df = pd.DataFrame(csv_data)
    st.download_button(
        "⬇  Download CSV",
        df.to_csv(index=False).encode(),
        f"report_{driver.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )

with dl2:
    st.download_button(
        "⬇  Download JSON",
        json.dumps(session, indent=2).encode(),
        f"session_{driver.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
        "application/json",
        use_container_width=True
    )



st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
            padding:1rem 1.2rem;font-size:0.82rem;color:#64748b;">
  <span style="font-weight:600;color:#2563eb;">NOTE</span>
  &nbsp; The HTML report opens in any browser and can be printed as a PDF using
  <strong style="color:#1e293b;">Ctrl+P → Save as PDF</strong> — no extra software needed.
</div>
""", unsafe_allow_html=True)