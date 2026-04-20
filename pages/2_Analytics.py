"""
pages/2_Analytics.py  —  Session analytics dashboard with charts
"""

import streamlit as st
import numpy as np
import pandas as pd
import os, sys, json, glob
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.styles import inject_css, page_header, sidebar_branding, info_card

st.set_page_config(page_title="Analytics · DriveGuard", layout="wide", page_icon="📊")
inject_css()
sidebar_branding()
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

page_header("Analytics Dashboard", "Visualise behavior trends, risk scores, and session insights")

# ── Load session ───────────────────────────────────────────────────────────────
session = st.session_state.get("last_session", None)

session_files = sorted(glob.glob("outputs/session_*.json"), reverse=True)
if not session and session_files:
    with open(session_files[0]) as f:
        session = json.load(f)

if session_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Load session")
    names = [os.path.basename(f) for f in session_files]
    choice = st.sidebar.selectbox("Session file", names)
    if st.sidebar.button("Load selected"):
        with open(f"outputs/{choice}") as f:
            session = json.load(f)
        st.session_state["last_session"] = session

if session is None:
    st.markdown("""
    <div style="background:#ffffff;border:1px dashed #cbd5e1;border-radius:14px;
                padding:3rem;text-align:center;">
      <div style="font-size:0.8rem;font-weight:600;color:#2563eb;margin-bottom:0.8rem;">
        NO SESSION DATA
      </div>
      <p style="color:#64748b;font-size:0.9rem;margin:0;">
        Run a detection session first on the Live Detection page,
        or place a <code>session_*.json</code> file in the <code>outputs/</code> folder.
      </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("▶  Go to Live Detection"):
        st.switch_page("pages/1_Live_Detection.py")
    st.stop()

# ── Unpack session data ────────────────────────────────────────────────────────
risk_log   = session.get("risk_log", [])
label_log  = session.get("label_log", [])
incidents  = session.get("incidents", [])
total_seq  = session.get("total_sequences", len(label_log))
max_risk   = session.get("max_risk", max(risk_log) if risk_log else 0)
avg_risk   = session.get("avg_risk", int(np.mean(risk_log)) if risk_log else 0)
driver     = session.get("driver", "Unknown")
date_str   = session.get("date", "")

incident_count = len(incidents)

# ── Session header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
            padding:1rem 1.4rem;margin-bottom:1.5rem;display:flex;
            justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:0.7rem;font-weight:600;color:#2563eb;letter-spacing:0.1em;">
      SESSION REPORT
    </div>
    <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-top:2px;">{driver}</div>
  </div>
  <div style="text-align:right;font-size:0.8rem;color:#64748b;">{date_str}</div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total sequences", total_seq)
m2.metric("Incidents",       incident_count)
m3.metric("Max risk",        f"{max_risk}/100")
m4.metric("Avg risk",        f"{avg_risk}/100")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    st.markdown("#### Risk score timeline")
    if risk_log:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=risk_log, mode="lines",
                line=dict(color="#2563eb", width=1.5),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
                name="Risk score"
            ))
            fig.add_hline(y=65, line_dash="dash", line_color="#dc2626",
                          annotation_text="High risk threshold", annotation_font_color="#dc2626")
            fig.add_hline(y=30, line_dash="dash", line_color="#d97706",
                          annotation_text="Moderate threshold", annotation_font_color="#d97706")
            fig.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#64748b", family="Inter"),
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="#e2e8f0", showgrid=True,
                           title="Sequence index", title_font_color="#64748b"),
                yaxis=dict(gridcolor="#e2e8f0", showgrid=True,
                           range=[0, 105], title="Risk score", title_font_color="#64748b"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(range(len(risk_log)), risk_log, alpha=0.15, color="#2563eb")
            ax.plot(risk_log, color="#2563eb", lw=1.5)
            ax.axhline(65, color="#dc2626", linestyle="--", lw=1)
            ax.axhline(30, color="#d97706", linestyle="--", lw=1)
            ax.set_facecolor("#f8fafc")
            fig.patch.set_facecolor("#ffffff")
            ax.tick_params(colors="#64748b")
            ax.set_ylim(0, 105)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e2e8f0")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No risk data in this session.")

# ── Safety score card ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Driver safety score")
safety_score = max(0, 100 - avg_risk)
grade = "A" if safety_score >= 90 else ("B" if safety_score >= 75 else ("C" if safety_score >= 60 else ("D" if safety_score >= 45 else "F")))
grade_color = {"A": "#16a34a", "B": "#16a34a", "C": "#d97706", "D": "#ea580c", "F": "#dc2626"}[grade]

gc1, gc2, gc3 = st.columns([1, 1, 3])
with gc1:
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:1.5rem;text-align:center;">
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;margin-bottom:0.5rem;">
        SAFETY GRADE
      </div>
      <div style="font-size:3.5rem;font-weight:700;color:{grade_color};">{grade}</div>
    </div>""", unsafe_allow_html=True)

with gc2:
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:1.5rem;text-align:center;">
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;margin-bottom:0.5rem;">
        SAFETY SCORE
      </div>
      <div style="font-size:3rem;font-weight:700;color:{grade_color};">{safety_score}</div>
      <div style="font-size:0.75rem;color:#94a3b8;">out of 100</div>
    </div>""", unsafe_allow_html=True)

with gc3:
    tips = {
        "A": "Excellent driving! Maintain your focus and keep up the great work.",
        "B": "Good driving. Watch for occasional distractions and stay consistent.",
        "C": "Moderate risk detected. Avoid phone usage and stay focused on the road.",
        "D": "High distraction rate. Consider taking a break and avoiding phone use entirely.",
        "F": "Dangerous driving detected. Immediately stop using your phone while driving.",
    }
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:1.5rem;">
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;margin-bottom:0.8rem;">
        RECOMMENDATION
      </div>
      <p style="color:#1e293b;font-size:0.9rem;line-height:1.7;margin:0;">{tips[grade]}</p>
      <div style="margin-top:1rem;font-size:0.8rem;color:#64748b;">
        Incidents: <strong style="color:#d97706;">{len(incidents)}</strong>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Incident table ─────────────────────────────────────────────────────────────
if incidents:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Incident log")
    df = pd.DataFrame(incidents)
    df.columns = [c.title() for c in df.columns]
    df["Conf"] = df["Conf"].apply(lambda x: f"{x*100:.0f}%")
    st.dataframe(df, use_container_width=True, hide_index=True)