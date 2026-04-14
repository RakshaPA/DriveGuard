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
n_normal   = session.get("normal", label_log.count("Normal"))
n_dist     = session.get("distracted", label_log.count("Distracted"))
n_phone    = session.get("phone_usage", label_log.count("Phone Usage"))
max_risk   = session.get("max_risk", max(risk_log) if risk_log else 0)
avg_risk   = session.get("avg_risk", int(np.mean(risk_log)) if risk_log else 0)
driver     = session.get("driver", "Unknown")
date_str   = session.get("date", "")

safe_pct   = round(n_normal / max(total_seq, 1) * 100, 1)
risky_pct  = round(100 - safe_pct, 1)

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
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total sequences", total_seq)
m2.metric("Normal events",   n_normal,  delta=f"{safe_pct}%")
m3.metric("Distractions",    n_dist,    delta=f"-{round(n_dist/max(total_seq,1)*100,1)}%",  delta_color="inverse")
m4.metric("Phone usage",     n_phone,   delta=f"-{round(n_phone/max(total_seq,1)*100,1)}%", delta_color="inverse")
m5.metric("Max risk",        f"{max_risk}/100")
m6.metric("Avg risk",        f"{avg_risk}/100")

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

with right:
    st.markdown("#### Behavior split")
    pie_values = [n_normal, n_dist, n_phone]
    if HAS_PLOTLY:
        if sum(pie_values) > 0:
            fig2 = go.Figure(go.Pie(
                labels=["Normal", "Distracted", "Phone Usage"],
                values=pie_values,
                hole=0.55,
                marker_colors=["#16a34a", "#d97706", "#dc2626"],
                textfont_color="#ffffff",
            ))
            fig2.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                font=dict(color="#64748b", family="Inter"),
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(color="#64748b"), bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(text=f"{safe_pct}%<br>safe",
                                  font=dict(size=13, color="#16a34a", family="Inter"),
                                  showarrow=False)]
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No labeled data to display.")
    else:
        if sum(pie_values) > 0:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie(
                pie_values,
                labels=["Normal", "Distracted", "Phone"],
                colors=["#16a34a", "#d97706", "#dc2626"],
                textprops={"color": "#1e293b", "fontsize": 8}
            )
            fig2.patch.set_facecolor("#ffffff")
            ax2.set_facecolor("#ffffff")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No labeled data to display.")

# ── Charts row 2 ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
l2, r2 = st.columns(2)

with l2:
    st.markdown("#### Behavior over time")
    if label_log and HAS_PLOTLY:
        label_num = {"Normal": 0, "Distracted": 1, "Phone Usage": 2}
        nums  = [label_num.get(l, 0) for l in label_log]
        color = ["#16a34a" if l == "Normal" else ("#d97706" if l == "Distracted" else "#dc2626")
                 for l in label_log]
        fig3 = go.Figure(go.Scatter(
            x=list(range(len(nums))), y=nums, mode="markers",
            marker=dict(color=color, size=4),
        ))
        fig3.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", family="Inter"),
            height=220, margin=dict(l=20, r=10, t=10, b=30),
            xaxis=dict(gridcolor="#e2e8f0", title="Sequence"),
            yaxis=dict(gridcolor="#e2e8f0", tickvals=[0, 1, 2],
                       ticktext=["Normal", "Distracted", "Phone"], title=""),
        )
        st.plotly_chart(fig3, use_container_width=True)

with r2:
    st.markdown("#### Risk distribution")
    if risk_log and HAS_PLOTLY:
        fig4 = go.Figure(go.Histogram(
            x=risk_log, nbinsx=20,
            marker_color="#2563eb", opacity=0.75,
            marker_line_color="#ffffff", marker_line_width=1,
        ))
        fig4.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", family="Inter"),
            height=220, margin=dict(l=20, r=10, t=10, b=30),
            xaxis=dict(gridcolor="#e2e8f0", title="Risk score"),
            yaxis=dict(gridcolor="#e2e8f0", title="Frequency"),
        )
        st.plotly_chart(fig4, use_container_width=True)

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
        Safe driving: <strong style="color:#16a34a;">{safe_pct}%</strong> &nbsp;·&nbsp;
        Risky events: <strong style="color:#dc2626;">{risky_pct}%</strong> &nbsp;·&nbsp;
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