import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.styles import inject_css, page_header, sidebar_branding, status_badge

st.set_page_config(
    page_title="DriveGuard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
sidebar_branding()

st.sidebar.markdown("### Navigation")
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

page_header("DriveGuard", "Driver behavior monitoring system")

# ── Status row ─────────────────────────────────────────────────────────────────
model_ok = os.path.exists("models/cnn_lstm_best.h5")
data_ok  = os.path.exists("data/sequences/X_train.npy")
vid_ok   = os.path.exists("data/raw_videos")

s1, s2, s3, s4 = st.columns(4)
with s1: status_badge("Model loaded"  if model_ok else "Model not found", "normal"  if model_ok else "danger")
with s2: status_badge("Dataset ready" if data_ok  else "Training data", "normal"  if data_ok  else "warning")
with s3: status_badge("Videos folder" if vid_ok   else "No video folder", "normal"  if vid_ok   else "warning")
with s4: status_badge("System online", "normal")

st.markdown("<br>", unsafe_allow_html=True)

# ── Detection categories ────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size:0.78rem;font-weight:600;color:#94a3b8;letter-spacing:0.07em;
           text-transform:uppercase;margin-bottom:0.8rem;">Detection Categories</p>
""", unsafe_allow_html=True)

categories = [
    ("✅", "Normal Driving",     "#16a34a", "#f0fdf4", "#bbf7d0",
     "Risk: 0 – 30", "Driver is attentive, eyes on the road, hands on the wheel."),
    ("⚠️", "Distracted Driving", "#b45309", "#fffbeb", "#fde68a",
     "Risk: 30 – 75", "Attention is diverted — head movement or loss of road focus detected."),
    ("🚨", "Phone Usage",        "#dc2626", "#fef2f2", "#fecaca",
     "Risk: 75 – 100", "Mobile phone detected in active use while the vehicle is in motion."),
]

c1, c2, c3 = st.columns(3)
for col, (icon, title, color, bg, border, risk_range, desc) in zip([c1, c2, c3], categories):
    with col:
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:12px;
                    padding:1.4rem 1.5rem;">
          <div style="font-size:1.5rem;margin-bottom:0.75rem;">{icon}</div>
          <div style="font-size:0.93rem;font-weight:600;color:#0f172a;
                      margin-bottom:0.2rem;">{title}</div>
          <div style="font-size:0.74rem;font-weight:600;color:{color};
                      margin-bottom:0.6rem;">{risk_range}</div>
          <p style="font-size:0.81rem;color:#475569;line-height:1.6;margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick actions ───────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size:0.78rem;font-weight:600;color:#94a3b8;letter-spacing:0.07em;
           text-transform:uppercase;margin-bottom:0.8rem;">Quick Actions</p>
""", unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("▶  Start Detection", use_container_width=True):
        st.switch_page("pages/1_Live_Detection.py")
with b2:
    if st.button("📊  Analytics", use_container_width=True):
        st.switch_page("pages/2_Analytics.py")
with b3:
    if st.button("🚨  Alert Panel", use_container_width=True):
        st.switch_page("pages/3_Alert_Panel.py")
with b4:
    if st.button("📄  Generate Report", use_container_width=True):
        st.switch_page("pages/5_Report.py")