"""
pages/3_Alert_Panel.py  —  Real-time incident log and alert history
"""

import streamlit as st
import os, sys, json, glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.styles import inject_css, page_header, sidebar_branding, alert_card_html

st.set_page_config(page_title="Alert Panel · DriveGuard", layout="wide", page_icon="🚨")
inject_css()
sidebar_branding()
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

page_header("Alert Panel", "Timestamped log of all detected risky driving incidents")

# ── Load all sessions ──────────────────────────────────────────────────────────
session_files = sorted(glob.glob("outputs/session_*.json"), reverse=True)
all_incidents = []

for sf in session_files:
    with open(sf) as f:
        s = json.load(f)
    for inc in s.get("incidents", []):
        all_incidents.append({
            "Session": os.path.basename(sf).replace("session_","").replace(".json",""),
            "Driver":  s.get("driver","Unknown"),
            "Time":    inc.get("time","—"),
            "Event":   inc.get("label","—"),
            "Risk":    inc.get("risk", 0),
            "Confidence": inc.get("conf", 0),
        })

current_session = st.session_state.get("last_session", {})
current_incidents = current_session.get("incidents", [])

# ── Filters ────────────────────────────────────────────────────────────────────
f1, f2, f3 = st.columns(3)
with f1:
    event_filter = st.multiselect("Filter by event type",
                                  ["Distracted", "Phone Usage"],
                                  default=["Distracted","Phone Usage"])
with f2:
    risk_min = st.slider("Minimum risk score", 0, 100, 30)
with f3:
    sort_by = st.selectbox("Sort by", ["Time", "Risk (highest first)", "Event type"])

# ── Current session alerts ─────────────────────────────────────────────────────
st.markdown("### Current session alerts")

if current_incidents:
    filtered = [i for i in current_incidents
                if i["label"] in event_filter and i["risk"] >= risk_min]

    if sort_by == "Risk (highest first)":
        filtered.sort(key=lambda x: x["risk"], reverse=True)
    elif sort_by == "Event type":
        filtered.sort(key=lambda x: x["label"])

    if filtered:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total alerts",      len(current_incidents))
        a2.metric("Filtered alerts",   len(filtered))
        a3.metric("Highest risk",      f"{max(i['risk'] for i in current_incidents)}/100")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown("#### Alert timeline")
            for inc in filtered:
                st.markdown(
                    alert_card_html(inc["time"], inc["label"], inc["risk"], inc["conf"]),
                    unsafe_allow_html=True
                )
        with col_r:
            st.markdown("#### Severity breakdown")
            high   = sum(1 for i in filtered if i["risk"] >= 65)
            medium = sum(1 for i in filtered if 30 <= i["risk"] < 65)
            low    = sum(1 for i in filtered if i["risk"] < 30)

            for label, count, color in [
                ("High risk (≥65)",   high,   "#FF4757"),
                ("Moderate (30–64)",  medium, "#FFA502"),
                ("Low (<30)",         low,    "#00C896"),
            ]:
                st.markdown(f"""
                <div style="background:#131929;border-left:3px solid {color};
                            border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;
                            display:flex;justify-content:space-between;align-items:center;">
                  <span style="color:#6B7A99;font-size:0.85rem;">{label}</span>
                  <span style="font-family:'Share Tech Mono',monospace;color:{color};
                               font-size:1.2rem;">{count}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            phone_count = sum(1 for i in filtered if i["label"]=="Phone Usage")
            dist_count  = sum(1 for i in filtered if i["label"]=="Distracted")
            st.markdown(f"""
            <div style="background:#131929;border:1px solid rgba(0,200,150,0.18);
                        border-radius:12px;padding:1rem;">
              <div style="font-size:0.7rem;color:#6B7A99;letter-spacing:0.12em;
                          margin-bottom:0.8rem;">EVENT TYPE SPLIT</div>
              <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                <span style="color:#FFA502;font-size:0.85rem;">Distracted</span>
                <span style="font-family:'Share Tech Mono',monospace;color:#FFA502;">{dist_count}</span>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <span style="color:#FF4757;font-size:0.85rem;">Phone Usage</span>
                <span style="font-family:'Share Tech Mono',monospace;color:#FF4757;">{phone_count}</span>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No alerts match your current filter settings.")
else:
    st.markdown("""
    <div style="background:#131929;border:1px dashed rgba(0,200,150,0.25);border-radius:12px;
                padding:2rem;text-align:center;color:#6B7A99;">
      No alerts in current session. Run a detection session to see incidents here.
    </div>""", unsafe_allow_html=True)

# ── All sessions history ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### All sessions — alert history")

if all_incidents:
    df = pd.DataFrame(all_incidents)
    df_filtered = df[
        df["Event"].isin(event_filter) &
        (df["Risk"] >= risk_min)
    ].copy()
    df_filtered["Confidence"] = df_filtered["Confidence"].apply(lambda x: f"{x*100:.0f}%")

    if sort_by == "Risk (highest first)":
        df_filtered = df_filtered.sort_values("Risk", ascending=False)
    elif sort_by == "Event type":
        df_filtered = df_filtered.sort_values("Event")

    st.dataframe(
        df_filtered.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%d"
            )
        }
    )

    csv = df_filtered.to_csv(index=False).encode()
    st.download_button("⬇  Download alert log (CSV)", csv,
                       "alert_log.csv", "text/csv")
else:
    st.markdown("""
    <div style="background:#131929;border:1px dashed rgba(0,200,150,0.25);border-radius:12px;
                padding:2rem;text-align:center;color:#6B7A99;">
      No session files found in <code>outputs/</code>. Run detection first.
    </div>""", unsafe_allow_html=True)
