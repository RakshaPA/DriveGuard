import streamlit as st

MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Force light background everywhere ── */
html, body,
[class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, .main > div,
section.main,
div.stApp {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.2rem 2.8rem 4rem !important;
    max-width: 1180px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] * {
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.5rem 1rem !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    color: #ffffff !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
    border-radius: 7px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #ffffff !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Progress ── */
.stProgress > div > div { background: #2563eb !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; margin: 1.2rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
"""


def inject_css():
    st.markdown(MASTER_CSS, unsafe_allow_html=True)


def sidebar_branding():
    st.sidebar.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:0.9rem;">
        <div style="width:34px;height:34px;background:#eff6ff;border:1px solid #bfdbfe;
                    border-radius:8px;display:flex;align-items:center;
                    justify-content:center;font-size:1.1rem;">🚗</div>
        <div>
          <div style="font-size:0.95rem;font-weight:600;color:#0f172a;">DriveGuard</div>
          <div style="font-size:0.65rem;color:#94a3b8;letter-spacing:0.03em;">Monitoring System</div>
        </div>
      </div>
      <div style="height:1px;background:#e2e8f0;margin-bottom:0.8rem;"></div>
      <div style="font-size:0.76rem;color:#64748b;line-height:2.1;">
        <div><span style="color:#334155;font-weight:500;">Student</span>&nbsp;&nbsp;Raksha P A</div>
        <div><span style="color:#334155;font-weight:500;">Roll No</span>&nbsp;&nbsp;&nbsp;23PT26</div>
        <div><span style="color:#334155;font-weight:500;">Model</span>&nbsp;&nbsp;&nbsp;&nbsp;MobileNetV2 + LSTM</div>
      </div>
      <div style="height:1px;background:#e2e8f0;margin-top:0.8rem;"></div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:1.8rem;">
      <h1 style="font-family:'Inter',sans-serif;font-size:1.6rem;font-weight:700;
                 color:#0f172a;margin:0 0 0.2rem;letter-spacing:-0.02em;">{title}</h1>
      {"<p style='color:#64748b;font-size:0.87rem;margin:0;font-weight:400;'>" + subtitle + "</p>" if subtitle else ""}
      <div style="height:1px;background:#e2e8f0;margin-top:1rem;"></div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(label: str, level: str = "normal"):
    styles = {
        "normal":  ("#16a34a", "#f0fdf4", "#bbf7d0"),
        "warning": ("#b45309", "#fffbeb", "#fde68a"),
        "danger":  ("#dc2626", "#fef2f2", "#fecaca"),
    }
    fg, bg, border = styles.get(level, styles["normal"])
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:7px;background:{bg};
                border:1px solid {border};border-radius:6px;padding:5px 13px;">
      <div style="width:6px;height:6px;border-radius:50%;background:{fg};"></div>
      <span style="font-size:0.78rem;font-weight:500;color:{fg};">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def risk_gauge_html(risk: int) -> str:
    color = "#16a34a" if risk < 30 else "#d97706" if risk < 65 else "#dc2626"
    label = "Low Risk" if risk < 30 else "Moderate" if risk < 65 else "High Risk"
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:1.2rem 1.4rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
      <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.4rem;">Risk Score</div>
      <div style="font-size:2rem;font-weight:700;color:{color};line-height:1;">
        {risk}<span style="font-size:0.85rem;color:#94a3b8;font-weight:400;">/100</span>
      </div>
      <div style="margin-top:0.7rem;height:4px;background:#f1f5f9;border-radius:2px;">
        <div style="width:{min(risk,100)}%;height:4px;background:{color};border-radius:2px;"></div>
      </div>
      <div style="font-size:0.7rem;font-weight:600;color:{color};margin-top:0.4rem;
                  letter-spacing:0.05em;">{label.upper()}</div>
    </div>
    """


def alert_card_html(timestamp: str, label: str, risk: int, conf: float) -> str:
    color  = "#dc2626" if label == "Phone Usage" else "#d97706"
    border = "#fecaca" if label == "Phone Usage" else "#fde68a"
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:3px solid {color};
                border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin-bottom:0.5rem;
                display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:0.83rem;font-weight:600;color:{color};">{label}</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">{timestamp}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:0.92rem;font-weight:600;color:{color};">{risk}/100</div>
        <div style="font-size:0.7rem;color:#94a3b8;">{conf*100:.0f}% conf</div>
      </div>
    </div>
    """


def info_card(title: str, value: str, color: str = "#2563eb"):
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:1rem 1.2rem;text-align:center;">
      <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:0.08em;">{title}</div>
      <div style="font-size:1.5rem;font-weight:700;color:{color};margin-top:0.3rem;">{value}</div>
    </div>
    """