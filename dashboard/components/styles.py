import streamlit as st

def inject_global_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Fond général ── */
.stApp { background: #f4f6f3 !important; }
section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8e4 !important; }

/* ── Sidebar nav ── */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px;
    font-size: 13px; font-weight: 500; color: #555;
    cursor: pointer; margin-bottom: 2px;
    transition: background .15s, color .15s;
    text-decoration: none;
}
.nav-item:hover { background: #f0f7f3; color: #1a5c38; }
.nav-item.active { background: #e8f5ee; color: #1a5c38; font-weight: 600; }
.nav-icon { font-size: 16px; width: 20px; text-align: center; }

/* ── KPI cards ── */
.kpi-featured {
    background: #1a5c38; border-radius: 16px;
    padding: 18px 20px; color: white; height: 110px;
    display: flex; flex-direction: column; justify-content: space-between;
}
.kpi-featured .kpi-val { font-size: 28px; font-weight: 600; letter-spacing: -0.5px; }
.kpi-featured .kpi-lbl { font-size: 11px; opacity: .75; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
.kpi-featured .kpi-sub { font-size: 11px; opacity: .65; }

.kpi-card {
    background: #ffffff; border-radius: 16px;
    padding: 18px 20px; height: 110px; border: 1px solid #e8ede9;
    display: flex; flex-direction: column; justify-content: space-between;
}
.kpi-card .kpi-val { font-size: 26px; font-weight: 600; color: #111; letter-spacing: -0.5px; }
.kpi-card .kpi-lbl { font-size: 11px; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
.kpi-card .kpi-sub { font-size: 11px; color: #aaa; }

/* ── Section cards ── */
.section-card {
    background: #ffffff; border-radius: 16px;
    padding: 20px 22px; border: 1px solid #e8ede9;
    margin-bottom: 0;
}
.section-card .card-title {
    font-size: 13px; font-weight: 600; color: #1a1a1a;
    margin-bottom: 14px; letter-spacing: -.01em;
}

/* ── Track rows ── */
.track-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #f2f5f3;
    font-size: 12px;
}
.track-row:last-child { border-bottom: none; }
.track-num { color: #bbb; width: 16px; text-align: center; font-size: 11px; }
.track-name { font-weight: 500; color: #1a1a1a; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.track-album { font-size: 10px; color: #aaa; }
.track-streams { font-size: 11px; color: #555; font-family: 'DM Mono', monospace; flex-shrink: 0; }

/* ── Emotion pills ── */
.emo-joie      { background:#e8f5ee; color:#1a5c38; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.emo-tristesse { background:#e6f1fb; color:#185fa5; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.emo-colere    { background:#fcebeb; color:#a32d2d; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.emo-peur      { background:#eeedfe; color:#534ab7; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.emo-surprise  { background:#faeeda; color:#854f0b; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.emo-degout    { background:#e1f5ee; color:#0f6e56; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; display:inline-block; }

/* ── Timeline ── */
.tl-item { display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; }
.tl-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
.tl-title { font-size: 12px; font-weight: 500; color: #1a1a1a; }
.tl-sub   { font-size: 11px; color: #aaa; }

/* ── Status tags ── */
.tag-pos  { background:#e8f5ee; color:#1a5c38; font-size:10px; padding:3px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.tag-warn { background:#faeeda; color:#854f0b; font-size:10px; padding:3px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.tag-neu  { background:#f2f2f2; color:#666; font-size:10px; padding:3px 9px; border-radius:20px; font-weight:600; display:inline-block; }
.tag-neg  { background:#fcebeb; color:#a32d2d; font-size:10px; padding:3px 9px; border-radius:20px; font-weight:600; display:inline-block; }

/* ── Misc ── */
.divider { border: none; border-top: 1px solid #eef1ee; margin: 14px 0; }
.mono { font-family: 'DM Mono', monospace; }

/* ── Streamlit overrides ── */
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 600 !important; }
.stSelectbox > div > div { border-radius: 10px !important; }
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
div[data-testid="stHorizontalBlock"] { gap: 12px !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1200px !important; }
h1 { font-size: 22px !important; font-weight: 600 !important; color: #1a1a1a !important; }
h2 { font-size: 17px !important; font-weight: 600 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)
