"""Colorful 3D glass theme for the churn demo.

Self-contained: no CDN, no WebGL dependency, nothing to download. Uses pure CSS
animated gradients + floating orbs so it always renders on any machine.
"""
import streamlit as st

_CSS = """
<style>
  :root {
    --bg1: #0f0c29;
    --bg2: #302b63;
    --bg3: #24243e;
    --panel: rgba(255,255,255,0.08);
    --panel-border: rgba(255,255,255,0.16);
    --text: #ffffff;
    --muted: #b8c0d4;
    --cyan: #22d3ee;
    --pink: #f472b6;
    --violet: #a78bfa;
    --lime: #a3e635;
  }

  /* animated multi-color aurora background */
  .stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #7c3aed, #0ea5e9, #ec4899, #0f0c29);
    background-size: 400% 400%;
    animation: aurora 18s ease infinite;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    color: var(--text);
  }

  @keyframes aurora {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* floating 3D-feel orbs (blurred glowing spheres) */
  .orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.55;
    z-index: 0;
    pointer-events: none;
    animation: float 14s ease-in-out infinite;
  }
  .orb-1 { width: 420px; height: 420px; top: -120px; left: -100px;
           background: radial-gradient(circle, #22d3ee 0%, transparent 70%); }
  .orb-2 { width: 380px; height: 380px; bottom: -110px; right: -90px;
           background: radial-gradient(circle, #f472b6 0%, transparent 70%);
           animation-delay: -5s; }
  .orb-3 { width: 300px; height: 300px; top: 45%; left: 55%;
           background: radial-gradient(circle, #a78bfa 0%, transparent 70%);
           animation-delay: -9s; }

  @keyframes float {
    0%   { transform: translateY(0) scale(1); }
    50%  { transform: translateY(-45px) scale(1.12); }
    100% { transform: translateY(0) scale(1); }
  }

  h1, h2, h3, h4 {
    font-family: 'Segoe UI', -apple-system, sans-serif;
    color: var(--text);
    letter-spacing: -0.01em;
  }

  /* glass cards */
  [data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 16px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  }
  [data-testid="stMetricValue"] {
    color: var(--text);
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 2rem;
  }
  [data-testid="stMetricDelta"] { color: var(--cyan); font-weight: 600; }

  [data-testid="stExpander"], [data-testid="stAlert"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    backdrop-filter: blur(8px);
  }

  /* sidebars & widgets readable on color */
  [data-testid="stSidebar"] {
    background: rgba(10,8,30,0.55);
    border-right: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(6px);
  }

  .stSelectbox [data-baseweb="select"] > div,
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--cyan); }
  .stSlider [data-baseweb="slider"] > div > div { background: linear-gradient(90deg, var(--cyan), var(--violet)); }

  .stCheckbox [data-testid="stCheckbox"] [role="checkbox"]:checked,
  .stToggle [data-testid="stCheckbox"] [role="checkbox"]:checked {
    background: linear-gradient(135deg, var(--cyan), var(--violet)) !important;
    border-color: var(--cyan) !important;
  }

  [data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--violet), var(--cyan));
    color: #fff;
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1.4rem;
    font-weight: 700;
  }
  [data-testid="stButton"] button:hover { filter: brightness(1.15); transform: translateY(-1px); }

  a { color: var(--cyan); }
</style>
"""

_ORBS_HTML = """
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
"""


def apply_theme():
    st.markdown(_ORBS_HTML, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
