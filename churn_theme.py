"""NEXUS theme for the churn demo.

Dark-luxury futuristic styling in the NEXUS visual language:
near-black base, blue/cyan/violet accents, film grain, ambient glows and a
genuine 3D "neural globe" built from pure CSS transforms (perspective,
preserve-3d, rotateX/Y). No CDN, no WebGL, no network requests — cannot
break offline or on blocked networks.
"""
import streamlit as st

_CSS = """
<style>
  :root {
    --bg0: #020305;
    --bg1: #05070B;
    --bg2: #080A10;
    --panel: rgba(255,255,255,0.03);
    --panel-strong: rgba(255,255,255,0.05);
    --line: rgba(255,255,255,0.08);
    --line-strong: rgba(255,255,255,0.14);
    --text: #F5F7FA;
    --muted: #8B93A1;
    --dim: #525967;
    --blue: #1677FF;
    --cyan: #20D9FF;
    --violet: #8B4DFF;
    --violet-2: #A855F7;
  }

  html, body, .stApp {
    background: var(--bg0);
    color: var(--text);
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  /* film grain */
  .nx-grain {
    position: fixed; inset: -50%; z-index: 0; pointer-events: none; opacity: 0.05;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    animation: nxGrain 8s steps(6) infinite;
  }
  @keyframes nxGrain {
    0%,100% { transform: translate(0,0); }
    20% { transform: translate(-3%,2%); }
    40% { transform: translate(2%,-4%); }
    60% { transform: translate(-2%,-2%); }
    80% { transform: translate(3%,3%); }
  }

  /* ambient glow orbs */
  .nx-orb {
    position: fixed; border-radius: 50%; z-index: 0; pointer-events: none;
    filter: blur(110px); opacity: 0.55; animation: nxFloat 16s ease-in-out infinite;
  }
  .nx-orb-a { width: 52vw; height: 52vw; top: -16vw; right: -10vw;
    background: radial-gradient(circle, rgba(22,119,255,0.22), transparent 62%); }
  .nx-orb-b { width: 46vw; height: 46vw; bottom: -14vw; left: -8vw;
    background: radial-gradient(circle, rgba(139,77,255,0.18), transparent 62%); animation-delay: -6s; }
  .nx-orb-c { width: 34vw; height: 34vw; top: 46vh; left: 30vw;
    background: radial-gradient(circle, rgba(32,217,255,0.10), transparent 62%); animation-delay: -11s; }
  @keyframes nxFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(2vw, -3vh) scale(1.08); }
  }

  /* ================= 3D NEURAL GLOBE ================= */
  .scene {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    perspective: 900px; overflow: hidden;
  }
  .gimbal {
    position: absolute; top: 50%; left: 50%; width: 0; height: 0;
    transform-style: preserve-3d;
    animation: nxSpin 30s linear infinite;
  }
  .gimbal-2 {
    position: absolute; width: 0; height: 0;
    transform-style: preserve-3d;
    animation: nxTilt 8s ease-in-out infinite;
  }
  @keyframes nxSpin { from { transform: rotateY(0deg) rotateX(14deg); } to { transform: rotateY(360deg) rotateX(14deg); } }
  @keyframes nxTilt { 0%,100% { transform: rotateX(0deg); } 50% { transform: rotateX(-12deg); } }

  .ring {
    position: absolute; width: 380px; height: 380px; margin: -190px 0 0 -190px;
    border-radius: 50%; transform-style: preserve-3d;
    border: 2px solid rgba(32,217,255,0.32);
    box-shadow: 0 0 26px rgba(32,217,255,0.22), inset 0 0 26px rgba(32,217,255,0.08);
  }
  .ring.r1 { transform: rotateY(0deg); }
  .ring.r2 { transform: rotateY(60deg); }
  .ring.r3 { transform: rotateY(120deg); }
  .ring.r4 { transform: rotateY(180deg); }
  .ring.r5 { transform: rotateY(240deg); }
  .ring.r6 { transform: rotateY(300deg); }
  .ring.roll { transform: rotateX(90deg); border-color: rgba(139,77,255,0.28); box-shadow: 0 0 26px rgba(139,77,255,0.22); }
  .ring.roll2 { transform: rotateX(90deg) rotateY(90deg); border-color: rgba(22,119,255,0.30); box-shadow: 0 0 26px rgba(22,119,255,0.20); }

  .node {
    position: absolute; width: 14px; height: 14px; border-radius: 50%;
    margin: -7px 0 0 -7px; left: 190px; top: 190px; transform-style: preserve-3d;
  }
  .node span {
    display: block; width: 100%; height: 100%; border-radius: 50%;
    background: radial-gradient(circle, #ffffff, var(--cyan));
    box-shadow: 0 0 18px 4px rgba(32,217,255,0.6);
    animation: nxPulse 2.4s ease-in-out infinite;
  }
  .node.n2 span { background: radial-gradient(circle, #ffffff, var(--violet)); box-shadow: 0 0 18px 4px rgba(139,77,255,0.6); animation-delay: -0.6s; }
  .node.n3 span { background: radial-gradient(circle, #ffffff, var(--blue));   box-shadow: 0 0 18px 4px rgba(22,119,255,0.6);  animation-delay: -1.2s; }
  .node.n4 span { background: radial-gradient(circle, #ffffff, var(--violet-2)); box-shadow: 0 0 18px 4px rgba(168,85,247,0.6); animation-delay: -1.8s; }
  @keyframes nxPulse { 0%,100% { transform: scale(0.7); opacity: 0.6; } 50% { transform: scale(1.3); opacity: 1; } }

  .cube {
    position: absolute; width: 46px; height: 46px; transform-style: preserve-3d;
    border: 1px solid rgba(32,217,255,0.20);
    background: rgba(22,119,255,0.10);
    box-shadow: 0 0 40px rgba(22,119,255,0.35);
    animation: nxCube 14s linear infinite, nxDrift 18s ease-in-out infinite;
  }
  .cube.c2 { width: 34px; height: 34px; border-color: rgba(139,77,255,0.22); background: rgba(139,77,255,0.10); box-shadow: 0 0 40px rgba(139,77,255,0.35); animation-delay: -5s, -7s; }
  .cube.c3 { width: 26px; height: 26px; border-color: rgba(32,217,255,0.24); background: rgba(32,217,255,0.10); box-shadow: 0 0 40px rgba(32,217,255,0.35); animation-delay: -9s, -12s; }
  @keyframes nxCube { from { transform: rotateX(0deg) rotateY(0deg); } to { transform: rotateX(360deg) rotateY(360deg); } }
  @keyframes nxDrift { 0%,100% { transform: translate(0,0); } 50% { transform: translate(1.2vw, -2vh); } }

  /* ================= CONTENT ================= */
  [data-testid="stAppViewContainer"] .main { z-index: 1; position: relative; }

  h1, h2, h3, h4 {
    font-family: 'Segoe UI', -apple-system, sans-serif;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  [data-testid="stHeading"] h1 {
    background: linear-gradient(100deg, var(--blue), var(--cyan) 55%, var(--violet));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    font-weight: 800;
  }

  .nx-eyebrow {
    font-size: 12px; letter-spacing: 0.34em; text-transform: uppercase;
    color: var(--cyan); margin-bottom: 10px; opacity: 0.9;
  }

  [data-testid="stCaptionContainer"], [data-testid="stCaption"] { color: var(--muted); }

  /* glass metric card */
  [data-testid="stMetric"] {
    position: relative;
    background: linear-gradient(165deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 18px 46px rgba(0,0,0,0.45);
    overflow: hidden;
  }
  [data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue), var(--cyan), var(--violet));
  }
  [data-testid="stMetricLabel"] { color: var(--muted); letter-spacing: 0.04em; }
  [data-testid="stMetricValue"] { color: var(--text); font-weight: 700; }
  [data-testid="stMetricDelta"] { color: var(--cyan); font-weight: 600; }

  /* glass panels */
  [data-testid="stExpander"],
  [data-testid="stAlert"] {
    background: linear-gradient(165deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border: 1px solid var(--line);
    border-radius: 16px;
    backdrop-filter: blur(10px);
  }
  [data-testid="stAlert"] { border-left: 3px solid var(--cyan); }
  [data-testid="stExpander"] summary { color: var(--text); font-weight: 600; }

  /* sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg1), var(--bg2));
    border-right: 1px solid var(--line);
  }

  /* form controls */
  .stSelectbox [data-baseweb="select"] > div,
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--cyan); }
  .stSlider [data-baseweb="slider"] > div > div { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
  .stSelectbox [data-baseweb="select"] { background: var(--panel-strong); border-radius: 10px; }
  .stSelectbox [data-baseweb="select"] > div { border-color: var(--line-strong) !important; }

  .stCheckbox [data-testid="stCheckbox"] [role="checkbox"]:checked,
  .stToggle [data-testid="stCheckbox"] [role="checkbox"]:checked {
    background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
    border-color: var(--cyan) !important;
  }

  [data-testid="stRadio"] label { color: var(--muted); }
  [data-testid="stRadio"] input:checked + div { color: var(--cyan); }

  [data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    color: #041018;
    border: none; border-radius: 999px;
    padding: 0.55rem 1.6rem;
    font-weight: 700; letter-spacing: 0.02em;
    box-shadow: 0 10px 30px rgba(22,119,255,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
  }
  [data-testid="stButton"] button:hover { filter: brightness(1.12); transform: translateY(-1px); box-shadow: 0 14px 38px rgba(32,217,255,0.45); }

  a { color: var(--cyan); }

  .nx-footer {
    margin-top: 56px; padding-top: 18px;
    border-top: 1px solid var(--line);
    color: var(--dim); font-size: 12.5px; letter-spacing: 0.12em; text-align: center;
  }
  .nx-footer b { color: var(--muted); font-weight: 600; }
</style>
"""

_SCENE_HTML = """
<div class="nx-grain"></div>
<div class="nx-orb nx-orb-a"></div>
<div class="nx-orb nx-orb-b"></div>
<div class="nx-orb nx-orb-c"></div>
<div class="scene">
  <div class="gimbal">
    <div class="gimbal-2">
      <div class="ring r1"></div>
      <div class="ring r2"></div>
      <div class="ring r3"></div>
      <div class="ring r4"></div>
      <div class="ring r5"></div>
      <div class="ring r6"></div>
      <div class="ring roll"></div>
      <div class="ring roll2"></div>
      <div class="node" style="left:190px;top:60px"><span></span></div>
      <div class="node n2" style="left:340px;top:190px"><span></span></div>
      <div class="node n3" style="left:190px;top:320px"><span></span></div>
      <div class="node n4" style="left:40px;top:190px"><span></span></div>
    </div>
  </div>
  <div class="cube" style="left:12vw; top:65vh;"></div>
  <div class="cube c2" style="left:84vw; top:18vh;"></div>
  <div class="cube c3" style="left:88vw; top:78vh;"></div>
</div>
"""


def apply_theme():
    st.markdown(_SCENE_HTML, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
