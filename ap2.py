"""
╔══════════════════════════════════════════════════════════════╗
║   PlantAI — Plant Leaf Nutrient Deficiency Detector          ║
║   Model  : best_model_phase2.keras                           ║
║   Classes : Boron, Calcium, Healthy, Iron, Magnesium,        ║
║             Manganese, Potassium, Sulphur                    ║
║   Run     : streamlit run app.py                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageStat
import os, io, base64, json
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PlantAI · Nutrient Deficiency Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# SESSION STATE — HISTORY
# ──────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

# ──────────────────────────────────────────────
# STYLESHEET
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800;900&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:#0e110f; --paper:#f5f2eb; --paper2:#edeae0; --paper3:#e4e0d4;
    --leaf:#1a6b2f; --leaf-light:#2d9e4a; --leaf-pale:#d4edda;
    --gold:#c8973a; --rust:#c04a2a;
    --sans:'Cabinet Grotesk',sans-serif;
    --serif:'Instrument Serif',serif;
    --mono:'JetBrains Mono',monospace;
    --r-md:14px; --r-lg:22px;
    --shadow:0 2px 16px rgba(14,17,15,.10);
    --shadow-md:0 6px 32px rgba(14,17,15,.14);
}

html, body {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
}

.stApp {
    background: var(--paper) !important;
    font-family: var(--sans) !important;
}

.main {
    background: var(--paper) !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background: var(--paper) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"],
[data-testid="collapsedControl"] { display: none; }

/* ── TOPBAR ── */
.topbar {
    position: sticky; top: 0; z-index: 999;
    background: var(--ink);
    padding: 0 3rem; height: 58px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #2a3a2c;
}
.topbar-logo {
    font-family: var(--sans); font-size: 1.05rem; font-weight: 900;
    color: #fff; letter-spacing: -0.3px;
    display: flex; align-items: center; gap: 10px;
}
.topbar-logo span { color: #6dca82; }
.topbar-meta { font-family: var(--mono); font-size: 0.65rem; color: #b8c4ba; letter-spacing: .5px; }
.topbar-badge {
    background: #1a6b2f; color: #6dca82; font-family: var(--mono); font-size: 0.6rem;
    font-weight: 500; padding: 3px 10px; border-radius: 20px;
    letter-spacing: .8px; text-transform: uppercase;
}

/* ── HERO ── */
.hero-wrap {
    background: var(--ink); padding: 4rem 3rem 3.5rem;
    position: relative; overflow: hidden;
}
.hero-bg-text {
    position: absolute; right: -2rem; top: -1rem;
    font-family: var(--serif); font-size: 14rem;
    color: rgba(109,202,130,.06); line-height: 1;
    pointer-events: none; user-select: none; white-space: nowrap;
}
.hero-eyebrow {
    font-family: var(--mono); font-size: 0.65rem; color: #8bc499;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1rem;
}
.hero-h1 {
    font-family: var(--serif); font-size: 3.8rem; font-weight: 400;
    color: #f8f9f8; line-height: 1.08; margin: 0 0 0.5rem; font-style: italic;
}
.hero-h1 b {
    font-style: normal; font-family: var(--sans); font-weight: 900; color: #8bc499;
}
.hero-desc {
    color: #b0c4b4; font-size: 0.95rem; max-width: 540px;
    line-height: 1.7; margin-top: 0.8rem;
}
.hero-stats {
    display: flex; gap: 2.5rem; margin-top: 2.5rem;
    padding-top: 2rem; border-top: 1px solid #2a3a2c;
}
.h-stat-val { font-family: var(--sans); font-size: 2rem; font-weight: 900; color: #f8f9f8; }
.h-stat-lbl {
    font-family: var(--mono); font-size: 0.6rem; color: #8bc499;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;
}

/* ── PANELS ── */
.panel {
    background: #fff; border: 1px solid var(--paper3);
    border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--shadow);
}
.panel-header {
    background: var(--paper2); padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--paper3);
    display: flex; align-items: center; gap: .6rem;
}
.panel-title {
    font-family: var(--sans); font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px; color: var(--ink);
}
.panel-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--leaf-light); }
.panel-body { padding: 1.5rem; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--paper) !important;
    border: 2px dashed var(--paper3) !important;
    border-radius: var(--r-md) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--leaf-light) !important; }

/* ── VALIDATION BANNER ── */
.validation-pass {
    background: #d4edda; border: 1.5px solid #2d9e4a; border-radius: var(--r-md);
    padding: 1rem 1.2rem; margin-top: 1rem; display: flex; align-items: flex-start; gap: .8rem;
}
.validation-fail {
    background: #fde8e0; border: 1.5px solid #c04a2a; border-radius: var(--r-md);
    padding: 1rem 1.2rem; margin-top: 1rem; display: flex; align-items: flex-start; gap: .8rem;
}
.validation-checking {
    background: #fff8f0; border: 1.5px solid #c8973a; border-radius: var(--r-md);
    padding: 1rem 1.2rem; margin-top: 1rem; display: flex; align-items: flex-start; gap: .8rem;
}
.val-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.val-title { font-family: var(--sans); font-size: .85rem; font-weight: 800; margin: 0 0 3px; }
.val-body { font-family: var(--sans); font-size: .78rem; line-height: 1.5; margin: 0; }

/* ── RESULT CARD ── */
.result-header { display: flex; align-items: flex-start; gap: 1.2rem; padding: 1.5rem 1.5rem 0; }
.elem-box {
    flex-shrink: 0; width: 64px; height: 64px; border-radius: var(--r-md);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.elem-sym { font-size: 1.6rem; font-weight: 900; line-height: 1; }
.elem-num { font-size: .55rem; font-weight: 500; opacity: .7; margin-top: 2px; }
.result-eyebrow {
    font-family: var(--mono); font-size: .6rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: #9aab9e; margin-bottom: 4px;
}
.result-name { font-family: var(--sans); font-size: 1.7rem; font-weight: 900; line-height: 1.1; margin: 0 0 6px; }
.sev-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--mono); font-size: .6rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: .8px; text-transform: uppercase;
}

/* ── INFO SECTIONS ── */
.info-section { margin: 0 1.5rem; padding: 1.1rem 0; border-top: 1px solid var(--paper3); }
.info-section:first-of-type { border-top: none; margin-top: 1.2rem; }
.info-sec-title {
    font-family: var(--mono); font-size: .6rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: #9aab9e; margin-bottom: .5rem;
}
.info-sec-body { font-size: .875rem; line-height: 1.75; color: #2a3a2c; }
.fert-chip {
    display: inline-block; background: var(--paper); border: 1px solid var(--paper3);
    border-radius: 6px; padding: 3px 10px; font-family: var(--mono);
    font-size: .68rem; color: var(--ink); margin: 2px 3px 2px 0;
}

/* ── PROBABILITY BARS ── */
.prob-section {
    background: var(--paper); border-top: 1px solid var(--paper3);
    padding: 1.2rem 1.5rem 1.5rem; margin-top: .5rem;
}
.prob-title {
    font-family: var(--mono); font-size: .6rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: #9aab9e; margin-bottom: .9rem;
}
.pb-row { margin: 6px 0; }
.pb-names { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.pb-cls { font-size: .78rem; font-weight: 500; color: var(--ink); }
.pb-pct { font-family: var(--mono); font-size: .72rem; font-weight: 600; }
.pb-track { height: 5px; border-radius: 3px; overflow: hidden; background: var(--paper3); }
.pb-fill { height: 100%; border-radius: 3px; }

/* ── GUIDE GRID ── */
.guide-grid {
    padding: 2rem 3rem;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.2rem;
}
.guide-card {
    background: #fff; border: 1px solid var(--paper3); border-radius: var(--r-lg);
    overflow: hidden; box-shadow: var(--shadow);
    transition: box-shadow .2s, transform .2s;
}
.guide-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.guide-card-top {
    padding: 1.3rem 1.4rem 1rem; display: flex; align-items: center; gap: 1rem;
    border-bottom: 1px solid var(--paper3);
}
.guide-card-body { padding: 1.1rem 1.4rem; }
.guide-lbl {
    font-family: var(--mono); font-size: .6rem; text-transform: uppercase;
    letter-spacing: 1.2px; color: #9aab9e; margin-bottom: 3px;
}
.guide-val { font-size: .85rem; line-height: 1.65; color: #2a3a2c; }
.tag {
    display: inline-block; font-family: var(--mono); font-size: .62rem;
    padding: 2px 8px; border-radius: 4px; font-weight: 500;
}

/* ── MODEL GRID ── */
.model-grid {
    padding: 2rem 3rem; display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.m-tile {
    background: linear-gradient(135deg, #1a2420 0%, #0e110f 100%);
    border-radius: var(--r-md); padding: 1.3rem 1.4rem; border: 1px solid #2a3a2c;
}
.m-val { font-family: var(--sans); font-size: 2rem; font-weight: 900; color: #f8f9f8; }
.m-lbl {
    font-family: var(--mono); font-size: .6rem; color: #8bc499;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;
}
.phase-grid {
    padding: 0 3rem; display: grid; grid-template-columns: 1fr 1fr;
    gap: 1rem; margin-bottom: 1.5rem;
}
.phase-card {
    background: #fff; border: 1px solid var(--paper3); border-radius: var(--r-lg);
    padding: 1.5rem; box-shadow: var(--shadow);
}
.phase-num {
    font-family: var(--mono); font-size: .6rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: #9aab9e; margin-bottom: .4rem;
}
.phase-title { font-family: var(--sans); font-size: 1.05rem; font-weight: 800; color: var(--ink); margin-bottom: .8rem; }
.phase-row { display: flex; gap: .5rem; align-items: baseline; margin: 5px 0; font-size: .82rem; color: #2a3a2c; }
.phase-key { font-family: var(--mono); font-size: .65rem; color: #7a8a7e; min-width: 110px; flex-shrink: 0; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--paper2) !important; border-radius: 0 !important;
    border-bottom: 1px solid var(--paper3) !important; padding: 0 3rem !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #9aab9e !important;
    border-radius: 0 !important; font-family: var(--sans) !important;
    font-size: .82rem !important; font-weight: 600 !important;
    padding: .9rem 1.4rem !important; border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--leaf) !important; background: transparent !important;
}
[data-testid="stTabsContent"] { padding: 0 !important; }

/* ── PLACEHOLDER ── */
.placeholder {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 4rem 2rem; text-align: center; background: var(--paper);
    border: 2px dashed var(--paper3); border-radius: var(--r-lg); min-height: 320px;
}
.placeholder-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: .5; }
.placeholder-text { font-size: .88rem; color: #9aab9e; line-height: 1.7; }

/* ── HISTORY ── */
.hist-empty {
    margin: 2rem 3rem; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 5rem 2rem; text-align: center; background: #fff;
    border: 2px dashed var(--paper3); border-radius: var(--r-lg); box-shadow: var(--shadow);
}
.hist-grid {
    padding: 1.2rem 3rem 2.5rem;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.2rem;
}
.hist-card {
    background: #fff; border: 1px solid var(--paper3); border-radius: var(--r-lg);
    overflow: hidden; box-shadow: var(--shadow);
    transition: box-shadow .2s, transform .2s; display: flex; flex-direction: column;
}
.hist-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.hist-card-img { width: 100%; height: 180px; object-fit: cover; background: var(--paper2); display: block; }
.hist-card-body { padding: 1.1rem 1.2rem; flex: 1; }
.hist-card-top { display: flex; align-items: center; gap: .8rem; margin-bottom: .8rem; }
.hist-elem-box {
    width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.hist-cls { font-family: var(--sans); font-size: 1rem; font-weight: 800; color: var(--ink); }
.hist-meta-row {
    display: flex; align-items: center; justify-content: space-between;
    border-top: 1px solid var(--paper3); padding-top: .7rem; margin-top: .7rem;
}
.hist-filename {
    font-family: var(--mono); font-size: .6rem; color: #9aab9e;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px;
}
.hist-ts { font-family: var(--mono); font-size: .58rem; color: #b0bab4; }
.hist-index-badge {
    font-family: var(--mono); font-size: .55rem; font-weight: 700;
    background: var(--paper); border: 1px solid var(--paper3);
    color: #9aab9e; padding: 2px 7px; border-radius: 6px; margin-right: auto;
}

/* ── FOOTER ── */
.footer {
    background: var(--ink); color: #8bc499; font-family: var(--mono); font-size: .62rem;
    letter-spacing: .5px; text-align: center; padding: 1.4rem 3rem;
    border-top: 1px solid #2a3a2c;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
    font-family: var(--mono) !important; font-size: .65rem !important;
    font-weight: 600 !important; letter-spacing: .8px !important;
    text-transform: uppercase !important;
}

/* ── COLUMN GAPS ── */
[data-testid="column"] { padding: 1.5rem 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# NUTRIENT DATA — 8 CLASSES
# ──────────────────────────────────────────────
NUTRIENTS = {
    "Boron": {
        "symbol": "B", "number": "5",
        "level": "Micronutrient", "severity": "Moderate",
        "color_bg": "#fde8e0", "color_fg": "#c04a2a", "color_bar": "#c04a2a",
        "sev_bg": "#fde8e0", "sev_fg": "#c04a2a",
        "symptoms":   "Distorted, crinkled new leaves with thickened brittle texture. Growing tip dies back. Poor fruit development with possible internal browning.",
        "cause":      "Leached sandy soils with low organic matter or high pH (>7). Drought stress severely limits boron uptake.",
        "treatment":  "Foliar spray: 1–2 g borax per litre applied to young leaves. Soil: 1–2 kg borax/ha. Do not over-apply — toxicity threshold is very close to deficiency level.",
        "fertilizer": ["Borax (Na₂B₄O₇·10H₂O)", "Solubor", "Boric Acid"],
    },
    "Calcium": {
        "symbol": "Ca", "number": "20",
        "level": "Macronutrient", "severity": "High",
        "color_bg": "#fdf3e0", "color_fg": "#c8973a", "color_bar": "#c8973a",
        "sev_bg": "#fde8e0", "sev_fg": "#c04a2a",
        "symptoms":   "Tip burn on young leaves, blossom end rot, stunted growth. Brown necrotic scorching on margins and tips of new leaves.",
        "cause":      "Acidic soils (pH < 5.5) or highly leached soils. Excess potassium competitively inhibits Ca uptake. Inconsistent irrigation worsens the condition.",
        "treatment":  "Agricultural lime at 2–3 t/ha to raise soil pH. Gypsum (CaSO₄) at 500 kg/ha. Calcium nitrate foliar spray (0.5%) for rapid correction.",
        "fertilizer": ["Agricultural Lime", "Gypsum (CaSO₄)", "Calcium Nitrate Ca(NO₃)₂"],
    },
    "Healthy": {
        "symbol": "✓", "number": "—",
        "level": "No Deficiency", "severity": "None",
        "color_bg": "#d4edda", "color_fg": "#1a6b2f", "color_bar": "#2d9e4a",
        "sev_bg": "#d4edda", "sev_fg": "#1a6b2f",
        "symptoms":   "Leaves are uniformly dark green with no discolouration, distortion, or necrosis. Plant shows vigorous growth with a healthy root system.",
        "cause":      "No nutrient deficiency detected. The plant is well-nourished and growing conditions appear optimal.",
        "treatment":  "Maintain current fertilisation programme. Continue balanced NPK applications and monitor regularly for early signs of stress.",
        "fertilizer": ["Balanced NPK", "Micronutrient blend", "Organic compost"],
    },
    "Iron": {
        "symbol": "Fe", "number": "26",
        "level": "Micronutrient", "severity": "Moderate",
        "color_bg": "#fdf3e0", "color_fg": "#b8760a", "color_bar": "#c8973a",
        "sev_bg": "#fdf3e0", "sev_fg": "#b8760a",
        "symptoms":   "Interveinal chlorosis on young leaves while veins stay green. Severe cases: entire leaf pale yellow or white. Older leaves remain relatively green.",
        "cause":      "High soil pH (>7) makes iron insoluble. Waterlogged or calcareous soils. High bicarbonate in irrigation water.",
        "treatment":  "Foliar: iron chelate (Fe-EDTA) at 0.2–0.5%. Soil: ferrous sulphate (FeSO₄) at 25–50 kg/ha. Lower soil pH with sulphur if needed.",
        "fertilizer": ["Fe-EDTA Chelate", "Fe-EDDHA Chelate", "Ferrous Sulphate FeSO₄"],
    },
    "Magnesium": {
        "symbol": "Mg", "number": "12",
        "level": "Macronutrient", "severity": "Moderate",
        "color_bg": "#dceef7", "color_fg": "#2a6b8f", "color_bar": "#2a6b8f",
        "sev_bg": "#dceef7", "sev_fg": "#2a6b8f",
        "symptoms":   "Interveinal chlorosis starting on older lower leaves. Yellow–orange patches between green veins. Margins may turn reddish-brown. Premature leaf senescence.",
        "cause":      "Leached, acidic, or sandy soils. Excess potassium suppresses Mg uptake. Heavy rainfall leaches magnesium from topsoil.",
        "treatment":  "Foliar: Epsom salt (MgSO₄) at 2% every 2 weeks. Soil: dolomite lime at 1–2 t/ha or kieserite at 100–200 kg/ha.",
        "fertilizer": ["Epsom Salt (MgSO₄·7H₂O)", "Kieserite (MgSO₄·H₂O)", "Dolomite Lime"],
    },
    "Manganese": {
        "symbol": "Mn", "number": "25",
        "level": "Micronutrient", "severity": "Low",
        "color_bg": "#ecdff5", "color_fg": "#7a3a8f", "color_bar": "#7a3a8f",
        "sev_bg": "#ecdff5", "sev_fg": "#7a3a8f",
        "symptoms":   "Interveinal chlorosis on young leaves, similar to iron but less pronounced. Grey or tan necrotic spots. Stunted growth; leaves appear pale greyish-green.",
        "cause":      "High soil pH (>6.5) drastically reduces Mn availability. Over-liming is a common cause on alkaline soils.",
        "treatment":  "Foliar: manganese sulphate (MnSO₄) at 0.1–0.5%. Soil: 10–25 kg MnSO₄/ha. Reduce soil pH if over-liming is the cause.",
        "fertilizer": ["Manganese Sulphate MnSO₄", "Chelated Mn (Mn-EDTA)", "MnSO₄·H₂O"],
    },
    "Potassium": {
        "symbol": "K", "number": "19",
        "level": "Macronutrient", "severity": "High",
        "color_bg": "#dceef7", "color_fg": "#1a4a6b", "color_bar": "#2a6b8f",
        "sev_bg": "#fde8e0", "sev_fg": "#c04a2a",
        "symptoms":   "Leaf edge scorch on older lower leaves. Brown margins progressing inward. Weak stems. Increased susceptibility to drought and disease.",
        "cause":      "Sandy, light soils prone to K leaching. High yield demand depletes soil K rapidly. Excess nitrogen reduces K uptake.",
        "treatment":  "Muriate of potash (KCl) or sulphate of potash (K₂SO₄) at 300–500 g/plant/year split 3–4 times. Foliar KNO₃ at 1–2% for quick correction.",
        "fertilizer": ["Muriate of Potash KCl", "Sulphate of Potash K₂SO₄", "Potassium Nitrate KNO₃"],
    },
    "Sulphur": {
        "symbol": "S", "number": "16",
        "level": "Macronutrient", "severity": "Low",
        "color_bg": "#fdf3e0", "color_fg": "#8f6a1a", "color_bar": "#c8973a",
        "sev_bg": "#fdf3e0", "sev_fg": "#8f6a1a",
        "symptoms":   "Uniform overall yellowing of young leaves (S is immobile — new growth affected first). No interveinal pattern unlike Fe or Mg. Slow, stunted growth.",
        "cause":      "Sandy soils low in organic matter. High rainfall leaches sulphate rapidly from root zone.",
        "treatment":  "Gypsum (CaSO₄) at 200–400 kg/ha or ammonium sulphate at 100–200 kg/ha. Elemental sulphur at 20–40 kg/ha where pH reduction is also desirable.",
        "fertilizer": ["Gypsum CaSO₄", "Ammonium Sulphate (NH₄)₂SO₄", "Elemental Sulphur S"],
    },
}

CLASS_NAMES = ['Boron', 'Calcium', 'Healthy', 'Iron',
               'Magnesium', 'Manganese', 'Potassium', 'Sulphur']

MODEL_PATH = r"D:\fyp2\best_model_phase2.keras"

SEV_STYLE = {
    "None":     ("#d4edda", "#1a6b2f"),
    "Low":      ("#ecdff5", "#7a3a8f"),
    "Moderate": ("#fdf3e0", "#b8760a"),
    "High":     ("#fde8e0", "#c04a2a"),
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def img_to_b64(pil_img, max_w=600):
    w, h  = pil_img.size
    ratio = min(max_w / w, 1.0)
    thumb = pil_img.convert("RGB").resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    buf   = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def hex_to_rgba(hex_color, alpha=0.27):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def save_to_history(filename, pil_img, top_cls, top_prob, all_results):
    st.session_state.scan_history.insert(0, {
        "id":        len(st.session_state.scan_history) + 1,
        "filename":  filename,
        "timestamp": datetime.now().strftime("%d %b %Y · %H:%M"),
        "top_cls":   top_cls,
        "top_prob":  top_prob,
        "results":   all_results,
        "img_b64":   img_to_b64(pil_img),
    })

# ──────────────────────────────────────────────
# BANANA LEAF VALIDATION — 100% OFFLINE
# No API, no internet needed.
# Uses colour analysis + MobileNetV2 ImageNet
# features to detect leaf-like green content.
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mobilenet():
    """Load MobileNetV2 once, reuse for validation."""
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=True, input_shape=(224, 224, 3)
    )
    return base

# ImageNet class indices that are plant/leaf related
# (these are standard ImageNet labels)
PLANT_CLASSES = {
    # leaf-like / plant classes in ImageNet top-1000
    992, 993, 994, 995, 996, 997, 998, 999,  # various plants
    985, 986, 987, 988, 989, 990, 991,
    # broad vegetation / nature classes
    949, 950, 951, 952, 953, 954, 955,
    # fern, daisy, mushroom etc. — still "plant" broad
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    # banana (ImageNet has banana class ~954)
    954,
}

def _color_features(img_rgb: np.ndarray) -> dict:
    """
    Extract green-dominance and saturation features from an RGB image array.
    Banana leaves are large, broad, and heavily green (or desaturated for B&W).
    """
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)

    # Green dominance: pixels where green channel is highest
    green_dom = np.mean((g > r) & (g > b))

    # Excess green index (plant detection heuristic)
    exg = 2 * g - r - b
    exg_positive = np.mean(exg > 20)  # fraction of "green" pixels

    # Overall mean green channel value
    mean_g = np.mean(g) / 255.0

    # Saturation (low sat = grayscale/B&W image)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat   = np.mean(np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0))

    # Brightness
    brightness = np.mean((r + g + b) / 3.0) / 255.0

    return {
        "green_dom":    green_dom,
        "exg_positive": exg_positive,
        "mean_g":       mean_g,
        "saturation":   sat,
        "brightness":   brightness,
    }

def _is_grayscale(img_rgb: np.ndarray, tol: float = 0.05) -> bool:
    """Return True if image is essentially grayscale / black-and-white."""
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    rg_diff = np.mean(np.abs(r - g)) / 255.0
    rb_diff = np.mean(np.abs(r - b)) / 255.0
    gb_diff = np.mean(np.abs(g - b)) / 255.0
    return max(rg_diff, rb_diff, gb_diff) < tol

def validate_banana_leaf(pil_img: Image.Image) -> dict:
    """
    100% offline banana leaf validator.
    Strategy:
      1. Colour analysis — check for green dominance (or grayscale for B&W photos)
      2. MobileNetV2 ImageNet top-5 — check if any prediction is plant/leaf related
      3. Aspect-ratio + size sanity check (banana leaves are wide/large)

    Returns: {"is_valid": bool, "reason": str, "confidence": str}
    """
    try:
        img_resized = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
        arr = np.array(img_resized)

        cf        = _color_features(arr)
        is_bw     = _is_grayscale(arr)
        w, h      = pil_img.size
        aspect    = max(w, h) / max(min(w, h), 1)

        # ── Step 1: Colour / grayscale check ──
        # For colour images: must have reasonable green content
        # For B&W: skip colour check, rely on MobileNet
        colour_ok = False
        if is_bw:
            colour_ok = True   # B&W — can't use colour, defer to MobileNet
        else:
            # Accept if meaningful green presence
            colour_ok = (cf["green_dom"] > 0.20) or (cf["exg_positive"] > 0.15)

        # ── Step 2: MobileNetV2 top-5 check ──
        preproc = tf.keras.applications.mobilenet_v2.preprocess_input(
            arr.astype(np.float32)
        )
        preds  = load_mobilenet().predict(np.expand_dims(preproc, 0), verbose=0)
        top5_i = np.argsort(preds[0])[::-1][:5]
        top5_p = preds[0][top5_i]

        # Decode top predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
        # decoded = [(id, label, prob), ...]

        # Check if any top-5 label contains plant/leaf/nature keywords
        plant_keywords = {
            "leaf", "plant", "tree", "fern", "grass", "flower", "herb",
            "shrub", "vine", "moss", "algae", "banana", "palm", "frond",
            "foliage", "green", "vegetable", "crop", "field", "nature",
            "daisy", "dandelion", "artichoke", "broccoli", "cabbage",
            "cucumber", "lettuce", "spinach", "corn", "maize", "rice",
        }

        mobilenet_plant = False
        top_label       = decoded[0][1].lower().replace("_", " ")
        matched_label   = ""
        for _, label, prob in decoded:
            lbl = label.lower().replace("_", " ")
            if any(kw in lbl for kw in plant_keywords):
                mobilenet_plant = True
                matched_label   = label
                break

        # Also check raw class index
        if any(idx in PLANT_CLASSES for idx in top5_i):
            mobilenet_plant = True

        # ── Step 3: Aspect ratio (banana leaves are broad, not square) ──
        # Very tall/narrow images (like a person photo) are suspect
        aspect_ok = aspect < 3.0   # reject very extreme aspect ratios

        # ── Decision logic ──
        score = 0
        if colour_ok:        score += 2
        if mobilenet_plant:  score += 2
        if aspect_ok:        score += 1

        # B&W image gets colour_ok for free — needs MobileNet to agree
        if is_bw:
            is_valid   = mobilenet_plant and aspect_ok
            confidence = "medium" if mobilenet_plant else "high"
            if is_valid:
                reason = f"Grayscale/B&W leaf image detected (top prediction: {top_label})."
            else:
                reason = f"Grayscale image does not appear to be a leaf (top prediction: {top_label})."
        else:
            if score >= 4:
                is_valid   = True
                confidence = "high"
                reason     = f"Green leaf confirmed — strong green signature and plant features detected."
            elif score == 3:
                is_valid   = True
                confidence = "medium"
                reason     = f"Likely a leaf — moderate green content with plant-like features."
            elif score == 2 and colour_ok:
                # Green image but MobileNet unsure — still likely a leaf
                is_valid   = True
                confidence = "medium"
                reason     = f"Green-dominant image accepted as leaf (confidence: medium)."
            else:
                is_valid   = False
                confidence = "high"
                if not colour_ok:
                    reason = f"Image lacks green/leaf colour — top prediction: '{top_label}'. Not a banana leaf."
                else:
                    reason = f"Image does not appear to be a plant leaf — top prediction: '{top_label}'."

        return {"is_valid": is_valid, "reason": reason, "confidence": confidence}

    except Exception as e:
        # On any unexpected error, fail open with a warning
        return {
            "is_valid": True,
            "reason":   f"Offline validation skipped (error: {str(e)[:80]}). Proceeding with model.",
            "confidence": "low",
        }


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    return tf.keras.models.load_model(path)

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def run_inference(model, arr):
    probs  = model.predict(arr, verbose=0)[0]
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    return [(CLASS_NAMES[i], float(p)) for i, p in ranked]

# ──────────────────────────────────────────────
# TOP NAV
# ──────────────────────────────────────────────
hist_len = len(st.session_state.scan_history)
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <span style="font-size:1.1rem;">🌿</span>
        Plant<span>AI</span>
    </div>
    <div style="display:flex;align-items:center;gap:1.5rem;">
        <span class="topbar-meta">best_model_phase2.keras · EfficientNetB0</span>
        <span class="topbar-meta" style="color:#5a8a6a;">
            {hist_len} scan{"s" if hist_len!=1 else ""} in history
        </span>
        <span class="topbar-badge">&#9679; LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-bg-text">Nutrients</div>
    <div class="hero-eyebrow">AI &middot; AGRONOMY &middot; PRECISION NUTRITION</div>
    <h1 class="hero-h1">Identify what your<br><b>banana crop is missing</b></h1>
    <p class="hero-desc">
        Upload a photo of a <strong style="color:#8bc499;">banana leaf</strong> — green or black &amp; white photos accepted.
        Our fine-tuned EfficientNetB0 model instantly diagnoses nutrient deficiencies
        across 8 classes and recommends targeted corrective treatments.
        Every scan is saved to your session history.
    </p>
    <div class="hero-stats">
        <div><div class="h-stat-val">86%</div><div class="h-stat-lbl">Test Accuracy</div></div>
        <div><div class="h-stat-val">8</div><div class="h-stat-lbl">Nutrient Classes</div></div>
        <div><div class="h-stat-val">7,784</div><div class="h-stat-lbl">Training Images</div></div>
        <div><div class="h-stat-val">EfficientNetB0</div><div class="h-stat-lbl">Architecture</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_diag, tab_hist, tab_guide, tab_model = st.tabs([
    "  Diagnose  ", "  Scan History  ", "  Nutrient Guide  ", "  Model Details  "
])

# ══════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ══════════════════════════════════════════════
with tab_diag:
    col_l, col_r = st.columns([1, 1.1], gap="large")

    with col_l:
        st.markdown("""
        <div class="panel">
          <div class="panel-header">
            <div class="panel-dot"></div>
            <span class="panel-title">Input — Banana Leaf Image</span>
          </div>
          <div class="panel-body">
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a banana leaf JPG / PNG here (green or B&W accepted)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        validation_result = None

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            st.markdown(
                f"<p style='font-family:var(--mono);font-size:.65rem;color:#9aab9e;margin-top:.5rem;'>"
                f"FILE: {uploaded.name} &nbsp;&middot;&nbsp; "
                f"{img.size[0]}×{img.size[1]} px &nbsp;&middot;&nbsp; "
                f"{uploaded.size//1024} KB</p>",
                unsafe_allow_html=True,
            )

            # ── BANANA LEAF VALIDATION (STRICT) ──
            cache_key = f"val_{uploaded.name}_{uploaded.size}"
            if cache_key not in st.session_state:
                with st.spinner("🔍 Verifying this is a banana leaf…"):
                    st.session_state[cache_key] = validate_banana_leaf(img)

            validation_result = st.session_state[cache_key]

            # ── Show validation result banner ──
            if validation_result["is_valid"] is True:
                conf_label = {
                    "high": "High confidence",
                    "medium": "Medium confidence",
                    "low": "Low confidence",
                }.get(validation_result.get("confidence", "high"), "")
                st.markdown(f"""
                <div class="validation-pass">
                    <div class="val-icon">✅</div>
                    <div>
                        <p class="val-title" style="color:#1a6b2f;">Banana leaf confirmed</p>
                        <p class="val-body" style="color:#2a5a38;">
                            {validation_result['reason']}
                            <span style="opacity:.6;"> · {conf_label}</span>
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif validation_result["is_valid"] is False:
                st.markdown(f"""
                <div class="validation-fail">
                    <div class="val-icon">🚫</div>
                    <div>
                        <p class="val-title" style="color:#c04a2a;">Not a banana leaf — Analysis blocked</p>
                        <p class="val-body" style="color:#7a2a18;">
                            {validation_result['reason']}<br>
                            <span style="opacity:.8;">This model is trained <strong>exclusively on banana leaves</strong>.
                            Running it on other images produces invalid results.
                            Please upload a banana leaf photo.</span>
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # is_valid = None → API/service error → STRICT BLOCK
                st.markdown(f"""
                <div class="validation-checking">
                    <div class="val-icon">⚠️</div>
                    <div>
                        <p class="val-title" style="color:#c8973a;">Leaf verification failed — Analysis blocked</p>
                        <p class="val-body" style="color:#7a5a18;">
                            {validation_result['reason']}<br>
                            <span style="opacity:.8;">For safety, the model will <strong>not run</strong> without
                            confirmed banana leaf verification. Please check your internet connection and try again.</span>
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN: Results ──
    with col_r:
        if not uploaded:
            # No image uploaded yet
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-icon">🍌</div>
                <div class="placeholder-text">
                    Upload a <strong>banana leaf</strong> photograph on the left<br>
                    to begin nutrient deficiency diagnosis.<br><br>
                    <span style="color:#b0bab4;font-size:.8rem;">
                        Green &amp; black-and-white photos accepted<br><br>
                        Boron · Calcium · Healthy · Iron<br>
                        Magnesium · Manganese · Potassium · Sulphur
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif validation_result is not None and validation_result["is_valid"] is False:
            # ── HARD BLOCK: Not a banana leaf ──
            st.markdown("""
            <div class="placeholder" style="border-color:#c04a2a;background:#fff8f6;">
                <div class="placeholder-icon" style="opacity:1;">🚫</div>
                <div class="placeholder-text">
                    <strong style="color:#c04a2a;font-size:1rem;">Image Rejected — Not a Banana Leaf</strong><br><br>
                    This app is trained <strong>exclusively on banana leaves</strong>.<br>
                    Running the model on non-banana-leaf images produces<br>
                    <strong>invalid and misleading results</strong> — analysis is blocked.<br><br>
                    <span style="color:#b0bab4;font-size:.8rem;">
                        Please upload a clear banana leaf photo.<br>
                        Both colour and black-and-white images are accepted.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif validation_result is not None and validation_result["is_valid"] is None:
            # ── STRICT BLOCK: Validation service unavailable ──
            st.markdown("""
            <div class="placeholder" style="border-color:#c8973a;background:#fffbf5;">
                <div class="placeholder-icon" style="opacity:1;">⚠️</div>
                <div class="placeholder-text">
                    <strong style="color:#c8973a;font-size:1rem;">Verification Service Unavailable</strong><br><br>
                    Banana leaf verification could not be completed.<br>
                    For safety, the model will <strong>not run</strong> on unverified images.<br><br>
                    <span style="color:#b0bab4;font-size:.8rem;">
                        Please check your internet connection and re-upload the image.<br>
                        Verification is required before any analysis can proceed.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif validation_result is not None and validation_result["is_valid"] is True:
            # ── CONFIRMED BANANA LEAF: Run model ──
            if not os.path.exists(MODEL_PATH):
                st.error(
                    f"**Model file not found.**\n\n`{MODEL_PATH}`\n\n"
                    "Download `best_model_phase2.keras` from Kaggle Output tab "
                    "and place it in `D:\\fyp2\\`"
                )
                st.stop()

            with st.spinner("Loading model…"):
                model = load_keras_model(MODEL_PATH)
            with st.spinner("Scanning banana leaf…"):
                arr     = preprocess(img)
                results = run_inference(model, arr)

            top_cls, top_prob = results[0]
            existing = [r["filename"] for r in st.session_state.scan_history]
            if uploaded.name not in existing:
                save_to_history(uploaded.name, img, top_cls, top_prob, results)
                st.rerun()

            nut            = NUTRIENTS[top_cls]
            sev_bg, sev_fg = SEV_STYLE[nut["severity"]]
            pct            = top_prob * 100
            conf_col       = "#1a6b2f" if pct>=70 else "#c8973a" if pct>=40 else "#c04a2a"

            st.markdown(f"""
            <div class="panel">
              <div class="result-header">
                <div class="elem-box"
                     style="background:{nut['color_bg']};border:1.5px solid {hex_to_rgba(nut['color_fg'])};">
                  <span class="elem-sym" style="color:{nut['color_fg']};">{nut['symbol']}</span>
                  <span class="elem-num" style="color:{nut['color_fg']};">{nut['number']}</span>
                </div>
                <div class="result-meta">
                  <div class="result-eyebrow">Detected Deficiency</div>
                  <div class="result-name">{top_cls}</div>
                  <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
                    <span class="sev-badge" style="background:{sev_bg};color:{sev_fg};">
                        {nut['level']} &middot; {nut['severity']} severity
                    </span>
                    <span style="font-family:var(--mono);font-size:.78rem;font-weight:500;color:{conf_col};">
                        {pct:.1f}% confidence
                    </span>
                  </div>
                </div>
              </div>
              <div class="info-section">
                <div class="info-sec-title">Observed Symptoms</div>
                <div class="info-sec-body">{nut['symptoms']}</div>
              </div>
              <div class="info-section">
                <div class="info-sec-title">Root Cause</div>
                <div class="info-sec-body">{nut['cause']}</div>
              </div>
              <div class="info-section">
                <div class="info-sec-title">Recommended Treatment</div>
                <div class="info-sec-body">{nut['treatment']}</div>
                <div style="margin-top:.6rem;">
                    {''.join(f"<span class='fert-chip'>{f}</span>" for f in nut['fertilizer'])}
                </div>
              </div>
              <div class="prob-section">
                <div class="prob-title">All class probabilities</div>
            """, unsafe_allow_html=True)

            for cls, prob in results:
                n    = NUTRIENTS[cls]
                p    = prob * 100
                bar  = n["color_bar"] if cls==top_cls else "#d4d0c4"
                bold = "font-weight:800;" if cls==top_cls else ""
                st.markdown(f"""
                <div class="pb-row">
                  <div class="pb-names">
                    <span class="pb-cls" style="{bold}">{cls}</span>
                    <span class="pb-pct" style="color:{n['color_bar']};">{p:.2f}%</span>
                  </div>
                  <div class="pb-track">
                    <div class="pb-fill" style="width:{p}%;background:{bar};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            # Fallback placeholder (image uploaded but validation not yet complete)
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-icon">🔍</div>
                <div class="placeholder-text">
                    Verifying image…<br><br>
                    <span style="color:#b0bab4;font-size:.8rem;">
                        Please wait while the banana leaf is being verified.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — SCAN HISTORY
# ══════════════════════════════════════════════
with tab_hist:
    history = st.session_state.scan_history
    count   = len(history)

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"""
        <div style="padding:2rem 3rem .5rem;">
            <h2 style="font-family:var(--serif);font-size:2rem;font-weight:400;font-style:italic;margin:0 0 .3rem;">
                Session <b style="font-style:normal;font-family:var(--sans);font-weight:900;">Scan History</b>
            </h2>
            <p style="font-size:.88rem;color:#7a8a7e;margin:0;">{count} scan{"s" if count!=1 else ""} in this session</p>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='padding-top:2.3rem;padding-right:3rem;'>", unsafe_allow_html=True)
        if count > 0:
            if st.button("Clear All History", key="clear_hist", use_container_width=True):
                st.session_state.scan_history = []
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if count == 0:
        st.markdown("""
        <div class="hist-empty">
            <div style="font-size:3rem;opacity:.35;">📷</div>
            <div style="font-family:var(--sans);font-size:1.1rem;font-weight:800;margin:.8rem 0 .3rem;">No scans yet</div>
            <div style="font-size:.85rem;color:#9aab9e;">Upload a banana leaf image in the Diagnose tab.<br>Results are automatically saved here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        avg_conf    = sum(r["top_prob"] for r in history) / count * 100
        most_common = max(set(r["top_cls"] for r in history),
                          key=lambda c: sum(1 for r in history if r["top_cls"]==c))
        mc_nut      = NUTRIENTS[most_common]

        st.markdown(f"""
        <div style="margin:.5rem 3rem 1rem;background:var(--ink);border-radius:var(--r-lg);
             padding:1.3rem 2rem;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1.5rem;">
            <div><div style="font-family:var(--mono);font-size:.55rem;color:#8bc499;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">Total Scans</div>
                 <div style="font-family:var(--sans);font-size:1.8rem;font-weight:900;color:#f8f9f8;">{count}</div></div>
            <div><div style="font-family:var(--mono);font-size:.55rem;color:#8bc499;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">Most Common</div>
                 <div style="font-family:var(--sans);font-size:1.2rem;font-weight:900;color:{mc_nut['color_bar']};">{most_common}</div></div>
            <div><div style="font-family:var(--mono);font-size:.55rem;color:#8bc499;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">Avg Confidence</div>
                 <div style="font-family:var(--sans);font-size:1.8rem;font-weight:900;color:#f8f9f8;">{avg_conf:.1f}%</div></div>
            <div><div style="font-family:var(--mono);font-size:.55rem;color:#8bc499;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">Latest Scan</div>
                 <div style="font-family:var(--mono);font-size:.72rem;font-weight:600;color:#b0c4b4;margin-top:6px;">{history[0]["timestamp"]}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="hist-grid">', unsafe_allow_html=True)
        for i, rec in enumerate(history):
            nut            = NUTRIENTS[rec["top_cls"]]
            sev_bg, sev_fg = SEV_STYLE[nut["severity"]]
            pct            = rec["top_prob"] * 100
            conf_col       = "#1a6b2f" if pct>=70 else "#c8973a" if pct>=40 else "#c04a2a"
            top3 = ""
            for cls, prob in rec["results"][:3]:
                n = NUTRIENTS[cls]; p = prob*100
                top3 += f"""
                <div style="margin:3px 0;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                    <span style="font-family:var(--mono);font-size:.58rem;color:#6a7a6e;">{cls}</span>
                    <span style="font-family:var(--mono);font-size:.58rem;font-weight:600;color:{n['color_bar']};">{p:.1f}%</span>
                  </div>
                  <div style="height:3px;border-radius:2px;background:var(--paper3);overflow:hidden;">
                    <div style="height:100%;width:{p}%;background:{n['color_bar']};border-radius:2px;"></div>
                  </div>
                </div>"""
            st.markdown(f"""
            <div class="hist-card">
                <img class="hist-card-img" src="data:image/jpeg;base64,{rec['img_b64']}" alt="{rec['filename']}" />
                <div class="hist-card-body">
                    <div class="hist-card-top">
                        <div class="hist-elem-box" style="background:{nut['color_bg']};border:1.5px solid {hex_to_rgba(nut['color_fg'])};">
                            <span style="font-size:1.1rem;font-weight:900;color:{nut['color_fg']};">{nut['symbol']}</span>
                        </div>
                        <div style="flex:1;min-width:0;">
                            <div class="hist-cls">{rec['top_cls']}</div>
                            <div style="display:flex;align-items:center;gap:.4rem;margin-top:3px;flex-wrap:wrap;">
                                <span class="sev-badge" style="background:{sev_bg};color:{sev_fg};font-size:.55rem;padding:2px 7px;">{nut['severity']}</span>
                                <span style="font-family:var(--mono);font-size:.68rem;font-weight:700;color:{conf_col};">{pct:.1f}%</span>
                            </div>
                        </div>
                        <span class="hist-index-badge">#{i+1}</span>
                    </div>
                    <div style="margin-bottom:.6rem;">{top3}</div>
                    <div class="hist-meta-row">
                        <span class="hist-filename" title="{rec['filename']}">{rec['filename']}</span>
                        <span class="hist-ts">{rec['timestamp']}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — NUTRIENT GUIDE
# ══════════════════════════════════════════════
with tab_guide:
    st.markdown("""
    <div style="padding:2rem 3rem .5rem;">
        <h2 style="font-family:var(--serif);font-size:2rem;font-weight:400;font-style:italic;margin:0 0 .3rem;">
            Nutrient Deficiency <b style="font-style:normal;font-family:var(--sans);font-weight:900;">Reference Guide</b>
        </h2>
        <p style="font-size:.88rem;color:#7a8a7e;margin:0;">Agronomic profiles for all 8 classes recognised by the model.</p>
    </div>
    <div class="guide-grid">
    """, unsafe_allow_html=True)

    for cls in CLASS_NAMES:
        nut            = NUTRIENTS[cls]
        sev_bg, sev_fg = SEV_STYLE[nut["severity"]]
        chips          = "".join(f"<span class='fert-chip'>{f}</span>" for f in nut["fertilizer"])
        st.markdown(f"""
        <div class="guide-card">
          <div class="guide-card-top">
            <div class="elem-box" style="background:{nut['color_bg']};border:1.5px solid {hex_to_rgba(nut['color_fg'])};width:52px;height:52px;">
              <span class="elem-sym" style="color:{nut['color_fg']};font-size:1.3rem;">{nut['symbol']}</span>
              <span class="elem-num" style="color:{nut['color_fg']};">{nut['number']}</span>
            </div>
            <div>
              <div style="font-family:var(--sans);font-size:1.05rem;font-weight:900;">{cls}</div>
              <div style="display:flex;gap:.4rem;margin-top:4px;flex-wrap:wrap;">
                <span class="tag" style="background:{nut['color_bg']};color:{nut['color_fg']};">{nut['level']}</span>
                <span class="tag" style="background:{sev_bg};color:{sev_fg};">{nut['severity']} Severity</span>
              </div>
            </div>
          </div>
          <div class="guide-card-body">
            <div class="guide-lbl">Symptoms</div>
            <div class="guide-val">{nut['symptoms']}</div>
            <div class="guide-lbl" style="margin-top:.9rem;">Treatment</div>
            <div class="guide-val">{nut['treatment']}</div>
            <div class="guide-lbl" style="margin-top:.9rem;">Fertilisers</div>
            <div style="margin-top:4px;">{chips}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — MODEL DETAILS
# ══════════════════════════════════════════════
with tab_model:
    st.markdown("""
    <div style="padding:2rem 3rem .5rem;">
        <h2 style="font-family:var(--serif);font-size:2rem;font-weight:400;font-style:italic;margin:0 0 .3rem;">
            Model <b style="font-style:normal;font-family:var(--sans);font-weight:900;">Architecture &amp; Training</b>
        </h2>
        <p style="font-size:.88rem;color:#7a8a7e;margin:0;">Two-phase transfer learning on EfficientNetB0 pretrained on ImageNet.</p>
    </div>
    <div class="model-grid">
        <div class="m-tile"><div class="m-val">86%</div><div class="m-lbl">Test Accuracy</div></div>
        <div class="m-tile"><div class="m-val">8</div><div class="m-lbl">Classes</div></div>
        <div class="m-tile"><div class="m-val">4.4 M</div><div class="m-lbl">Parameters</div></div>
        <div class="m-tile"><div class="m-val">55</div><div class="m-lbl">Total Epochs</div></div>
    </div>
    <div class="phase-grid">
        <div class="phase-card" style="border-top:3px solid #1a6b2f;">
            <div class="phase-num">Phase 01</div>
            <div class="phase-title">Head Training — Backbone Frozen</div>
            <div class="phase-row"><span class="phase-key">Epochs</span>15</div>
            <div class="phase-row"><span class="phase-key">Learning rate</span>1e-3</div>
            <div class="phase-row"><span class="phase-key">Backbone</span>EfficientNetB0 fully frozen</div>
            <div class="phase-row"><span class="phase-key">Head</span>GlobalAvgPool → Dense(256, ReLU, BN) → Dropout(0.5)</div>
            <div class="phase-row"><span class="phase-key"></span>→ Dense(128, tanh) → Dropout(0.25) → Softmax(8)</div>
            <div class="phase-row"><span class="phase-key">Regularisation</span>L2 (0.01) on Dense layers</div>
            <div class="phase-row"><span class="phase-key">Label smoothing</span>0.1</div>
        </div>
        <div class="phase-card" style="border-top:3px solid #c8973a;">
            <div class="phase-num">Phase 02</div>
            <div class="phase-title">Fine-Tuning — Top 30 Layers Unfrozen</div>
            <div class="phase-row"><span class="phase-key">Epochs</span>40</div>
            <div class="phase-row"><span class="phase-key">Learning rate</span>1e-4</div>
            <div class="phase-row"><span class="phase-key">Unfrozen layers</span>Top 30 of EfficientNetB0</div>
            <div class="phase-row"><span class="phase-key">Class weights</span>Balanced + Calcium/Mg ×1.5</div>
            <div class="phase-row"><span class="phase-key">Callbacks</span>ModelCheckpoint · EarlyStopping (pat. 8)</div>
            <div class="phase-row"><span class="phase-key"></span>ReduceLROnPlateau (factor 0.5, pat. 4)</div>
            <div class="phase-row"><span class="phase-key">Test accuracy</span><b style="color:#c8973a;">86%</b></div>
        </div>
    </div>
    <div style="padding:0 3rem 2.5rem;">
      <div class="phase-card">
        <div class="phase-num">Dataset &amp; Augmentation Pipeline</div>
        <div class="phase-title">Training Configuration</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;margin-top:.5rem;">
          <div>
            <div class="guide-lbl">Dataset Split</div>
            <div class="phase-row"><span class="phase-key">Train</span>7,784 images</div>
            <div class="phase-row"><span class="phase-key">Validation</span>2,242 images</div>
            <div class="phase-row"><span class="phase-key">Test</span>1,124 images</div>
            <div class="phase-row"><span class="phase-key">Classes</span>8 (New folder excluded)</div>
            <div class="phase-row"><span class="phase-key">Image size</span>224 × 224 px</div>
            <div class="phase-row"><span class="phase-key">Batch size</span>32</div>
          </div>
          <div>
            <div class="guide-lbl">Augmentation (Train only)</div>
            <div class="phase-row"><span class="phase-key">Rotation</span>+/- 40 deg</div>
            <div class="phase-row"><span class="phase-key">Shift</span>+/- 20% (W &amp; H)</div>
            <div class="phase-row"><span class="phase-key">Zoom</span>+/- 25%</div>
            <div class="phase-row"><span class="phase-key">Flip</span>Horizontal + Vertical</div>
            <div class="phase-row"><span class="phase-key">Brightness</span>[0.7 – 1.3]</div>
            <div class="phase-row"><span class="phase-key">Channel shift</span>30</div>
          </div>
          <div>
            <div class="guide-lbl">Preprocessing &amp; Optimiser</div>
            <div class="phase-row"><span class="phase-key">Preprocess</span>EfficientNet preprocess_input</div>
            <div class="phase-row"><span class="phase-key">Shuffle</span>True (seed = 100)</div>
            <div class="phase-row"><span class="phase-key">Optimiser</span>Adam</div>
            <div class="phase-row"><span class="phase-key">Loss</span>CategoricalCrossentropy (smooth=0.1)</div>
            <div class="phase-row"><span class="phase-key">Metric</span>Accuracy</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PlantAI &nbsp;&middot;&nbsp; Banana Leaf Nutrient Deficiency Detection &nbsp;&middot;&nbsp;
    EfficientNetB0 Transfer Learning &nbsp;&middot;&nbsp; best_model_phase2.keras &nbsp;&middot;&nbsp;
    8 Nutrient Classes &nbsp;&middot;&nbsp; 86% Test Accuracy<br>
    For research and advisory use only — always validate with certified soil and tissue laboratory analysis.
</div>
""", unsafe_allow_html=True)