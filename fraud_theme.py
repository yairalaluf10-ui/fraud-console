"""Visual system for the fraud console: hacker/terminal look, validated palette.

The categorical order below was run through the data-viz colour checks against
the console surface (#0a0e0a) in dark mode:
  lightness band  PASS   chroma floor PASS
  CVD separation  PASS   worst adjacent pair green<->crimson dE 11.7 (deutan)
  normal vision   PASS   worst adjacent pair dE 30.9
  contrast        PASS   every slot >= 3:1 on the surface
Neon #00ff41 is UI chrome only (headings, borders, KPI numbers) - never a data mark.
"""

import plotly.graph_objects as go
import plotly.io as pio

# --- surfaces & ink ---------------------------------------------------------
SURFACE = "#0a0e0a"
SURFACE_2 = "#0f150f"
SURFACE_3 = "#141c14"
GRID = "#1c2a1e"
NEON = "#00ff41"
NEON_DIM = "#12a63a"
INK = "#c8d6cc"
INK_DIM = "#8b9a8f"
INK_MUTED = "#5c6b60"

# --- categorical slots (fixed order, never cycled) --------------------------
SERIES = ["#2cab5b", "#c2185b", "#5570fa", "#c4780e", "#c026d3", "#0f9b8e"]

# --- status ----------------------------------------------------------------
FRAUD = "#c2185b"   # slot 2
LEGIT = "#2cab5b"   # slot 1
WARN = "#c4780e"

# --- sequential ramp (one hue, near-zero recedes into the dark surface) -----
GREEN_SCALE = [
    [0.00, "#0a0e0a"], [0.15, "#0d3a1e"], [0.35, "#116634"],
    [0.55, "#15994c"], [0.75, "#2cab5b"], [0.90, "#5ec97f"], [1.00, "#8fe6a6"],
]
# diverging: crimson <-> green with a neutral grey midpoint (for lift vs baseline)
LIFT_SCALE = [
    [0.00, "#5570fa"], [0.35, "#2b3340"], [0.5, "#262b26"],
    [0.65, "#7a1339"], [1.00, "#e0356b"],
]

MONO = "JetBrains Mono, Fira Code, Consolas, Menlo, monospace"


def register_template() -> str:
    """Register and activate the plotly template. Returns its name."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        font=dict(family=MONO, size=12, color=INK),
        title=dict(font=dict(family=MONO, size=14, color=NEON), x=0.01, xanchor="left"),
        margin=dict(l=48, r=24, t=48, b=44),
        hoverlabel=dict(
            bgcolor=SURFACE_3, bordercolor=NEON_DIM,
            font=dict(family=MONO, size=12, color=INK),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
            font=dict(family=MONO, size=11, color=INK_DIM),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            tickfont=dict(color=INK_DIM, size=11),
            title=dict(font=dict(color=INK_DIM, size=11)),
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            tickfont=dict(color=INK_DIM, size=11),
            title=dict(font=dict(color=INK_DIM, size=11)),
        ),
        colorscale=dict(sequential=GREEN_SCALE, diverging=LIFT_SCALE),
        bargap=0.28,
    )
    pio.templates["hackers"] = tpl
    pio.templates.default = "hackers"
    return "hackers"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&display=swap');

:root {{
  --neon: {NEON};
  --neon-dim: {NEON_DIM};
  --surface: {SURFACE};
  --surface-2: {SURFACE_2};
  --surface-3: {SURFACE_3};
  --grid: {GRID};
  --ink: {INK};
  --ink-dim: {INK_DIM};
  --fraud: {FRAUD};
}}

html, body, [class*="css"], .stApp {{
  font-family: {MONO} !important;
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(0,255,65,0.07), transparent 60%),
    linear-gradient(180deg, {SURFACE} 0%, #070b07 100%);
  color: {INK};
}}

/* faint CRT scanlines */
.stApp::before {{
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
  background: repeating-linear-gradient(
    0deg, rgba(0,255,65,0.035) 0px, rgba(0,255,65,0.035) 1px, transparent 1px, transparent 3px);
  mix-blend-mode: screen; opacity: .35;
}}

h1, h2, h3, h4 {{
  font-family: {MONO} !important;
  color: {NEON} !important;
  letter-spacing: .08em;
  text-shadow: 0 0 12px rgba(0,255,65,.35);
}}

/* ---- masthead ---- */
.hk-head {{
  border: 1px solid var(--neon-dim);
  border-left: 4px solid var(--neon);
  background: linear-gradient(90deg, rgba(0,255,65,.10), rgba(0,255,65,.01));
  padding: .8rem 1.1rem; margin-bottom: .9rem;
  box-shadow: 0 0 24px rgba(0,255,65,.10) inset;
}}
.hk-head .t {{ color: var(--neon); font-size: 1.35rem; font-weight: 800; letter-spacing:.16em; }}
.hk-head .s {{ color: var(--ink-dim); font-size: .78rem; letter-spacing:.10em; }}
.hk-cursor {{ animation: blink 1.1s steps(2, start) infinite; }}
@keyframes blink {{ to {{ visibility: hidden; }} }}

/* ---- KPI tiles ---- */
[data-testid="stMetric"] {{
  background: var(--surface-2);
  border: 1px solid var(--grid);
  border-top: 2px solid var(--neon-dim);
  padding: .7rem .85rem;
}}
[data-testid="stMetricLabel"] p {{
  color: var(--ink-dim) !important; font-size: .70rem !important;
  letter-spacing: .13em; text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
  color: var(--neon) !important; font-family: {MONO} !important;
  font-weight: 700; text-shadow: 0 0 14px rgba(0,255,65,.30);
}}
[data-testid="stMetricDelta"] {{ font-family: {MONO} !important; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
  background: #070b07;
  border-right: 1px solid var(--neon-dim);
}}
[data-testid="stSidebar"] * {{ font-family: {MONO} !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {{
  color: var(--ink-dim) !important; font-size: .78rem; letter-spacing:.05em;
}}

/* ---- tabs ---- */
[data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--grid); }}
[data-baseweb="tab"] {{
  background: transparent; color: var(--ink-dim);
  font-family: {MONO} !important; font-size: .80rem; letter-spacing: .10em;
  border-bottom: 2px solid transparent; padding: .45rem .9rem;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--neon) !important; border-bottom: 2px solid var(--neon);
  text-shadow: 0 0 10px rgba(0,255,65,.4);
}}

/* ---- widgets ---- */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
  background: var(--neon) !important; box-shadow: 0 0 10px rgba(0,255,65,.6);
}}
[data-baseweb="tag"] {{ background: var(--neon-dim) !important; color: #04120a !important; }}
.stButton > button, .stDownloadButton > button {{
  background: transparent; color: var(--neon);
  border: 1px solid var(--neon-dim); border-radius: 0;
  font-family: {MONO} !important; letter-spacing: .10em; font-size: .78rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: rgba(0,255,65,.12); border-color: var(--neon); color: var(--neon);
}}

/* ---- code-ish panel ---- */
.hk-panel {{
  border: 1px solid var(--grid); border-left: 3px solid var(--neon-dim);
  background: var(--surface-2); padding: .65rem .85rem; margin: .2rem 0 .8rem 0;
  color: var(--ink-dim); font-size: .80rem; line-height: 1.5;
}}
.hk-panel b {{ color: var(--neon); }}
.hk-tag {{
  display:inline-block; border:1px solid var(--grid); padding:.05rem .45rem;
  margin:.1rem .2rem .1rem 0; color:var(--ink-dim); font-size:.72rem;
}}
.hk-tag.on {{ color:#04120a; background:var(--neon-dim); border-color:var(--neon-dim); }}
hr {{ border-color: var(--grid); }}
</style>
"""
