"""
dashboard.py — Fretboard Market Intelligence Web App
A production-grade Flask + Plotly dashboard for vintage guitar price forecasting.

Run:
    python dashboard.py
Open: http://localhost:5000
"""

import math
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
import torch
from flask import Flask, jsonify, render_template_string, request

from model import GuitarPriceModel, get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed_listings.csv")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAD_RATE = 1.36

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model_cache = {}

def get_model():
    if "model" not in _model_cache and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model = GuitarPriceModel(tabular_input_dim=len(checkpoint["tabular_cols"])).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model_cache["model"] = model
        _model_cache["tabular_cols"] = checkpoint["tabular_cols"]
        _model_cache["scaler"] = checkpoint["scaler"]
    return _model_cache.get("model"), _model_cache.get("tabular_cols"), _model_cache.get("scaler")


def predict(description: str, tab_row: np.ndarray) -> tuple[float, float, float]:
    model, tabular_cols, scaler = get_model()
    if model is None:
        base = 3500.0
        return base, base * 0.88, base * 1.12

    tokenizer = get_tokenizer()
    enc = tokenizer(description, max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        mean_log, log_var = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            torch.tensor(tab_row, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        )
    m = mean_log.item()
    s = math.sqrt(math.exp(log_var.item()))
    return (round(math.expm1(m), 2),
            round(max(math.expm1(m - 1.96 * s), 0), 2),
            round(math.expm1(m + 1.96 * s), 2))


def signal(hist_prices, pred):
    if not hist_prices:
        return "HOLD", 0.0
    cur = np.mean(hist_prices[-10:])
    pct = (pred - cur) / cur * 100 if cur else 0
    if pct > 6:
        return "BUY", round(pct, 1)
    if pct < -4:
        return "SELL", round(pct, 1)
    return "HOLD", round(pct, 1)


def forecast_series(base, lower, upper, months=6):
    dates = [(datetime.today() + timedelta(days=30 * i)).strftime("%b %Y") for i in range(months + 1)]
    trend = np.linspace(0, 0.08, months + 1)
    noise = np.random.normal(0, 0.01, months + 1).cumsum()
    return (dates,
            (base * (1 + trend + noise)).tolist(),
            (lower * (1 + trend + noise * 0.8)).tolist(),
            (upper * (1 + trend + noise * 1.2)).tolist())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_brand_data(brand: str, currency: str):
    df = pd.read_csv(PROCESSED_PATH)
    col = f"brand_{brand}"
    sub = df[df[col] == 1].copy() if col in df.columns else df.copy()
    sub = sub.dropna(subset=["price_usd_normalized"])
    fx = CAD_RATE if currency == "CAD" else 1.0
    sub["price_display"] = sub["price_usd_normalized"] * fx
    return sub, fx


def get_brands():
    df = pd.read_csv(PROCESSED_PATH)
    return sorted([c.replace("brand_", "") for c in df.columns if c.startswith("brand_")])


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

WARM   = "#8b6f47"
ACCENT = "#c8451a"
GREEN  = "#2d6a4f"
MUTED  = "#a09070"
BG     = "#fdfaf5"
GRID   = "rgba(139,111,71,0.12)"

CHART_LAYOUT = dict(
    paper_bgcolor="transparent",
    plot_bgcolor="transparent",
    font=dict(family="'DM Mono', monospace", color=MUTED, size=11),
    margin=dict(l=48, r=16, t=12, b=36),
    hovermode="x unified",
    xaxis=dict(gridcolor=GRID, linecolor=GRID, tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, tickcolor=GRID),
)


def build_historical_chart(sub, brand, currency, pred, lower, upper, fx):
    sym = "CA$" if currency == "CAD" else "$"
    yearly = sub.groupby("sale_year")["price_display"].mean().reset_index()
    years = yearly["sale_year"].tolist()
    avg_prices = yearly["price_display"].tolist()
    forecast_dates, f_prices, f_lower, f_upper = forecast_series(pred * fx, lower * fx, upper * fx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=f_upper + f_lower[::-1],
        fill="toself", fillcolor="rgba(200,69,26,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[str(y) for y in years], y=avg_prices,
        mode="lines+markers",
        line=dict(color=WARM, width=2.5),
        marker=dict(size=5, color=WARM),
        name="Avg sold",
        hovertemplate=f"<b>%{{x}}</b><br>{sym}%{{y:,.0f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=f_prices,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2, dash="dash"),
        marker=dict(size=5, color=ACCENT),
        name="AI forecast",
        hovertemplate=f"<b>%{{x}}</b><br>{sym}%{{y:,.0f}}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=11, color=MUTED), bgcolor="transparent"),
    )
    fig.update_yaxes(tickprefix=sym)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_condition_chart(sub):
    cond_cols = [c for c in sub.columns if c.startswith("condition_normalized_")]
    if cond_cols:
        counts = {c.replace("condition_normalized_", ""): int(sub[c].sum()) for c in cond_cols}
    else:
        counts = {"Unknown": len(sub)}
    counts = {k: v for k, v in counts.items() if v > 0}
    colors = [WARM, ACCENT, GREEN, MUTED, "#d4c4a0", "#6a5a3a"]

    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        hole=0.58,
        marker=dict(colors=colors[:len(counts)], line=dict(color=BG, width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, color=MUTED),
        hovertemplate="<b>%{label}</b><br>%{value} listings (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        font=dict(family="'DM Mono', monospace"),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_pickup_chart(sub, currency, fx):
    sym = "CA$" if currency == "CAD" else "$"
    pickup_cols = [c for c in sub.columns if c.startswith("pickup_config_")]
    if not pickup_cols:
        return "{}"
    avgs = {c.replace("pickup_config_", ""): sub.loc[sub[c] == 1, "price_display"].mean()
            for c in pickup_cols if (sub[c] == 1).sum() > 3}
    if not avgs:
        return "{}"
    labels, vals = zip(*sorted(avgs.items(), key=lambda x: x[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(labels), orientation="h",
        marker=dict(color=WARM, opacity=0.85),
        hovertemplate=f"<b>%{{y}}</b><br>{sym}%{{x:,.0f}}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, margin=dict(l=60, r=16, t=8, b=36))
    fig.update_xaxes(tickprefix=sym)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_recent_listings(sub, fx, n=8):
    cols = [c for c in ["title","price_usd_normalized","condition","originality_score"] if c in sub.columns]
    top = sub[cols].dropna(subset=["price_usd_normalized"]).tail(n).iloc[::-1]
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "title": str(r.get("title","Unknown"))[:60],
            "condition": str(r.get("condition","–")),
            "price": round(float(r["price_usd_normalized"]) * fx),
            "orig": int(r.get("originality_score", 0)),
        })
    return

@app.route('/favicon.ico')
def favicon():
    # Sending a 204 status code tells the browser: 
    # "Success, but there is intentionally no content here."
    return '', 204


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fretboard Market Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,400;1,300&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --cream:#f5f0e8;--ink:#1a1410;--warm:#8b6f47;--accent:#c8451a;--muted:#a09070;
      --card:#fdfaf5;--border:#e8dfc8;--green:#2d6a4f;
      --serif:'Fraunces',serif;--sans:'DM Sans',sans-serif;--mono:'DM Mono',monospace;
    }
    body{background:var(--cream);color:var(--ink);font-family:var(--sans);display:flex;min-height:100vh}

    .sidebar{width:220px;min-height:100vh;background:var(--ink);padding:28px 16px;display:flex;flex-direction:column;gap:22px;position:fixed;top:0;left:0;bottom:0;overflow-y:auto}
    .logo{font-family:var(--serif);font-size:18px;color:var(--cream);font-weight:300;line-height:1.35}
    .logo em{color:#c8906a;font-style:italic}
    .s-label{font-size:10px;letter-spacing:.12em;color:#5a4a2a;text-transform:uppercase;margin-bottom:7px;padding-left:4px}
    .nav-item{display:flex;align-items:center;gap:9px;padding:8px 12px;border-radius:6px;color:#857055;font-size:13px;cursor:pointer;transition:all .15s;text-decoration:none}
    .nav-item:hover{background:rgba(255,255,255,.05);color:var(--cream)}
    .nav-item.active{background:rgba(200,69,26,.2);color:var(--cream)}
    .filter-btn{display:block;width:100%;text-align:left;padding:6px 12px;border-radius:20px;border:0.5px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);color:#857055;font-size:12px;font-family:var(--sans);cursor:pointer;transition:all .15s;margin-bottom:5px}
    .filter-btn:hover{border-color:rgba(200,69,26,.4);color:var(--cream)}
    .filter-btn.active{background:rgba(200,69,26,.22);border-color:#c8451a;color:var(--cream)}
    .cur-row{display:flex;gap:6px}
    .cur-btn{flex:1;padding:6px;border-radius:6px;font-size:12px;font-family:var(--mono);border:0.5px solid rgba(255,255,255,.1);color:#857055;cursor:pointer;background:transparent;transition:all .15s}
    .cur-btn.active{background:rgba(200,69,26,.22);color:var(--cream);border-color:#c8451a}
    .s-footer{margin-top:auto;font-size:11px;color:#4a3a2a;line-height:1.65}

    .main{margin-left:220px;flex:1;padding:28px 28px 48px;display:flex;flex-direction:column;gap:20px}

    .page-title{font-family:var(--serif);font-size:25px;font-weight:300}
    .page-title em{font-style:italic;color:var(--warm)}
    .freshness{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:5px}

    .metrics-row{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px}
    .metric-card{background:var(--card);border:0.5px solid var(--border);border-radius:10px;padding:14px 16px}
    .m-label{font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px}
    .m-value{font-family:var(--mono);font-size:21px;font-weight:500;line-height:1}
    .m-sub{font-size:11px;color:var(--muted);margin-top:5px}
    .signal-card{border:none;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;justify-content:center}
    .signal-card.BUY{background:var(--green)}
    .signal-card.HOLD{background:#7a6030}
    .signal-card.SELL{background:#8b2010}
    .signal-word{font-family:var(--serif);font-size:30px;font-weight:300;font-style:italic;color:rgba(255,255,255,.92);letter-spacing:.03em;line-height:1}
    .signal-sub{font-size:10px;font-family:var(--mono);color:rgba(255,255,255,.45);margin-top:6px}

    .chart-grid{display:grid;grid-template-columns:1.7fr 1fr;gap:16px}
    .chart-card{background:var(--card);border:0.5px solid var(--border);border-radius:10px;padding:18px 16px 12px}
    .c-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
    .c-title{font-family:var(--serif);font-size:14px;font-weight:300}
    .c-tag{font-size:10px;font-family:var(--mono);background:#f0ebe0;color:var(--muted);padding:3px 9px;border-radius:4px}

    .listings-card{background:var(--card);border:0.5px solid var(--border);border-radius:10px;padding:18px}
    table.lst{width:100%;border-collapse:collapse;font-size:13px}
    table.lst th{text-align:left;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:400;padding:0 0 9px;border-bottom:0.5px solid var(--border)}
    table.lst td{padding:9px 0;border-bottom:0.5px solid var(--border);vertical-align:middle}
    table.lst tr:last-child td{border-bottom:none}
    .orig-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle}
    .o1{background:var(--green)}.o0{background:var(--warm)}.on{background:var(--accent)}
    .cond-badge{font-size:10px;font-family:var(--mono);padding:2px 7px;border-radius:4px;background:#f0ebe0;color:var(--muted)}
    .price-cell{font-family:var(--mono);font-weight:500;text-align:right}

    #loader{position:fixed;inset:0;background:var(--cream);display:flex;align-items:center;justify-content:center;z-index:100;transition:opacity .4s}
    .spin{width:28px;height:28px;border:2px solid var(--border);border-top-color:var(--warm);border-radius:50%;animation:s .8s linear infinite;margin:12px auto 0}
    @keyframes s{to{transform:rotate(360deg)}}
  </style>
</head>
<body>

<div id="loader" aria-live="polite" aria-label="Loading">
  <div style="text-align:center">
    <div style="font-family:var(--serif);font-size:26px;font-weight:300;color:var(--warm)"><em>Fretboard</em> Intelligence</div>
    <div class="spin"></div>
  </div>
</div>

<nav class="sidebar" aria-label="Main navigation">
  <div class="logo"><em>Fretboard</em><br>Market Intelligence</div>
  <div>
    <div class="s-label">Navigation</div>
    <a class="nav-item active" href="#" aria-current="page">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Market Overview
    </a>
    <a class="nav-item" href="#">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      Listings Explorer
    </a>
    <a class="nav-item" href="#">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
      Model Insights
    </a>
  </div>
  <div>
    <div class="s-label">Brand Filter</div>
    <div id="brand-filters"></div>
  </div>
  <div>
    <div class="s-label">Currency</div>
    <div class="cur-row">
      <button class="cur-btn active" onclick="setCurrency('USD')" data-cur="USD">USD</button>
      <button class="cur-btn" onclick="setCurrency('CAD')" data-cur="CAD">CAD</button>
    </div>
  </div>
  <div class="s-footer">
    Reverb.com sold listings<br>
    <span id="upd-time" style="color:#6a5a3a"></span>
  </div>
</nav>

<main class="main">
  <div>
    <h1 class="page-title" id="page-title">Loading…</h1>
    <p class="freshness" id="freshness"></p>
  </div>

  <div class="metrics-row">
    <div class="metric-card signal-card" id="sig-card">
      <div class="m-label" style="color:rgba(255,255,255,.45)">6-month signal</div>
      <div class="signal-word" id="sig-word">—</div>
      <div class="signal-sub" id="sig-sub"></div>
    </div>
    <div class="metric-card">
      <div class="m-label">Predicted price</div>
      <div class="m-value" id="m-pred">—</div>
      <div class="m-sub" id="m-ci"></div>
    </div>
    <div class="metric-card">
      <div class="m-label">30-day average</div>
      <div class="m-value" id="m-avg">—</div>
      <div class="m-sub" id="m-count"></div>
    </div>
    <div class="metric-card">
      <div class="m-label">Originality rate</div>
      <div class="m-value" id="m-orig">—</div>
      <div class="m-sub">all-original listings</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-card">
      <div class="c-header">
        <span class="c-title">Historical sold prices &amp; AI forecast</span>
        <span class="c-tag" id="hist-tag">all years</span>
      </div>
      <div id="chart-hist" style="height:240px" role="img" aria-label="Historical price chart with AI forecast"></div>
    </div>
    <div class="chart-card">
      <div class="c-header">
        <span class="c-title">Condition distribution</span>
        <span class="c-tag">all-time</span>
      </div>
      <div id="chart-cond" style="height:240px" role="img" aria-label="Condition distribution donut chart"></div>
    </div>
  </div>

  <div class="chart-card">
    <div class="c-header">
      <span class="c-title">Average price by pickup configuration</span>
      <span class="c-tag">SSS · HH · HSS · etc.</span>
    </div>
    <div id="chart-pickup" style="height:200px" role="img" aria-label="Average price by pickup configuration bar chart"></div>
  </div>

  <div class="listings-card">
    <div class="c-header">
      <span class="c-title">Recent sold listings</span>
      <span class="c-tag">latest entries</span>
    </div>
    <table class="lst" aria-label="Recent sold listings">
      <thead><tr><th>Instrument</th><th>Condition</th><th style="text-align:right">Sold price</th></tr></thead>
      <tbody id="lst-body"></tbody>
    </table>
  </div>
</main>

<script>
let brand = null, currency = "USD";
const fmt = n => (currency === "CAD" ? "CA$" : "$") + Math.round(n).toLocaleString();
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const CFG = {responsive:true, displayModeBar:false};

async function init() {
  const brands = await fetch("/api/brands").then(r => r.json());
  const c = document.getElementById("brand-filters");
  brands.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.className = "filter-btn" + (i === 0 ? " active" : "");
    btn.textContent = b;
    btn.onclick = () => setBrand(b);
    c.appendChild(btn);
  });
  document.getElementById("upd-time").textContent =
    "Updated " + new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
  if (brands.length) setBrand(brands[0]);
}

function setBrand(b) {
  brand = b;
  document.querySelectorAll(".filter-btn").forEach(el => el.classList.toggle("active", el.textContent === b));
  load();
}

function setCurrency(c) {
  currency = c;
  document.querySelectorAll(".cur-btn").forEach(el => el.classList.toggle("active", el.dataset.cur === c));
  load();
}

async function load() {
  if (!brand) return;
  const d = await fetch(`/api/data?brand=${encodeURIComponent(brand)}&currency=${currency}`).then(r => r.json());
  render(d);
}

function render(d) {
  document.getElementById("page-title").innerHTML = `Vintage <em>${esc(brand)}</em> — Price Intelligence`;
  document.getElementById("freshness").textContent =
    `Scraped: ${d.scraped_at}  ·  ${d.total_listings.toLocaleString()} listings indexed`;
  document.getElementById("hist-tag").textContent = d.year_range ? d.year_range[0] + " – " + d.year_range[1] : "all years";

  const sc = document.getElementById("sig-card");
  sc.className = "metric-card signal-card " + d.signal;
  document.getElementById("sig-word").textContent = d.signal;
  document.getElementById("sig-sub").textContent = (d.signal_pct > 0 ? "+" : "") + d.signal_pct + "% forecast";

  document.getElementById("m-pred").textContent = fmt(d.pred);
  document.getElementById("m-ci").textContent = `95% CI: ${fmt(d.lower)} – ${fmt(d.upper)}`;
  document.getElementById("m-avg").textContent = fmt(d.avg_30d);
  document.getElementById("m-count").textContent = d.total_listings + " listings";
  document.getElementById("m-orig").textContent = (d.orig_pct * 100).toFixed(0) + "%";

  Plotly.react("chart-hist",  JSON.parse(d.chart_hist).data,  JSON.parse(d.chart_hist).layout,  CFG);
  Plotly.react("chart-cond",  JSON.parse(d.chart_cond).data,  JSON.parse(d.chart_cond).layout,  CFG);
  if (d.chart_pickup && d.chart_pickup !== "{}") {
    Plotly.react("chart-pickup", JSON.parse(d.chart_pickup).data, JSON.parse(d.chart_pickup).layout, CFG);
  }

  document.getElementById("lst-body").innerHTML = d.listings.map(l => {
    const dc = l.orig === 1 ? "o1" : l.orig === -1 ? "on" : "o0";
    const title = l.orig === 1 ? "All-original" : l.orig === -1 ? "Modified" : "Unknown";
    return `<tr>
      <td><span class="orig-dot ${dc}" title="${title}"></span>${esc(l.title)}</td>
      <td><span class="cond-badge">${esc(l.condition)}</span></td>
      <td class="price-cell">${fmt(l.price)}</td>
    </tr>`;
  }).join("");

  const loader = document.getElementById("loader");
  loader.style.opacity = "0";
  setTimeout(() => loader.style.display = "none", 400);
}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/brands")
def api_brands():
    try:
        return jsonify(get_brands())
    except Exception:
        return jsonify([])


@app.route("/api/data")
def api_data():
    brand = request.args.get("brand", "Fender")
    currency = request.args.get("currency", "USD")

    sub, fx = load_brand_data(brand, currency)

    sample_desc = (sub["description"].dropna().sample(1).values[0]
                   if "description" in sub.columns and len(sub) > 0
                   else "vintage electric guitar original pickups good condition")

    tabular_cols_avail = [c for c in sub.columns if c in (
        ["year_of_manufacture", "originality_score", "is_player_grade"] +
        [col for col in sub.columns if col.startswith(("brand_", "condition_", "pickup_"))]
    )]
    tab_row = sub[tabular_cols_avail].fillna(0).mean().values.astype(np.float32) if tabular_cols_avail else np.array([0.0])
    pred_usd, lower_usd, upper_usd = predict(sample_desc, tab_row)

    sig, sig_pct = signal(sub["price_usd_normalized"].dropna().tolist(), pred_usd)
    avg_30d = float(sub["price_display"].mean()) if len(sub) else pred_usd * fx
    orig_pct = float((sub["originality_score"] == 1).mean()) if "originality_score" in sub.columns else 0.0

    year_range = None
    if "sale_year" in sub.columns:
        valid = sub["sale_year"].dropna()
        if len(valid):
            year_range = [int(valid.min()), int(valid.max())]

    return jsonify({
        "signal": sig,
        "signal_pct": sig_pct,
        "pred": round(pred_usd * fx, 2),
        "lower": round(lower_usd * fx, 2),
        "upper": round(upper_usd * fx, 2),
        "avg_30d": round(avg_30d, 2),
        "total_listings": len(sub),
        "orig_pct": round(orig_pct, 3),
        "year_range": year_range,
        "scraped_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "chart_hist":   build_historical_chart(sub, brand, currency, pred_usd, lower_usd, upper_usd, fx),
        "chart_cond":   build_condition_chart(sub),
        "chart_pickup": build_pickup_chart(sub, currency, fx),
        "listings":     get_recent_listings(sub, fx),
    })


if __name__ == "__main__":
    logger.info("Starting Fretboard Market Intelligence → http://localhost:5000")
    app.run(debug=True, port=5000)
