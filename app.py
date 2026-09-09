"""FRAUD // CONSOLE - an exploratory dashboard for credit-card fraud signals.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fraud_analytics as fa
import fraud_theme as th

DATA_PATH = "credit_card_fraud_10k.csv"

st.set_page_config(
    page_title="FRAUD // CONSOLE",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)
TEMPLATE = th.register_template()
st.markdown(th.CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# cached data / models
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data(path: str) -> pd.DataFrame:
    return fa.load_data(path)


@st.cache_data(show_spinner="mining segments…")
def get_segments(df: pd.DataFrame, min_support: int, depth: int) -> pd.DataFrame:
    return fa.mine_segments(df, min_support=min_support, max_depth=depth)


@st.cache_data(show_spinner="training models…")
def get_models(df: pd.DataFrame) -> dict:
    return fa.train_and_score(df)


# --------------------------------------------------------------------------- #
# chart helpers - one house style, applied everywhere
# --------------------------------------------------------------------------- #
def _fig(height: int = 320, **layout) -> go.Figure:
    # the template has to be named on the figure itself: streamlit ships the
    # figure spec to plotly.js, which knows nothing about our python default.
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height, **layout)
    return fig


def baseline_line(fig: go.Figure, value: float, label: str,
                  position: str = "top right") -> None:
    fig.add_hline(
        y=value, line=dict(color=th.FRAUD, width=1.5, dash="dot"),
        annotation_text=label, annotation_position=position,
        annotation_font=dict(color=th.FRAUD, size=11, family=th.MONO),
    )


def rate_bars(table: pd.DataFrame, base: float, title: str,
              x: str = "level", height: int = 330) -> go.Figure:
    """Fraud rate per level, with the portfolio base rate drawn in."""
    labels = table[x].astype(str)
    fig = _fig(height, title=title)
    fig.add_bar(
        x=labels, y=table["fraud_rate"],
        marker=dict(color=th.SERIES[0], cornerradius=4,
                    line=dict(color=th.SURFACE, width=2)),
        customdata=np.stack([table["transactions"], table["frauds"], table["lift"]], -1),
        hovertemplate=(
            "<b>%{x}</b><br>fraud rate %{y:.2%}<br>"
            "%{customdata[1]:,} frauds / %{customdata[0]:,} txns<br>"
            "lift %{customdata[2]:.2f}x<extra></extra>"
        ),
        text=[f"{v:.1%}" for v in table["fraud_rate"]],
        textposition="outside",
        textfont=dict(color=th.INK_DIM, size=11, family=th.MONO),
        showlegend=False,
    )
    baseline_line(fig, base, f"base {base:.2%}")
    fig.update_yaxes(tickformat=".1%", title="fraud rate")
    fig.update_xaxes(title=None)
    top = float(table["fraud_rate"].max()) if len(table) else 0.01
    fig.update_yaxes(range=[0, max(top * 1.28, base * 2)])
    return fig


def volume_bars(table: pd.DataFrame, title: str, height: int = 200) -> go.Figure:
    fig = _fig(height, title=title)
    fig.add_bar(
        x=table["level"].astype(str), y=table["transactions"],
        marker=dict(color=th.SURFACE_3, cornerradius=4,
                    line=dict(color=th.NEON_DIM, width=1)),
        hovertemplate="<b>%{x}</b><br>%{y:,} transactions<extra></extra>",
        showlegend=False,
    )
    fig.update_yaxes(title="transactions")
    fig.update_xaxes(title=None)
    return fig


def fmt_pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:.{digits}f}%"


def panel(html: str) -> None:
    st.markdown(f'<div class="hk-panel">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# masthead
# --------------------------------------------------------------------------- #
df_all = get_data(DATA_PATH)
BASE_ALL = float(df_all[fa.TARGET].mean())

st.markdown(
    '<div class="hk-head">'
    '<div class="t">&gt; FRAUD // CONSOLE<span class="hk-cursor">_</span></div>'
    f'<div class="s">signal hunting on {len(df_all):,} card transactions &nbsp;·&nbsp; '
    f'{int(df_all[fa.TARGET].sum())} confirmed frauds &nbsp;·&nbsp; '
    f'base rate {fmt_pct(BASE_ALL)}</div></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# sidebar filters
# --------------------------------------------------------------------------- #
def reset_filters() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("f_"):
            del st.session_state[k]


with st.sidebar:
    st.markdown("### ▚ FILTERS")
    st.caption("every panel below reacts to these")

    cls = st.multiselect(
        "class", ["fraud", "legit"], default=["fraud", "legit"], key="f_class")

    cats = st.multiselect(
        "merchant category",
        sorted(df_all["merchant_category"].unique()),
        default=sorted(df_all["merchant_category"].unique()),
        key="f_cats",
    )

    amt = st.slider(
        "amount ($)", 0.0, float(df_all["amount"].max()),
        (0.0, float(df_all["amount"].max())), step=10.0, key="f_amt")

    hours = st.slider("hour of day", 0, 23, (0, 23), key="f_hours")

    trust = st.slider("device trust score", 0, 100, (0, 100), key="f_trust")

    vel = st.slider(
        "velocity last 24h", int(df_all["velocity_last_24h"].min()),
        int(df_all["velocity_last_24h"].max()),
        (int(df_all["velocity_last_24h"].min()), int(df_all["velocity_last_24h"].max())),
        key="f_vel")

    age = st.slider(
        "cardholder age", int(df_all["cardholder_age"].min()),
        int(df_all["cardholder_age"].max()),
        (int(df_all["cardholder_age"].min()), int(df_all["cardholder_age"].max())),
        key="f_age")

    foreign = st.radio("foreign transaction", ["any", "yes", "no"],
                       horizontal=True, key="f_foreign")
    mismatch = st.radio("location mismatch", ["any", "yes", "no"],
                        horizontal=True, key="f_mismatch")

    st.button("↺ reset filters", on_click=reset_filters, width="stretch")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["merchant_category"].isin(cats)
        & df["amount"].between(*amt)
        & df["transaction_hour"].between(*hours)
        & df["device_trust_score"].between(*trust)
        & df["velocity_last_24h"].between(*vel)
        & df["cardholder_age"].between(*age)
    )
    if foreign != "any":
        m &= df["foreign_transaction"] == (1 if foreign == "yes" else 0)
    if mismatch != "any":
        m &= df["location_mismatch"] == (1 if mismatch == "yes" else 0)
    wanted = {"fraud": 1, "legit": 0}
    m &= df[fa.TARGET].isin([wanted[c] for c in cls]) if cls else False
    return df[m].copy()


df = apply_filters(df_all)

if df.empty:
    st.error("▛ no transactions match these filters. Widen the range or reset.")
    st.stop()

BASE = float(df[fa.TARGET].mean())
N_FRAUD = int(df[fa.TARGET].sum())

# active-filter readout
chips = []
if len(cats) < df_all["merchant_category"].nunique():
    chips.append("category: " + ", ".join(cats))
if amt != (0.0, float(df_all["amount"].max())):
    chips.append(f"amount {amt[0]:.0f}-{amt[1]:.0f}")
if hours != (0, 23):
    chips.append(f"hour {hours[0]}-{hours[1]}")
if trust != (0, 100):
    chips.append(f"trust {trust[0]}-{trust[1]}")
if foreign != "any":
    chips.append(f"foreign={foreign}")
if mismatch != "any":
    chips.append(f"mismatch={mismatch}")
if len(cls) < 2:
    chips.append("class: " + ", ".join(cls))

st.markdown(
    '<div>' + (
        "".join(f'<span class="hk-tag on">{c}</span>' for c in chips)
        or '<span class="hk-tag">no filters — full portfolio</span>'
    ) + '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
k = st.columns(6)
k[0].metric("transactions", f"{len(df):,}",
            delta=f"{len(df) / len(df_all):.1%} of book", delta_color="off")
k[1].metric("frauds", f"{N_FRAUD:,}")
k[2].metric("fraud rate", fmt_pct(BASE),
            delta=f"{(BASE / BASE_ALL - 1) * 100:+.0f}% vs book" if BASE_ALL else None,
            delta_color="inverse")
k[3].metric("fraud $ exposure", f"${df.loc[df[fa.TARGET] == 1, 'amount'].sum():,.0f}")
k[4].metric("avg fraud amount",
            f"${df.loc[df[fa.TARGET] == 1, 'amount'].mean():,.0f}"
            if N_FRAUD else "—")
k[5].metric("avg legit amount",
            f"${df.loc[df[fa.TARGET] == 0, 'amount'].mean():,.0f}"
            if (df[fa.TARGET] == 0).any() else "—")

if N_FRAUD < 20:
    st.warning(
        f"▛ only {N_FRAUD} frauds survive these filters — rates below are noisy. "
        "Loosen the filters before reading anything into small segments."
    )

tab_overview, tab_signals, tab_hunter, tab_model, tab_data = st.tabs(
    ["◤ OVERVIEW", "◤ SIGNAL SCAN", "◤ SEGMENT HUNTER", "◤ MODEL LAB", "◤ RAW DATA"]
)

# --------------------------------------------------------------------------- #
# TAB 1 - overview
# --------------------------------------------------------------------------- #
with tab_overview:
    c1, c2 = st.columns([3, 2])

    with c1:
        hourly = df.groupby("transaction_hour", observed=False)[fa.TARGET].agg(
            ["count", "sum"])
        hourly.columns = ["transactions", "frauds"]
        hourly["fraud_rate"] = hourly["frauds"] / hourly["transactions"]
        fig = _fig(340, title="fraud rate by hour of day")
        fig.add_scatter(
            x=hourly.index, y=hourly["fraud_rate"], mode="lines+markers",
            line=dict(color=th.SERIES[0], width=2, shape="spline", smoothing=0.6),
            marker=dict(size=8, color=th.SERIES[0],
                        line=dict(color=th.SURFACE, width=2)),
            fill="tozeroy", fillcolor="rgba(44,171,91,0.13)",
            customdata=np.stack([hourly["transactions"], hourly["frauds"]], -1),
            hovertemplate=("hour %{x}:00<br>fraud rate %{y:.2%}<br>"
                           "%{customdata[1]} / %{customdata[0]:,} txns<extra></extra>"),
            showlegend=False,
        )
        baseline_line(fig, BASE, f"base {fmt_pct(BASE)}", position="bottom right")
        fig.update_yaxes(tickformat=".1%", title="fraud rate")
        fig.update_xaxes(title="hour", dtick=2)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, theme=None)

    with c2:
        t = fa.lift_table(df, "merchant_category").sort_values(
            "fraud_rate", ascending=False)
        st.plotly_chart(rate_bars(t, BASE, "fraud rate by merchant category", height=340),
                        theme=None)

    c3, c4 = st.columns(2)

    with c3:
        fig = _fig(330, title="amount distribution — fraud vs legit (share of class)")
        for name, val, color in (("legit", 0, th.LEGIT), ("fraud", 1, th.FRAUD)):
            sub = df.loc[df[fa.TARGET] == val, "amount"]
            if sub.empty:
                continue
            fig.add_histogram(
                x=sub, name=name, histnorm="percent", nbinsx=45,
                marker=dict(color=color, line=dict(color=th.SURFACE, width=1)),
                opacity=0.72,
                hovertemplate=f"{name}<br>$%{{x}}<br>%{{y:.1f}}% of {name}<extra></extra>",
            )
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="amount ($)")
        fig.update_yaxes(title="% of class", ticksuffix="%")
        st.plotly_chart(fig, theme=None)

    with c4:
        pivot = df.pivot_table(
            index="merchant_category", columns="transaction_hour_bin",
            values=fa.TARGET, aggfunc="mean", observed=False)
        counts = df.pivot_table(
            index="merchant_category", columns="transaction_hour_bin",
            values=fa.TARGET, aggfunc="count", observed=False)
        fig = _fig(330, title="fraud rate — category × time of day")
        fig.add_heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns], y=list(pivot.index),
            colorscale=th.GREEN_SCALE, xgap=2, ygap=2,
            customdata=counts.values,
            hovertemplate=("%{y} · %{x}<br>fraud rate %{z:.2%}<br>"
                           "%{customdata:,} txns<extra></extra>"),
            colorbar=dict(
                title=dict(text="rate", font=dict(color=th.INK_DIM, size=11)),
                tickformat=".1%", tickfont=dict(color=th.INK_DIM, size=10),
                outlinewidth=0, thickness=12,
            ),
        )
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        fig.update_layout(margin=dict(l=110, r=24))
        st.plotly_chart(fig, theme=None)

    panel(
        "The two charts on top answer <b>when</b> and <b>where</b> fraud lands; "
        "the bottom two answer <b>how big</b> and <b>which combination</b>. "
        "Anything sitting above the dotted base-rate line is a candidate signal — "
        "take it to <b>SIGNAL SCAN</b> to see how much information it really carries."
    )

# --------------------------------------------------------------------------- #
# TAB 2 - signal scan
# --------------------------------------------------------------------------- #
with tab_signals:
    dims = fa.dimension_columns(df)
    iv = fa.iv_ranking(df, dims)

    left, right = st.columns([2, 3])

    with left:
        st.markdown("#### ▚ predictive power ranking")
        panel(
            "<b>Information Value</b> scores how much a feature separates fraud from "
            "clean traffic across all of its levels. Rule of thumb: "
            "&lt;0.02 noise · 0.02-0.1 weak · 0.1-0.3 medium · 0.3+ strong."
        )
        fig = _fig(max(260, 34 * len(iv)), title="information value by feature")
        fig.add_bar(
            x=iv["iv"], y=iv["label"], orientation="h",
            marker=dict(color=th.SERIES[0], cornerradius=4,
                        line=dict(color=th.SURFACE, width=2)),
            customdata=np.stack([iv["max_lift"], iv["top_level"], iv["strength"]], -1),
            hovertemplate=("<b>%{y}</b><br>IV %{x:.3f} (%{customdata[2]})<br>"
                           "peak level: %{customdata[1]}<br>"
                           "peak lift %{customdata[0]:.1f}x<extra></extra>"),
            text=[f"{v:.2f}" for v in iv["iv"]], textposition="outside",
            textfont=dict(color=th.INK_DIM, size=11, family=th.MONO),
            showlegend=False,
        )
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="information value",
                         range=[0, float(iv["iv"].max()) * 1.18])
        fig.update_layout(margin=dict(l=150, r=40))
        st.plotly_chart(fig, theme=None)
        st.caption(
            "`stacked risk flags` is a derived counter (foreign + mismatch + "
            "night) — it scores high because it bundles three raw signals, so read "
            "it as a summary, not as a fourth independent feature."
        )

        corr = fa.point_biserial(df)
        st.markdown("#### ▚ linear correlation with fraud")
        st.dataframe(
            corr[["feature", "corr"]],
            hide_index=True, height=min(360, 40 + 35 * len(corr)),
            column_config={
                "feature": st.column_config.TextColumn("feature"),
                "corr": st.column_config.NumberColumn(
                    "corr with is_fraud", format="%.3f"),
            },
        )

    with right:
        st.markdown("#### ▚ break a feature open")
        pick = st.selectbox(
            "feature", iv["feature"].tolist(),
            format_func=fa.label_of, key="scan_dim",
        )
        t = fa.lift_table(df, pick)
        st.plotly_chart(
            rate_bars(t, BASE, f"fraud rate by {fa.label_of(pick)}", height=320),
            theme=None)
        st.plotly_chart(
            volume_bars(t, "transaction volume per level", height=190), theme=None)

        show = t.copy()
        show["level"] = show["level"].astype(str)
        st.dataframe(
            show[["level", "transactions", "frauds", "fraud_rate", "lift",
                  "share_of_frauds", "woe", "avg_amount"]],
            hide_index=True, width="stretch",
            column_config={
                "level": st.column_config.TextColumn("level"),
                "transactions": st.column_config.NumberColumn("txns", format="%d"),
                "frauds": st.column_config.NumberColumn("frauds", format="%d"),
                "fraud_rate": st.column_config.NumberColumn("fraud rate", format="percent"),
                "lift": st.column_config.NumberColumn("lift", format="%.2fx"),
                "share_of_frauds": st.column_config.ProgressColumn(
                    "share of all frauds", format="percent",
                    min_value=0.0, max_value=1.0),
                "woe": st.column_config.NumberColumn("WoE", format="%.2f"),
                "avg_amount": st.column_config.NumberColumn("avg $", format="$%.0f"),
            },
        )
        st.caption(
            "fraud rate and share are shown as fractions of 1 in the bar column; "
            "WoE > 0 means the level over-indexes on fraud."
        )

    if not iv.empty:
        top = iv.iloc[0]
        second = iv.iloc[1] if len(iv) > 1 else None
        msg = (f"Strongest separator right now is <b>{top['label']}</b> "
               f"(IV {top['iv']:.2f}); its worst level <b>{top['top_level']}</b> runs at "
               f"<b>{fmt_pct(top['top_level_rate'])}</b> fraud "
               f"({top['max_lift']:.1f}× the {fmt_pct(BASE)} base).")
        if second is not None:
            msg += (f" Next is <b>{second['label']}</b> (IV {second['iv']:.2f}), "
                    f"peaking at <b>{second['top_level']}</b>.")
        panel("▶ " + msg)

# --------------------------------------------------------------------------- #
# TAB 3 - segment hunter
# --------------------------------------------------------------------------- #
with tab_hunter:
    st.markdown("#### ▚ auto-mined segments")
    panel(
        "Every single condition and every profitable pair of conditions, ranked by "
        "<b>lift</b> — how many times more fraud-dense the segment is than the book. "
        "<b>recall</b> is the share of all frauds the rule would catch, "
        "<b>fraud rate</b> is what an analyst reviewing that queue would hit."
    )

    hc = st.columns([1, 1, 1, 3])
    min_support = hc[0].number_input(
        "min transactions", 20, 2000, 60, step=10, key="hunt_support")
    depth = hc[1].selectbox("max conditions", [1, 2], index=1, key="hunt_depth")
    sort_by = hc[2].selectbox("rank by", ["lift", "recall", "frauds"], key="hunt_sort")

    seg = get_segments(df, int(min_support), int(depth))

    if seg.empty:
        st.info("no segment clears the support floor — lower it or widen the filters.")
    else:
        seg = seg.sort_values(sort_by, ascending=False, ignore_index=True)
        fig = _fig(max(300, 30 * min(len(seg), 15)),
                   title=f"top segments by {sort_by}")
        head = seg.head(15).iloc[::-1]
        fig.add_bar(
            x=head["lift"], y=head["rule"], orientation="h",
            marker=dict(color=th.FRAUD, cornerradius=4,
                        line=dict(color=th.SURFACE, width=2)),
            customdata=np.stack([head["fraud_rate"], head["transactions"],
                                 head["frauds"], head["recall"]], -1),
            hovertemplate=("<b>%{y}</b><br>lift %{x:.1f}x<br>"
                           "fraud rate %{customdata[0]:.2%}<br>"
                           "%{customdata[2]} frauds / %{customdata[1]:,} txns<br>"
                           "catches %{customdata[3]:.0%} of all fraud<extra></extra>"),
            text=[f"{v:.1f}x" for v in head["lift"]], textposition="outside",
            textfont=dict(color=th.INK_DIM, size=11, family=th.MONO),
            showlegend=False,
        )
        fig.add_vline(x=1, line=dict(color=th.INK_MUTED, width=1, dash="dot"))
        fig.update_xaxes(title="lift vs base rate")
        fig.update_yaxes(title=None, tickfont=dict(size=10))
        fig.update_layout(margin=dict(l=410, r=70))
        st.plotly_chart(fig, theme=None)

        st.dataframe(
            seg[["rule", "transactions", "frauds", "fraud_rate", "lift",
                 "recall", "coverage", "avg_amount"]],
            hide_index=True, width="stretch", height=340,
            column_config={
                "rule": st.column_config.TextColumn("rule", width="large"),
                "transactions": st.column_config.NumberColumn("txns", format="%d"),
                "frauds": st.column_config.NumberColumn("frauds", format="%d"),
                "fraud_rate": st.column_config.NumberColumn("fraud rate", format="percent"),
                "lift": st.column_config.NumberColumn("lift", format="%.1fx"),
                "recall": st.column_config.ProgressColumn(
                    "% of all fraud caught", format="percent",
                    min_value=0.0, max_value=1.0),
                "coverage": st.column_config.NumberColumn("% of book", format="percent"),
                "avg_amount": st.column_config.NumberColumn("avg $", format="$%.0f"),
            },
        )

    st.divider()
    st.markdown("#### ▚ build your own rule")
    panel(
        "Compose a candidate alert rule and see what it would actually do to the "
        "review queue. Conditions are ANDed together; leave one blank to ignore it."
    )

    r1 = st.columns(4)
    r_foreign = r1[0].selectbox("foreign transaction", ["–", "yes", "no"], key="r_for")
    r_mismatch = r1[1].selectbox("location mismatch", ["–", "yes", "no"], key="r_mis")
    r_trust = r1[2].slider("device trust ≤", 0, 100, 40, key="r_trust")
    r_vel = r1[3].slider("velocity ≥", 0, int(df_all["velocity_last_24h"].max()), 3,
                         key="r_vel")

    r2 = st.columns(4)
    r_hours = r2[0].slider("hour between", 0, 23, (0, 5), key="r_hours")
    r_amount = r2[1].slider("amount ≥ ($)", 0, int(df_all["amount"].max()), 0, step=25,
                            key="r_amount")
    r_cats = r2[2].multiselect("categories", sorted(df_all["merchant_category"].unique()),
                               key="r_cats")
    use = r2[3].multiselect(
        "active conditions",
        ["foreign", "mismatch", "trust", "velocity", "hours", "amount", "category"],
        default=["trust", "hours"], key="r_use",
    )

    mask = pd.Series(True, index=df.index)
    parts = []
    if "foreign" in use and r_foreign != "–":
        mask &= df["foreign_transaction"] == (1 if r_foreign == "yes" else 0)
        parts.append(f"foreign_transaction = {r_foreign}")
    if "mismatch" in use and r_mismatch != "–":
        mask &= df["location_mismatch"] == (1 if r_mismatch == "yes" else 0)
        parts.append(f"location_mismatch = {r_mismatch}")
    if "trust" in use:
        mask &= df["device_trust_score"] <= r_trust
        parts.append(f"device_trust_score <= {r_trust}")
    if "velocity" in use:
        mask &= df["velocity_last_24h"] >= r_vel
        parts.append(f"velocity_last_24h >= {r_vel}")
    if "hours" in use:
        mask &= df["transaction_hour"].between(*r_hours)
        parts.append(f"transaction_hour in [{r_hours[0]}..{r_hours[1]}]")
    if "amount" in use and r_amount > 0:
        mask &= df["amount"] >= r_amount
        parts.append(f"amount >= {r_amount}")
    if "category" in use and r_cats:
        mask &= df["merchant_category"].isin(r_cats)
        parts.append("merchant_category in " + ", ".join(r_cats))

    res = fa.evaluate_rule(df, mask)
    st.code("FLAG WHEN  " + ("\n      AND  ".join(parts) or "<no condition selected>"),
            language="sql")

    m = st.columns(6)
    m[0].metric("alerts raised", f"{res['flagged']:,}",
                delta=f"{res['coverage']:.2%} of book", delta_color="off")
    m[1].metric("frauds caught", f"{res['caught']:,}")
    m[2].metric("precision", fmt_pct(res["precision"], 1))
    m[3].metric("recall", fmt_pct(res["recall"], 1))
    m[4].metric("lift", f"{res['lift']:.1f}x")
    m[5].metric("$ caught", f"${res['amount_caught']:,.0f}",
                delta=f"${res['amount_missed']:,.0f} missed", delta_color="inverse")

    if res["flagged"]:
        fig = _fig(190, title="what the queue looks like")
        fig.add_bar(
            x=[res["caught"]], y=["alert queue"], orientation="h", name="fraud caught",
            marker=dict(color=th.FRAUD, line=dict(color=th.SURFACE, width=2)),
            hovertemplate="fraud caught: %{x:,}<extra></extra>",
        )
        fig.add_bar(
            x=[res["false_alarms"]], y=["alert queue"], orientation="h",
            name="false alarms",
            marker=dict(color=th.SURFACE_3, line=dict(color=th.SURFACE, width=2)),
            hovertemplate="false alarms: %{x:,}<extra></extra>",
        )
        fig.add_bar(
            x=[res["missed"]], y=["missed fraud"], orientation="h", name="fraud missed",
            marker=dict(color=th.WARN, line=dict(color=th.SURFACE, width=2)),
            hovertemplate="fraud missed: %{x:,}<extra></extra>",
        )
        fig.update_layout(barmode="stack", margin=dict(l=110, r=24, t=44, b=24))
        fig.update_xaxes(title="transactions")
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, theme=None)

# --------------------------------------------------------------------------- #
# TAB 4 - model lab
# --------------------------------------------------------------------------- #
with tab_model:
    panel(
        "Both models are scored <b>out of fold</b> (5-fold stratified CV), so the "
        "curves below are what you would get on unseen transactions, not a memorised "
        "training fit. Importances come from a fit on the full selection."
    )
    res = get_models(df)

    if not res["ok"]:
        st.warning(f"▛ model not trained — {res['reason']}. Widen the filters.")
    else:
        rf, lr = res["curves"]["random forest"], res["curves"]["logistic"]
        mk = st.columns(4)
        mk[0].metric("ROC AUC · forest", f"{rf['auc']:.3f}")
        mk[1].metric("PR AUC · forest", f"{rf['ap']:.3f}",
                     delta=f"vs {res['base_rate']:.3f} random", delta_color="off")
        mk[2].metric("ROC AUC · logistic", f"{lr['auc']:.3f}")
        mk[3].metric("PR AUC · logistic", f"{lr['ap']:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = _fig(340, title="ROC — out-of-fold")
            for i, (name, c) in enumerate(res["curves"].items()):
                fig.add_scatter(
                    x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} ({c['auc']:.3f})",
                    line=dict(color=th.SERIES[i], width=2),
                    hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>")
            fig.add_scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="random",
                line=dict(color=th.INK_MUTED, width=1, dash="dot"),
                hoverinfo="skip")
            fig.update_xaxes(title="false positive rate")
            fig.update_yaxes(title="true positive rate")
            st.plotly_chart(fig, theme=None)

        with c2:
            fig = _fig(340, title="precision / recall — out-of-fold")
            for i, (name, c) in enumerate(res["curves"].items()):
                fig.add_scatter(
                    x=c["recall"], y=c["precision"], mode="lines",
                    name=f"{name} ({c['ap']:.3f})",
                    line=dict(color=th.SERIES[i], width=2),
                    hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>")
            fig.add_hline(
                y=res["base_rate"], line=dict(color=th.INK_MUTED, width=1, dash="dot"),
                annotation_text=f"random {res['base_rate']:.2%}",
                annotation_font=dict(color=th.INK_MUTED, size=10, family=th.MONO))
            fig.update_xaxes(title="recall")
            fig.update_yaxes(title="precision")
            st.plotly_chart(fig, theme=None)

        c3, c4 = st.columns([3, 2])
        with c3:
            imp = res["importance"].head(12).iloc[::-1]
            fig = _fig(380, title="what the forest actually leans on")
            fig.add_bar(
                x=imp["rf_importance"], y=imp["feature"], orientation="h",
                marker=dict(color=th.SERIES[0], cornerradius=4,
                            line=dict(color=th.SURFACE, width=2)),
                customdata=imp["odds_ratio"],
                hovertemplate=("<b>%{y}</b><br>importance %{x:.3f}<br>"
                               "logistic odds ratio %{customdata:.2f}<extra></extra>"),
                text=[f"{v:.3f}" for v in imp["rf_importance"]], textposition="outside",
                textfont=dict(color=th.INK_DIM, size=11, family=th.MONO),
                showlegend=False)
            fig.update_xaxes(title="mean decrease in impurity",
                             range=[0, float(imp["rf_importance"].max()) * 1.18])
            fig.update_yaxes(title=None)
            fig.update_layout(margin=dict(l=170, r=50))
            st.plotly_chart(fig, theme=None)

        with c4:
            st.markdown("#### ▚ direction of effect")
            st.dataframe(
                res["importance"][["feature", "odds_ratio", "rf_importance"]],
                hide_index=True, height=380, width="stretch",
                column_config={
                    "feature": st.column_config.TextColumn("feature"),
                    "odds_ratio": st.column_config.NumberColumn(
                        "odds ratio", format="%.2f",
                        help=">1 pushes toward fraud, <1 away (per 1 SD)"),
                    "rf_importance": st.column_config.ProgressColumn(
                        "forest weight", format="%.3f", min_value=0.0,
                        max_value=float(res["importance"]["rf_importance"].max())),
                },
            )

        st.divider()
        st.markdown("#### ▚ where would you set the cutoff?")
        tt = fa.threshold_table(df[fa.TARGET].to_numpy(), res["rf_score"],
                                df["amount"].to_numpy())
        fig = _fig(340, title="review budget vs fraud caught")
        fig.add_scatter(
            x=tt["review_rate"], y=tt["recall"], mode="lines", name="recall (fraud caught)",
            line=dict(color=th.SERIES[0], width=2),
            hovertemplate="review %{x:.1%}<br>recall %{y:.1%}<extra></extra>")
        fig.add_scatter(
            x=tt["review_rate"], y=tt["precision"], mode="lines",
            name="precision (queue hit rate)",
            line=dict(color=th.SERIES[1], width=2),
            hovertemplate="review %{x:.1%}<br>precision %{y:.1%}<extra></extra>")
        fig.update_xaxes(title="share of transactions sent to review", tickformat=".0%")
        fig.update_yaxes(title="rate", tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, theme=None)

        budget = st.slider("review budget — % of transactions", 0.2, 30.0, 2.0, 0.2,
                           format="%.1f%%", key="m_budget")
        row = tt.iloc[(tt["review_rate"] - budget / 100).abs().argmin()]
        b = st.columns(5)
        b[0].metric("alerts / 10k txns", f"{row['alerts'] / len(df) * 10000:,.0f}")
        b[1].metric("frauds caught", f"{int(row['caught']):,}")
        b[2].metric("precision", fmt_pct(row["precision"], 1))
        b[3].metric("recall", fmt_pct(row["recall"], 1))
        b[4].metric("$ recovered", f"${row['amount_recovered']:,.0f}",
                    delta=f"of ${row['amount_at_risk']:,.0f}", delta_color="off")

# --------------------------------------------------------------------------- #
# TAB 5 - raw data
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("#### ▚ filtered transactions")
    only_fraud = st.toggle("show confirmed frauds only", value=False, key="d_fraud")
    view = df[df[fa.TARGET] == 1] if only_fraud else df
    cols = [fa.ID_COL, "amount", "transaction_hour", "merchant_category",
            "foreign_transaction", "location_mismatch", "device_trust_score",
            "velocity_last_24h", "cardholder_age", "risk_flags", fa.TARGET]
    st.caption(f"{len(view):,} rows · {int(view[fa.TARGET].sum())} frauds")
    st.dataframe(
        view[cols], hide_index=True, height=520, width="stretch",
        column_config={
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
            "device_trust_score": st.column_config.ProgressColumn(
                "device trust", format="%d", min_value=0, max_value=100),
            "is_fraud": st.column_config.CheckboxColumn("fraud"),
        },
    )
    st.download_button(
        "⤓ export this selection (csv)",
        view[cols].to_csv(index=False).encode("utf-8"),
        file_name="fraud_selection.csv", mime="text/csv",
    )
