"""Analytics engine for the fraud console.

Pure pandas/sklearn - no streamlit imports here, so every function is testable
on its own and cheap to cache from the UI layer.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "is_fraud"
ID_COL = "transaction_id"
CATEGORICAL = ["merchant_category"]
BINARY = ["foreign_transaction", "location_mismatch"]
NUMERIC = [
    "amount", "transaction_hour", "device_trust_score",
    "velocity_last_24h", "cardholder_age",
]

# Explicit, human-readable edges beat quantile bins here: an analyst has to be
# able to turn a bin straight into a rule.
BIN_SPEC = {
    "amount": ([-0.01, 50, 100, 200, 350, 600, 1000, np.inf],
               ["0-50", "50-100", "100-200", "200-350", "350-600", "600-1k", "1k+"]),
    "transaction_hour": ([-0.01, 5, 11, 16, 20, 23],
                         ["00-05 night", "06-11 morning", "12-16 midday",
                          "17-20 evening", "21-23 late"]),
    "device_trust_score": ([-0.01, 20, 40, 60, 80, 100],
                           ["0-20 burner", "20-40 low", "40-60 mid",
                            "60-80 good", "80-100 trusted"]),
    "velocity_last_24h": ([-0.01, 0, 1, 2, 3, 5, np.inf],
                          ["0", "1", "2", "3", "4-5", "6+"]),
    "cardholder_age": ([-0.01, 25, 35, 45, 55, 65, np.inf],
                       ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, (edges, labels) in BIN_SPEC.items():
        if col in df.columns:
            df[f"{col}_bin"] = pd.cut(df[col], bins=edges, labels=labels, ordered=True)
    df["night_flag"] = df["transaction_hour"].between(0, 5).astype(int)
    df["risk_flags"] = (
        df["foreign_transaction"] + df["location_mismatch"] + df["night_flag"]
    )
    return df


def binned_columns(df: pd.DataFrame) -> list:
    return [f"{c}_bin" for c in BIN_SPEC if f"{c}_bin" in df.columns]


def dimension_columns(df: pd.DataFrame) -> list:
    """Every column usable as a discrete breakdown."""
    return CATEGORICAL + BINARY + ["night_flag", "risk_flags"] + binned_columns(df)


PRETTY = {
    "merchant_category": "merchant category",
    "foreign_transaction": "foreign transaction",
    "location_mismatch": "location mismatch",
    "night_flag": "night (00-05)",
    "risk_flags": "stacked risk flags",
    "amount_bin": "amount",
    "transaction_hour_bin": "hour of day",
    "device_trust_score_bin": "device trust",
    "velocity_last_24h_bin": "velocity 24h",
    "cardholder_age_bin": "cardholder age",
}


def label_of(col: str) -> str:
    return PRETTY.get(col, col.replace("_", " "))


# --------------------------------------------------------------------------- #
# lift / WoE / IV - the "which feature predicts fraud" core
# --------------------------------------------------------------------------- #
def lift_table(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Fraud rate, lift vs base rate, WoE and IV contribution per level of `dim`."""
    base = df[TARGET].mean()
    g = df.groupby(dim, observed=False)[TARGET].agg(["count", "sum"])
    g.columns = ["transactions", "frauds"]
    g = g[g["transactions"] > 0].copy()
    g["clean"] = g["transactions"] - g["frauds"]
    g["fraud_rate"] = g["frauds"] / g["transactions"]
    g["lift"] = g["fraud_rate"] / base if base > 0 else np.nan
    total_f = max(int(g["frauds"].sum()), 1)
    total_c = max(int(g["clean"].sum()), 1)
    # 0.5 smoothing keeps WoE finite for bins that hold zero frauds
    pct_f = (g["frauds"] + 0.5) / (total_f + 0.5 * len(g))
    pct_c = (g["clean"] + 0.5) / (total_c + 0.5 * len(g))
    g["share_of_frauds"] = g["frauds"] / total_f
    g["woe"] = np.log(pct_f / pct_c)
    g["iv"] = (pct_f - pct_c) * g["woe"]
    g["avg_amount"] = df.groupby(dim, observed=False)["amount"].mean().reindex(g.index)
    return g.reset_index().rename(columns={dim: "level"})


def iv_ranking(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    """Information Value per dimension - the headline 'what predicts fraud' list."""
    rows = []
    for d in dims:
        t = lift_table(df, d)
        if t.empty:
            continue
        rows.append({
            "feature": d,
            "label": label_of(d),
            "iv": float(t["iv"].sum()),
            "levels": int(len(t)),
            "max_lift": float(t["lift"].max()),
            "top_level": str(t.loc[t["lift"].idxmax(), "level"]),
            "top_level_rate": float(t["fraud_rate"].max()),
        })
    out = pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
    if out.empty:
        return out
    out["strength"] = pd.cut(
        out["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["noise", "weak", "medium", "strong", "very strong"],
    ).astype(str)
    return out


def point_biserial(df: pd.DataFrame) -> pd.DataFrame:
    """Linear correlation of every numeric/binary column with the fraud label."""
    cols = [c for c in NUMERIC + BINARY + ["night_flag", "risk_flags"] if c in df.columns]
    y = df[TARGET]
    rows = [{"feature": c, "corr": float(df[c].corr(y))} for c in cols]
    out = pd.DataFrame(rows)
    out["abs_corr"] = out["corr"].abs()
    return out.sort_values("abs_corr", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# segment mining - "find the rule that catches fraud"
# --------------------------------------------------------------------------- #
def _atomic_conditions(df: pd.DataFrame) -> list:
    conds = []
    for c in BINARY + ["night_flag"]:
        if c in df.columns:
            conds.append((c, f"{label_of(c)} = yes", df[c] == 1))
    for c in CATEGORICAL:
        if c in df.columns:
            for lvl in sorted(df[c].dropna().unique()):
                conds.append((c, f"{label_of(c)} = {lvl}", df[c] == lvl))
    for c in binned_columns(df):
        for lvl in df[c].cat.categories:
            mask = df[c] == lvl
            if bool(mask.any()):
                conds.append((c, f"{label_of(c)} = {lvl}", mask))
    return conds


def mine_segments(df: pd.DataFrame, min_support: int = 60, max_depth: int = 2,
                  top_k_singles: int = 14, limit: int = 40) -> pd.DataFrame:
    """Rank single and combined conditions by how much they concentrate fraud."""
    base = df[TARGET].mean()
    n_all = len(df)
    if n_all == 0 or base == 0:
        return pd.DataFrame()
    y = df[TARGET].to_numpy()
    amounts = df["amount"].to_numpy()
    total_f = int(y.sum())

    def score(name, mask, depth):
        n = int(mask.sum())
        if n < min_support:
            return None
        f = int(y[mask].sum())
        rate = f / n
        return {
            "rule": name, "depth": depth, "transactions": n,
            "coverage": n / n_all, "frauds": f, "fraud_rate": rate,
            "lift": rate / base, "recall": f / total_f if total_f else 0.0,
            "avg_amount": float(amounts[mask].mean()),
        }

    # night_flag and the 00-05 hour bin select the same rows, as can other
    # pairs; a signature keeps one copy of each distinct segment.
    seen = set()

    singles = []
    for col, name, m in _atomic_conditions(df):
        arr = m.to_numpy()
        sig = arr.tobytes()
        if sig in seen:
            continue
        s = score(name, arr, 1)
        if s:
            seen.add(sig)
            singles.append((col, s, arr))
    singles.sort(key=lambda t: t[1]["lift"], reverse=True)

    results = [s for _, s, _ in singles]
    if max_depth >= 2:
        pool = singles[:top_k_singles]
        for (col_a, a, ma), (col_b, b, mb) in itertools.combinations(pool, 2):
            if col_a == col_b:  # two levels of one feature never co-occur
                continue
            both = ma & mb
            sig = both.tobytes()
            if sig in seen:
                continue
            s = score(a["rule"] + "  AND  " + b["rule"], both, 2)
            if s and s["lift"] > max(a["lift"], b["lift"]) * 1.05:
                seen.add(sig)
                results.append(s)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    out = out.sort_values(["lift", "recall"], ascending=False, ignore_index=True)
    return out.head(limit)


def evaluate_rule(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Precision/recall/lift of an analyst-built rule, framed as an alert queue."""
    base = df[TARGET].mean()
    n = len(df)
    total_f = int(df[TARGET].sum())
    flagged = int(mask.sum())
    caught = int(df.loc[mask, TARGET].sum())
    return {
        "flagged": flagged,
        "coverage": flagged / n if n else 0.0,
        "caught": caught,
        "missed": total_f - caught,
        "precision": caught / flagged if flagged else 0.0,
        "recall": caught / total_f if total_f else 0.0,
        "lift": (caught / flagged) / base if flagged and base else 0.0,
        "false_alarms": flagged - caught,
        "amount_caught": float(df.loc[mask & (df[TARGET] == 1), "amount"].sum()),
        "amount_missed": float(df.loc[~mask & (df[TARGET] == 1), "amount"].sum()),
    }


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def model_matrix(df: pd.DataFrame):
    cols = [c for c in NUMERIC + BINARY + ["night_flag"] if c in df.columns]
    X = df[cols].copy()
    dummies = pd.get_dummies(df["merchant_category"], prefix="mcc", dtype=int)
    X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X, df[TARGET].reset_index(drop=True)


def train_and_score(df: pd.DataFrame, seed: int = 7) -> dict:
    """Out-of-fold scoring: honest curves, plus importances fitted on the full table."""
    X, y = model_matrix(df)
    if y.nunique() < 2 or int(y.sum()) < 25:
        return {"ok": False, "reason": "needs both classes and at least 25 frauds"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    rf_oof = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    lr_oof = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]

    rf.fit(X, y)
    lr.fit(X, y)
    coefs = lr[-1].coef_[0]

    importance = pd.DataFrame({
        "feature": list(X.columns),
        "rf_importance": rf.feature_importances_,
        "logit_coef": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("rf_importance", ascending=False, ignore_index=True)

    curves = {}
    for name, s in (("random forest", rf_oof), ("logistic", lr_oof)):
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        curves[name] = {
            "fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec,
            "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
        }

    return {
        "ok": True, "columns": list(X.columns), "importance": importance,
        "curves": curves, "rf_score": rf_oof, "lr_score": lr_oof,
        "base_rate": float(y.mean()),
    }


def threshold_table(y, score, amounts, steps: int = 60) -> pd.DataFrame:
    """Alert-queue economics across score cutoffs."""
    y = np.asarray(y)
    score = np.asarray(score)
    amounts = np.asarray(amounts)
    rows = []
    total_f = int(y.sum())
    total_amt = float(amounts[y == 1].sum())
    # geometric spacing on the review rate: the interesting decisions all live
    # in the top few percent of the queue, not spread evenly to 20%.
    for rate in np.geomspace(0.002, 0.30, steps):
        thr = float(np.quantile(score, 1.0 - rate))
        m = score >= thr
        flagged = int(m.sum())
        if flagged == 0:
            continue
        caught = int(y[m].sum())
        rows.append({
            "review_rate": flagged / len(y),
            "threshold": thr,
            "alerts": flagged,
            "caught": caught,
            "precision": caught / flagged,
            "recall": caught / total_f if total_f else 0.0,
            "amount_recovered": float(amounts[m & (y == 1)].sum()),
            "amount_at_risk": total_amt,
        })
    return pd.DataFrame(rows).drop_duplicates(subset="alerts", ignore_index=True)
