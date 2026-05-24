"""Uplift + fairness + PNG charts for the deck. Run after train."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from retainiq import artifacts, config, data, fairness, features, uplift  # noqa: E402

CHART_DIR = artifacts.chart_dir()
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def chart_cost_curve():
    df = pd.read_csv(config.DATA_PROCESSED / "cost_curve.csv")
    info = json.loads((config.DATA_PROCESSED / "cost_optimal_threshold.json").read_text())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["threshold"], df["total_cost_inr"] / 1000, color="#0b5394", lw=2)
    ax.fill_between(
        df["threshold"], 0, df["total_cost_inr"] / 1000, color="#0b5394", alpha=0.08
    )

    opt_t = info["threshold"]
    opt_c = info["total_cost_inr"] / 1000
    naive_c = info["naive_threshold_cost_inr"] / 1000
    theo_t = info["theoretical_optimal_threshold"]

    ax.axvline(0.5, color="#cc0000", ls="--", lw=1.4, alpha=0.85)
    ax.axvline(opt_t, color="#2e7d32", ls="--", lw=1.4)
    ax.axvline(theo_t, color="#666666", ls=":", lw=1.2)

    ax.scatter([0.5], [naive_c], color="#cc0000", s=80, zorder=5)
    ax.scatter([opt_t], [opt_c], color="#2e7d32", s=90, zorder=5)

    ax.annotate(
        f"default 0.5\nINR {naive_c:,.1f}k",
        xy=(0.5, naive_c),
        xytext=(0.55, naive_c * 0.9),
        color="#cc0000",
        fontsize=10,
        weight="bold",
    )
    ax.annotate(
        f"our threshold {opt_t:.3f}\nINR {opt_c:,.1f}k",
        xy=(opt_t, opt_c),
        xytext=(opt_t + 0.05, opt_c + 30),
        color="#2e7d32",
        fontsize=10,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1),
    )

    saved = naive_c - opt_c
    pct = info["savings_pct_vs_naive"]
    ax.set_title(
        f"Same model, different cutoff — INR {saved:,.0f}k saved ({pct:.1f}%)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel("Threshold on P(churn)")
    ax.set_ylabel("Total cost (INR thousands)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(df["total_cost_inr"] / 1000) * 1.1)
    fig.savefig(CHART_DIR / "08_cost_curve.png")
    plt.close(fig)
    print(f"  {CHART_DIR / '08_cost_curve.png'}")


def chart_feature_importance():
    imp_path = artifacts.feature_importances_path()
    if not imp_path.is_file():
        print(f"  skip 09_feature_importance.png — run train first (missing {imp_path.name})")
        return
    df = pd.read_csv(imp_path).head(12).iloc[::-1]
    pretty = df["feature"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(pretty, df["lgb_gain_pct"], color="#0b5394", alpha=0.88)
    for bar, val in zip(bars, df["lgb_gain_pct"]):
        ax.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
            color="#333",
        )

    ax.set_title("Top features (mean LightGBM gain)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Share of gain (%)")
    ax.set_xlim(0, df["lgb_gain_pct"].max() * 1.15)
    fig.savefig(CHART_DIR / "09_feature_importance.png")
    plt.close(fig)
    print(f"  {CHART_DIR / '09_feature_importance.png'}")


def run_uplift_and_chart():
    print("uplift...")
    df_train = data.load_train()
    X_raw, y = data.split_features_target(df_train)
    fe_state = joblib.load(config.DATA_PROCESSED / "fe_state.joblib")
    X_fe = features.transform(X_raw, fe_state)

    t_learner = uplift.fit_t_learner(X_fe, y)
    uplift.write_propensity_summary(
        X_fe, artifacts.uplift_propensity_summary_path()
    )
    cate = uplift.estimate_cate(t_learner, X_fe)
    print(f"  CATE mean={cate.mean():.4f}, std={cate.std():.4f}")

    oof = pd.read_parquet(config.DATA_PROCESSED / "oof_predictions.parquet")
    churn_proba = oof["p_calibrated"].values

    seg_report = uplift.segmentation_report(cate, churn_proba)
    seg_report.to_csv(config.DATA_PROCESSED / "uplift_segmentation.csv", index=False)
    print(seg_report.to_string(index=False))

    per_customer = pd.DataFrame(
        {
            "customer_id": df_train[config.ID_COL].values,
            "churn_actual": y.values,
            "churn_proba": churn_proba,
            "cate": cate,
            "segment": uplift.segment_customers(cate, churn_proba).values,
        }
    )
    per_customer.to_csv(config.DATA_PROCESSED / "uplift_per_customer.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {
        "persuadable": "#2e7d32",
        "sure-thing": "#1565c0",
        "lost-cause": "#cc0000",
        "sleeping-dog": "#ef6c00",
        "other": "#9e9e9e",
    }
    for seg, color in palette.items():
        mask = per_customer["segment"] == seg
        if mask.sum() == 0:
            continue
        ax.scatter(
            per_customer.loc[mask, "churn_proba"],
            per_customer.loc[mask, "cate"],
            s=8,
            alpha=0.45,
            color=color,
            label=f"{seg} (n={int(mask.sum())})",
        )
    ax.axhline(0, color="#444", lw=0.7)
    ax.axvline(0.5, color="#444", lw=0.7, ls=":")
    ax.set_xlabel("P(churn)")
    ax.set_ylabel("CATE (offer effect on staying)")
    ax.set_title("Who is actually worth calling?", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    fig.savefig(CHART_DIR / "10_uplift_quadrant.png")
    plt.close(fig)
    print(f"  {CHART_DIR / '10_uplift_quadrant.png'}")
    return per_customer


def run_fairness_and_chart(per_customer: pd.DataFrame):
    print("fairness...")
    info = json.loads((config.DATA_PROCESSED / "cost_optimal_threshold.json").read_text())
    threshold = float(info["threshold"])
    df_train = data.load_train()
    eval_df = pd.DataFrame(
        {
            config.ID_COL: df_train[config.ID_COL].values,
            "y_true": per_customer["churn_actual"].values,
            "y_pred": (per_customer["churn_proba"].values >= threshold).astype(int),
        }
    )
    for col in config.SENSITIVE_COLS:
        if col in df_train.columns:
            eval_df[col] = df_train[col].values

    report = fairness.fairness_report(eval_df)
    report["per_group"].to_csv(config.DATA_PROCESSED / "fairness_per_group.csv", index=False)
    report["summary"].to_csv(config.DATA_PROCESSED / "fairness_summary.csv", index=False)
    print(report["summary"].to_string(index=False))

    pg = report["per_group"]
    if pg.empty:
        return

    fig, axes = plt.subplots(1, len(config.SENSITIVE_COLS), figsize=(11, 4.5), sharey=True)
    if len(config.SENSITIVE_COLS) == 1:
        axes = [axes]
    for ax, attr in zip(axes, config.SENSITIVE_COLS):
        sub = pg[pg["group_attribute"] == attr]
        if sub.empty:
            continue
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub["predicted_positive_rate"], w, color="#0b5394", label="Flag rate")
        ax.bar(x + w / 2, sub["tpr"], w, color="#2e7d32", label="TPR")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group_value"], rotation=30, ha="right")
        ax.set_title(attr.title(), fontsize=12, weight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Fairness by group", fontsize=13, weight="bold")
    fig.savefig(CHART_DIR / "12_fairness.png")
    plt.close(fig)
    print(f"  {CHART_DIR / '12_fairness.png'}")


def main():
    artifacts.ensure_dirs()
    if not artifacts.trained_bundle_exists():
        raise FileNotFoundError(
            "Trained artifacts missing. Run: python -m scripts.train"
        )
    print("charts...")
    chart_cost_curve()
    chart_feature_importance()
    per_customer = run_uplift_and_chart()
    run_fairness_and_chart(per_customer)
    print("done.")


if __name__ == "__main__":
    main()
