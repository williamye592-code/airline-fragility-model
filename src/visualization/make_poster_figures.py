from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Figure 1: Pipeline diagram
# =========================
def plot_pipeline():
    steps = [
        "Sequence Data",
        "Feature\nEngineering",
        "Risk Model\n(XGBoost)",
        "Severity\nProxy β",
        "Cost Model",
        "Optimization\n(argmin)",
        "Recommended\nHold"
    ]

    fig, ax = plt.subplots(figsize=(12, 2.8))

    xs = list(range(len(steps)))
    y = 0

    for i, step in enumerate(steps):
        ax.text(
            xs[i], y, step,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
        )

    for i in range(len(steps) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.4, y),
            xytext=(xs[i] + 0.4, y),
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

    ax.set_xlim(-0.8, len(steps) - 0.2)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pipeline_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 2: Sensitivity curve
# Source data: outputs/tables/sensitivity_summary.csv
# Required columns:
#   disruption_cost
#   avg_recommended_hold
# =========================
def plot_sensitivity_curve():
    df = pd.read_csv(TABLES_DIR / "sensitivity_summary.csv")

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        df["disruption_cost"],
        df["avg_recommended_hold"],
        marker="o"
    )
    plt.xlabel("Disruption Cost (USD)")
    plt.ylabel("Average Recommended Hold (minutes)")
    plt.title("Sensitivity of Hold Policy to Disruption Cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sensitivity_curve.png", dpi=300)
    plt.close()


# =========================
# Figure 3: Policy distribution stacked bar
# Source data: outputs/tables/hold_distribution_summary.csv
# Required columns:
#   disruption_cost
#   hold_0_pct
#   hold_5_pct
#   hold_10_pct
#   hold_15_pct
# =========================
def plot_policy_distribution_stacked():
    df = pd.read_csv(TABLES_DIR / "hold_distribution_summary.csv")
    df = df.sort_values("disruption_cost")

    x = df["disruption_cost"].astype(str)

    hold_0 = df["hold_0_pct"]
    hold_5 = df["hold_5_pct"]
    hold_10 = df["hold_10_pct"]
    hold_15 = df["hold_15_pct"]

    plt.figure(figsize=(7, 5))
    plt.bar(x, hold_0, label="Hold 0")
    plt.bar(x, hold_5, bottom=hold_0, label="Hold 5")
    plt.bar(x, hold_10, bottom=hold_0 + hold_5, label="Hold 10")
    plt.bar(x, hold_15, bottom=hold_0 + hold_5 + hold_10, label="Hold 15")

    plt.xlabel("Disruption Cost (USD)")
    plt.ylabel("Share of Recommended Actions")
    plt.title("Policy Distribution Across Cost Scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "policy_distribution_stacked.png", dpi=300)
    plt.close()


# =========================
# Figure 4: Policy comparison bar
# Source data: outputs/tables/policy_comparison_summary.csv
# Required columns:
#   policy
#   avg_expected_cost
# =========================
def plot_policy_comparison_bar():
    df = pd.read_csv(TABLES_DIR / "policy_comparison_summary.csv")

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(df["policy"], df["avg_expected_cost"])
    plt.xlabel("Policy")
    plt.ylabel("Average Expected Cost")
    plt.title("Policy Comparison Under Main Cost Scenario")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "policy_comparison_bar.png", dpi=300)
    plt.close()


# =========================
# Figure 5 (optional but strong): Single-case cost curve
# Source data: create from CLI or manually define below
# =========================
def plot_single_case_cost_curve():
    # Replace these with your chosen sample's cost-by-minute output
    holds = list(range(16))
    costs = [
        5760.30, 5710.06, 5704.85, 5744.52,
        5828.93, 5957.94, 6131.41, 6349.21,
        6611.21, 6917.30, 7267.34, 7661.22,
        8098.83, 8580.05, 9104.79, 9672.93
    ]

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(holds, costs, marker="o")
    plt.xlabel("Hold Time (minutes)")
    plt.ylabel("Expected Cost")
    plt.title("Single-Sequence Cost Curve")

    min_idx = costs.index(min(costs))
    plt.scatter([holds[min_idx]], [costs[min_idx]], s=60)
    plt.annotate(
        f"Optimal = {holds[min_idx]} min",
        (holds[min_idx], costs[min_idx]),
        textcoords="offset points",
        xytext=(8, 8)
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "single_case_cost_curve.png", dpi=300)
    plt.close()


def main():
    plot_pipeline()
    plot_sensitivity_curve()
    plot_policy_distribution_stacked()
    plot_policy_comparison_bar()
    plot_single_case_cost_curve()

    print("Saved figures to:", PLOTS_DIR)


if __name__ == "__main__":
    main()